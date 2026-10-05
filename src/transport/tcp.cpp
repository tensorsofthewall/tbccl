#include <atomic>
#include <mutex>
#include <tbccl/error.hpp>
#include <tbccl/tcp.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <algorithm>
#include <poll.h>
#include <sys/uio.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "latency_trace.hpp"

namespace tbccl
{
namespace
{

    void apply_tcp_nodelay(int fd)
    {
        int enabled = 1;

        if (::setsockopt(
                fd,
                IPPROTO_TCP,
                TCP_NODELAY,
                &enabled,
                sizeof(enabled)) != 0)
        {

            throw Error(ErrorCode::TransportError, 
                "setsockopt(TCP_NODELAY) failed: " +
                std::string(std::strerror(errno)));
        }
    }

    void apply_busy_poll(int fd, int microseconds)
    {
#if defined(__linux__) && defined(SO_BUSY_POLL)
        if (microseconds <= 0)
        {
            return;
        }

        if (::setsockopt(
                fd,
                SOL_SOCKET,
                SO_BUSY_POLL,
                &microseconds,
                sizeof(microseconds)) != 0)
        {
            throw Error(ErrorCode::TransportError, 
                "setsockopt(SO_BUSY_POLL) failed: " +
                std::string(std::strerror(errno)));
        }
#else
        (void)fd;
        (void)microseconds;
#endif
    }

    void apply_options(int fd, const TcpOptions &options)
    {
        if (options.tcp_nodelay)
        {
            apply_tcp_nodelay(fd);
        }

        apply_busy_poll(fd, options.busy_poll_us);
    }

    std::string format_peer(int fd)
    {
        sockaddr_in address{};
        socklen_t length = sizeof(address);

        if (::getpeername(
                fd,
                reinterpret_cast<sockaddr *>(&address),
                &length) != 0)
        {
            return "unknown";
        }

        char text[INET_ADDRSTRLEN]{};

        ::inet_ntop(
            AF_INET,
            &address.sin_addr,
            text,
            sizeof(text));

        return std::string(text) + ":" +
               std::to_string(ntohs(address.sin_port));
    }

    class TcpConnection final : public Connection
    {
    public:
        explicit TcpConnection(int fd) : fd_(fd) {}

        ~TcpConnection() override
        {
            if (fd_ >= 0)
            {
                ::close(fd_);
            }
        }

        // shutdown(SHUT_RDWR) wakes blocked send/recv on every platform without releasing the descriptor
        // (close() from another thread could let the fd number be reused under a still-running syscall).
        // The fd is closed exactly once, in the destructor.
        void abort(const std::string &reason) override
        {
            if (aborted_.exchange(true)) return;
            {
                std::lock_guard<std::mutex> lock(reason_mutex_);
                reason_ = reason;
            }
            if (fd_ >= 0) ::shutdown(fd_, SHUT_RDWR);
        }

        // One sendmsg()/recvmsg() carries the header and the payload (looping over partial progress), so a framed message is one
        // system call and, for small payloads, one TCP segment.
        void send_framed(const void *header, std::size_t header_bytes, const void *data, std::size_t bytes) override
        {
            check_aborted();
            iovec iov[2];
            iov[0].iov_base = const_cast<void *>(header);
            iov[0].iov_len = header_bytes;
            iov[1].iov_base = const_cast<void *>(data);
            iov[1].iov_len = bytes;
            std::size_t index = 0;
            detail::lat_event(detail::kLatSendEnter, detail::tl_lat_current_id);
            while (index < 2)
            {
                if (iov[index].iov_len == 0)
                {
                    ++index;
                    continue;
                }
                msghdr message{};
                message.msg_iov = &iov[index];
                message.msg_iovlen = 2 - index;
#ifdef MSG_NOSIGNAL
                const ssize_t n = ::sendmsg(fd_, &message, MSG_NOSIGNAL);
#else
                const ssize_t n = ::sendmsg(fd_, &message, 0);
#endif
                if (n < 0)
                {
                    if (errno == EINTR) continue;
                    check_aborted();
                    if (errno == EAGAIN || errno == EWOULDBLOCK) throw Error(ErrorCode::Timeout, "timeout: send timed out");
                    throw Error(ErrorCode::TransportError, "send failed: " + std::string(std::strerror(errno)));
                }
                if (n == 0) throw Error(ErrorCode::TransportError, "send returned 0");
                advance(iov, index, static_cast<std::size_t>(n));
            }
            detail::lat_event(detail::kLatSendReturn, detail::tl_lat_current_id);
        }

        void recv_framed(
            void *header, std::size_t header_bytes, void *data, std::size_t bytes, const std::function<void(const void *)> &validate) override
        {
            check_aborted();
            bool validated = false;
            bool first_bytes = false;
            iovec iov[2];
            iov[0].iov_base = header;
            iov[0].iov_len = header_bytes;
            iov[1].iov_base = data;
            iov[1].iov_len = bytes;
            std::size_t index = 0;
            while (index < 2)
            {
                if (iov[index].iov_len == 0)
                {
                    ++index;
                    continue;
                }
                msghdr message{};
                message.msg_iov = &iov[index];
                message.msg_iovlen = 2 - index;
                const ssize_t n = ::recvmsg(fd_, &message, 0);
                if (n < 0)
                {
                    if (errno == EINTR) continue;
                    check_aborted();
                    if (errno == EAGAIN || errno == EWOULDBLOCK) throw Error(ErrorCode::Timeout, "timeout: recv timed out");
                    throw Error(ErrorCode::TransportError, "recv failed: " + std::string(std::strerror(errno)));
                }
                if (n == 0)
                {
                    check_aborted();
                    throw Error(ErrorCode::TransportError, "peer closed connection");
                }
                if (!first_bytes)
                {
                    first_bytes = true;
                    detail::lat_event(detail::kLatRecvFirstBytes, detail::tl_lat_current_id);
                }
                advance(iov, index, static_cast<std::size_t>(n));
                if (!validated && index >= 1)
                {
                    validated = true;
                    validate(header);
                    detail::lat_event(detail::kLatRecvHeaderOk, detail::tl_lat_current_id);
                }
            }
            detail::lat_event(detail::kLatPayloadDone, detail::tl_lat_current_id);
        }

        void set_io_timeout(std::chrono::milliseconds timeout) override
        {
            timeval tv{};
            tv.tv_sec = static_cast<decltype(tv.tv_sec)>(timeout.count() / 1000);
            tv.tv_usec = static_cast<decltype(tv.tv_usec)>((timeout.count() % 1000) * 1000);
            if (::setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) != 0 ||
                ::setsockopt(fd_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv)) != 0)
            {
                throw Error(ErrorCode::TransportError, "setsockopt(SO_RCVTIMEO/SO_SNDTIMEO) failed: " + std::string(std::strerror(errno)));
            }
        }

        void send(const void *data, std::size_t bytes) override
        {
            check_aborted();
            const auto *ptr = static_cast<const std::uint8_t *>(data);

            std::size_t sent = 0;

            while (sent < bytes)
            {
#ifdef MSG_NOSIGNAL
                const ssize_t n =
                    ::send(fd_, ptr + sent, bytes - sent, MSG_NOSIGNAL);
#else
                const ssize_t n =
                    ::send(fd_, ptr + sent, bytes - sent, 0);
#endif

                if (n < 0)
                {
                    if (errno == EINTR)
                    {
                        continue;
                    }
                    check_aborted();
                    if (errno == EAGAIN || errno == EWOULDBLOCK) throw Error(ErrorCode::Timeout, "timeout: send timed out");

                    throw Error(ErrorCode::TransportError, 
                        "send failed: " +
                        std::string(std::strerror(errno)));
                }

                if (n == 0)
                {
                    throw Error(ErrorCode::TransportError, "send returned 0");
                }

                sent += static_cast<std::size_t>(n);
            }
        }

        void recv(void *data, std::size_t bytes) override
        {
            check_aborted();
            auto *ptr = static_cast<std::uint8_t *>(data);

            std::size_t received = 0;

            while (received < bytes)
            {
                const ssize_t n =
                    ::recv(fd_, ptr + received, bytes - received, 0);

                if (n < 0)
                {
                    if (errno == EINTR)
                    {
                        continue;
                    }
                    check_aborted();
                    if (errno == EAGAIN || errno == EWOULDBLOCK) throw Error(ErrorCode::Timeout, "timeout: recv timed out");

                    throw Error(ErrorCode::TransportError, 
                        "recv failed: " +
                        std::string(std::strerror(errno)));
                }

                if (n == 0)
                {
                    check_aborted();
                    throw Error(ErrorCode::TransportError, "peer closed connection");
                }

                received += static_cast<std::size_t>(n);
            }
        }

        std::string peer_name() const override
        {
            return format_peer(fd_);
        }

    private:
        static void advance(iovec (&iov)[2], std::size_t &index, std::size_t n)
        {
            while (n > 0 && index < 2)
            {
                const std::size_t take = std::min(n, iov[index].iov_len);
                iov[index].iov_base = static_cast<char *>(iov[index].iov_base) + take;
                iov[index].iov_len -= take;
                n -= take;
                if (iov[index].iov_len == 0) ++index;
            }
        }

        // A local abort is the primary cause of any error that follows it; report it instead of the
        // platform-specific EOF/ECONNRESET/EPIPE/ENOTCONN the shutdown produced.
        void check_aborted()
        {
            if (!aborted_.load()) return;
            std::string reason;
            {
                std::lock_guard<std::mutex> lock(reason_mutex_);
                reason = reason_;
            }
            throw Error(ErrorCode::Aborted, "aborted: communicator aborted" + (reason.empty() ? "" : " (" + reason + ")"));
        }

        int fd_ = -1;
        std::atomic<bool> aborted_{false};
        std::mutex reason_mutex_;
        std::string reason_;
    };

    class TcpListener final : public Listener
    {
    public:
        TcpListener(int fd, TcpOptions options)
            : fd_(fd), options_(options) {}

        ~TcpListener() override
        {
            if (fd_ >= 0)
            {
                ::close(fd_);
            }
        }

        std::unique_ptr<Connection> accept() override
        {
            while (true)
            {
                const int client_fd = ::accept(fd_, nullptr, nullptr);

                if (client_fd < 0)
                {
                    if (errno == EINTR)
                    {
                        continue;
                    }

                    throw Error(ErrorCode::TransportError, 
                        "accept failed: " +
                        std::string(std::strerror(errno)));
                }

                return finish_accept(client_fd);
            }
        }

        std::uint16_t local_port() const override
        {
            sockaddr_in address{};
            socklen_t length = sizeof(address);
            if (::getsockname(fd_, reinterpret_cast<sockaddr *>(&address), &length) != 0) return 0;
            return ntohs(address.sin_port);
        }

        std::unique_ptr<Connection> accept_for(
            std::chrono::milliseconds timeout) override
        {
            const auto deadline =
                std::chrono::steady_clock::now() + timeout;

            while (true)
            {
                const auto remaining =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        deadline - std::chrono::steady_clock::now());

                if (remaining.count() <= 0)
                {
                    return nullptr;
                }

                pollfd poll_fd{};
                poll_fd.fd = fd_;
                poll_fd.events = POLLIN;

                const int poll_result =
                    ::poll(&poll_fd, 1, static_cast<int>(remaining.count()));

                if (poll_result < 0)
                {
                    if (errno == EINTR)
                    {
                        continue;
                    }

                    throw Error(ErrorCode::TransportError, 
                        "poll failed: " +
                        std::string(std::strerror(errno)));
                }

                if (poll_result == 0)
                {
                    // Timed out with no incoming connection.
                    return nullptr;
                }

                const int client_fd = ::accept(fd_, nullptr, nullptr);

                if (client_fd < 0)
                {
                    if (errno == EINTR ||
                        errno == EAGAIN ||
                        errno == EWOULDBLOCK ||
                        errno == ECONNABORTED)
                    {
                        // Spurious wakeup, or the peer dropped the
                        // connection between poll() and accept();
                        // retry within the remaining budget.
                        continue;
                    }

                    throw Error(ErrorCode::TransportError, 
                        "accept failed: " +
                        std::string(std::strerror(errno)));
                }

                return finish_accept(client_fd);
            }
        }

    private:
        std::unique_ptr<Connection> finish_accept(int client_fd)
        {
            try
            {
                apply_options(client_fd, options_);
            }
            catch (...)
            {
                ::close(client_fd);
                throw;
            }

            return std::make_unique<TcpConnection>(client_fd);
        }

        int fd_ = -1;
        TcpOptions options_;
    };

} // namespace

std::unique_ptr<Connection> tcp_connect(
    const std::string &host,
    std::uint16_t port,
    const TcpOptions &options)
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);

    if (fd < 0)
    {
        throw Error(ErrorCode::TransportError, 
            "socket failed: " + std::string(std::strerror(errno)));
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);

    if (::inet_pton(
            AF_INET,
            host.c_str(),
            &address.sin_addr) != 1)
    {

        ::close(fd);

        throw Error(ErrorCode::InvalidArgument, "invalid IPv4 address: " + host);
    }

    if (::connect(
            fd,
            reinterpret_cast<sockaddr *>(&address),
            sizeof(address)) != 0)
    {

        const std::string error = std::strerror(errno);

        ::close(fd);

        throw Error(ErrorCode::TransportError, 
            "connect failed to " + host + ":" +
            std::to_string(port) + ": " + error);
    }

    try
    {
        apply_options(fd, options);
    }
    catch (...)
    {
        ::close(fd);
        throw;
    }

    return std::make_unique<TcpConnection>(fd);
}

std::unique_ptr<Listener> tcp_listen(
    const std::string &bind_address,
    std::uint16_t port,
    const TcpOptions &options)
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);

    if (fd < 0)
    {
        throw Error(ErrorCode::TransportError, 
            "socket failed: " + std::string(std::strerror(errno)));
    }

    int reuse = 1;

    ::setsockopt(
        fd,
        SOL_SOCKET,
        SO_REUSEADDR,
        &reuse,
        sizeof(reuse));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);

    if (::inet_pton(
            AF_INET,
            bind_address.c_str(),
            &address.sin_addr) != 1)
    {

        ::close(fd);

        throw Error(ErrorCode::InvalidArgument, 
            "invalid IPv4 bind address: " + bind_address);
    }

    if (::bind(
            fd,
            reinterpret_cast<sockaddr *>(&address),
            sizeof(address)) != 0)
    {

        const std::string error = std::strerror(errno);

        ::close(fd);

        throw Error(ErrorCode::TransportError, 
            "bind failed on " + bind_address + ":" +
            std::to_string(port) + ": " + error);
    }

    if (::listen(fd, SOMAXCONN) != 0)
    {
        const std::string error = std::strerror(errno);

        ::close(fd);

        throw Error(ErrorCode::TransportError, "listen failed: " + error);
    }

    return std::make_unique<TcpListener>(fd, options);
}

bool busy_poll_supported()
{
#if defined(__linux__) && defined(SO_BUSY_POLL)
    return true;
#else
    return false;
#endif
}

// Phase 32 Part E: thin Transport adapter over an existing Connection.
// See transport.hpp for why this exists as a separate seam rather than
// treating Connection itself as "the" transport abstraction.
TcpTransport::TcpTransport(std::unique_ptr<Connection> connection)
    : connection_(std::move(connection))
{
    if (!connection_)
    {
        throw Error(ErrorCode::InvalidArgument, 
            "TcpTransport requires a non-null Connection");
    }
}

void TcpTransport::send(const void *data, std::size_t bytes)
{
    connection_->send(data, bytes);
}

void TcpTransport::recv(void *data, std::size_t bytes)
{
    connection_->recv(data, bytes);
}

void TcpTransport::send_framed(const void *header, std::size_t header_bytes, const void *data, std::size_t bytes)
{
    connection_->send_framed(header, header_bytes, data, bytes);
}

void TcpTransport::recv_framed(
    void *header, std::size_t header_bytes, void *data, std::size_t bytes, const std::function<void(const void *)> &validate)
{
    connection_->recv_framed(header, header_bytes, data, bytes, validate);
}

void TcpTransport::abort(const std::string &reason)
{
    connection_->abort(reason);
}

TransportCapabilities TcpTransport::capabilities() const noexcept
{
    TransportCapabilities caps;
    caps.reliable = true;
    caps.ordered = true;
    caps.supports_zero_copy = false;
    caps.supports_registered_memory = false;
    caps.supports_direct_device_memory = false;
    // TCP itself has no alignment requirement; 1 means "no preference",
    // not "must be byte-aligned only" -- higher layers are free to
    // choose any alignment they like when a transport reports 1.
    caps.preferred_chunk_alignment = 1;
    // One TCP stream has no notion of independently-outstanding
    // in-flight transfers at this layer -- send()/recv() calls on it
    // are serialized by the connection itself.
    caps.max_in_flight = 1;
    return caps;
}

std::string TcpTransport::peer_name() const
{
    return connection_->peer_name();
}

} // namespace tbccl
