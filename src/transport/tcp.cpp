#include <tbccl/tcp.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <stdexcept>

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

            throw std::runtime_error(
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
            throw std::runtime_error(
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

        void send(const void *data, std::size_t bytes) override
        {
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

                    throw std::runtime_error(
                        "send failed: " +
                        std::string(std::strerror(errno)));
                }

                if (n == 0)
                {
                    throw std::runtime_error("send returned 0");
                }

                sent += static_cast<std::size_t>(n);
            }
        }

        void recv(void *data, std::size_t bytes) override
        {
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

                    throw std::runtime_error(
                        "recv failed: " +
                        std::string(std::strerror(errno)));
                }

                if (n == 0)
                {
                    throw std::runtime_error("peer closed connection");
                }

                received += static_cast<std::size_t>(n);
            }
        }

        std::string peer_name() const override
        {
            return format_peer(fd_);
        }

    private:
        int fd_ = -1;
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

                    throw std::runtime_error(
                        "accept failed: " +
                        std::string(std::strerror(errno)));
                }

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
        }

    private:
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
        throw std::runtime_error(
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

        throw std::runtime_error("invalid IPv4 address: " + host);
    }

    if (::connect(
            fd,
            reinterpret_cast<sockaddr *>(&address),
            sizeof(address)) != 0)
    {

        const std::string error = std::strerror(errno);

        ::close(fd);

        throw std::runtime_error(
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
        throw std::runtime_error(
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

        throw std::runtime_error(
            "invalid IPv4 bind address: " + bind_address);
    }

    if (::bind(
            fd,
            reinterpret_cast<sockaddr *>(&address),
            sizeof(address)) != 0)
    {

        const std::string error = std::strerror(errno);

        ::close(fd);

        throw std::runtime_error(
            "bind failed on " + bind_address + ":" +
            std::to_string(port) + ": " + error);
    }

    if (::listen(fd, 1) != 0)
    {
        const std::string error = std::strerror(errno);

        ::close(fd);

        throw std::runtime_error("listen failed: " + error);
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

} // namespace tbccl
