#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    constexpr uint16_t kDefaultPort = 18515;

    constexpr uint32_t kModePingPong = 1;
    constexpr uint32_t kModeStream = 2;
    constexpr uint32_t kModeQuit = 3;

    struct WireCommand
    {
        uint32_t mode;
        uint32_t reserved;
        uint64_t size;
        uint64_t iterations;
    };

    struct Result
    {
        std::string mode;
        uint64_t size = 0;
        uint64_t iterations = 0;
        double seconds = 0.0;
        double rtt_us = 0.0;
        double gbps = 0.0;
        double messages_per_second = 0.0;
    };

    struct Options
    {
        std::string role;
        std::string bind_address = "0.0.0.0";
        std::string host;
        std::string mode = "all";
        std::string output;

        uint16_t port = kDefaultPort;
        int busy_poll_us = 0;

        std::vector<uint64_t> sizes = {
            64,
            256,
            1024,
            4ULL * 1024,
            16ULL * 1024,
            64ULL * 1024,
            256ULL * 1024,
            1ULL * 1024 * 1024,
            4ULL * 1024 * 1024,
            16ULL * 1024 * 1024,
            64ULL * 1024 * 1024,
        };
    };

    // -----------------------------------------------------------------------------
    // Byte ordering
    // -----------------------------------------------------------------------------

    uint64_t hton64(uint64_t value)
    {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return (static_cast<uint64_t>(
                    htonl(static_cast<uint32_t>(value & 0xffffffffULL)))
                << 32) |
               htonl(static_cast<uint32_t>(value >> 32));
#else
        return value;
#endif
    }

    uint64_t ntoh64(uint64_t value)
    {
        return hton64(value);
    }

    // -----------------------------------------------------------------------------
    // Socket helpers
    // -----------------------------------------------------------------------------

    void send_all(int fd, const void *data, size_t length)
    {
        const auto *ptr = static_cast<const uint8_t *>(data);

        size_t sent = 0;

        while (sent < length)
        {
#ifdef MSG_NOSIGNAL
            const ssize_t n =
                ::send(fd, ptr + sent, length - sent, MSG_NOSIGNAL);
#else
            const ssize_t n =
                ::send(fd, ptr + sent, length - sent, 0);
#endif

            if (n < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }

                throw std::runtime_error(
                    "send failed: " + std::string(std::strerror(errno)));
            }

            if (n == 0)
            {
                throw std::runtime_error("send returned 0");
            }

            sent += static_cast<size_t>(n);
        }
    }

    void recv_all(int fd, void *data, size_t length)
    {
        auto *ptr = static_cast<uint8_t *>(data);

        size_t received = 0;

        while (received < length)
        {
            const ssize_t n =
                ::recv(fd, ptr + received, length - received, 0);

            if (n < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }

                throw std::runtime_error(
                    "recv failed: " + std::string(std::strerror(errno)));
            }

            if (n == 0)
            {
                throw std::runtime_error("peer closed connection");
            }

            received += static_cast<size_t>(n);
        }
    }

    void set_tcp_nodelay(int fd)
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

    int create_server_socket(
        const std::string &bind_address,
        uint16_t port)
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

            throw std::runtime_error("bind failed: " + error);
        }

        if (::listen(fd, 1) != 0)
        {
            const std::string error = std::strerror(errno);

            ::close(fd);

            throw std::runtime_error("listen failed: " + error);
        }

        return fd;
    }

    int connect_socket(
        const std::string &host,
        uint16_t port)
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

            throw std::runtime_error(
                "invalid IPv4 address: " + host);
        }

        if (::connect(
                fd,
                reinterpret_cast<sockaddr *>(&address),
                sizeof(address)) != 0)
        {

            const std::string error = std::strerror(errno);

            ::close(fd);

            throw std::runtime_error("connect failed: " + error);
        }

        set_tcp_nodelay(fd);

        return fd;
    }

    // Returns true if SO_BUSY_POLL was actually applied to the socket.
    bool set_busy_poll(int fd, int microseconds)
    {
    #if defined(__linux__) && defined(SO_BUSY_POLL)
        if (microseconds <= 0)
        {
            return false;
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

        return true;
    #else
        (void)fd;
        (void)microseconds;
        return false;
    #endif
    }

    void print_busy_poll_status(int microseconds, bool applied)
    {
        if (applied)
        {
            std::cout
                << "SO_BUSY_POLL="
                << microseconds
                << " us\n";
        }
        else if (microseconds == 0)
        {
            std::cout << "SO_BUSY_POLL=off\n";
        }
        else
        {
            std::cout << "SO_BUSY_POLL=unsupported\n";
        }
    }

    // -----------------------------------------------------------------------------
    // Protocol
    // -----------------------------------------------------------------------------

    void send_command(
        int fd,
        uint32_t mode,
        uint64_t size,
        uint64_t iterations)
    {

        WireCommand command{};

        command.mode = htonl(mode);
        command.reserved = 0;
        command.size = hton64(size);
        command.iterations = hton64(iterations);

        send_all(fd, &command, sizeof(command));
    }

    WireCommand recv_command(int fd)
    {
        WireCommand wire{};

        recv_all(fd, &wire, sizeof(wire));

        WireCommand host{};

        host.mode = ntohl(wire.mode);
        host.size = ntoh64(wire.size);
        host.iterations = ntoh64(wire.iterations);

        return host;
    }

    // -----------------------------------------------------------------------------
    // Parsing
    // -----------------------------------------------------------------------------

    uint64_t parse_size(std::string text)
    {
        if (text.empty())
        {
            throw std::runtime_error("empty message size");
        }

        uint64_t multiplier = 1;

        char suffix =
            static_cast<char>(
                std::toupper(
                    static_cast<unsigned char>(text.back())));

        if (suffix == 'B')
        {
            text.pop_back();
        }
        else if (suffix == 'K')
        {
            text.pop_back();
            multiplier = 1024ULL;
        }
        else if (suffix == 'M')
        {
            text.pop_back();
            multiplier = 1024ULL * 1024ULL;
        }
        else if (suffix == 'G')
        {
            text.pop_back();
            multiplier =
                1024ULL * 1024ULL * 1024ULL;
        }

        if (text.empty())
        {
            throw std::runtime_error("invalid message size");
        }

        size_t consumed = 0;

        const uint64_t value =
            std::stoull(text, &consumed);

        if (consumed != text.size() || value == 0)
        {
            throw std::runtime_error(
                "invalid message size: " + text);
        }

        if (value >
            std::numeric_limits<uint64_t>::max() /
                multiplier)
        {

            throw std::runtime_error("message size overflow");
        }

        return value * multiplier;
    }

    std::vector<uint64_t> parse_sizes(
        const std::string &input)
    {

        std::vector<uint64_t> result;

        size_t start = 0;

        while (start < input.size())
        {
            const size_t comma =
                input.find(',', start);

            const size_t end =
                comma == std::string::npos
                    ? input.size()
                    : comma;

            result.push_back(
                parse_size(
                    input.substr(start, end - start)));

            if (comma == std::string::npos)
            {
                break;
            }

            start = comma + 1;
        }

        return result;
    }

    // -----------------------------------------------------------------------------
    // Benchmark sizing
    // -----------------------------------------------------------------------------

    uint64_t pingpong_iterations(uint64_t size)
    {
        // Roughly 64 MiB of payload per direction,
        // with reasonable limits for tiny/huge messages.
        constexpr uint64_t target =
            64ULL * 1024ULL * 1024ULL;

        return std::clamp<uint64_t>(
            target / size,
            10,
            10000);
    }

    uint64_t stream_iterations(uint64_t size)
    {
        // Aim for roughly 512 MiB per streaming test.
        constexpr uint64_t target =
            4ULL* 1024ULL * 1024ULL * 1024ULL;

        return std::clamp<uint64_t>(
            target / size,
            16,
            100000);
    }

    // -----------------------------------------------------------------------------
    // Server
    // -----------------------------------------------------------------------------

    void server_loop(int fd)
    {
        std::vector<uint8_t> buffer;

        const uint8_t ready = 1;

        while (true)
        {
            const WireCommand command =
                recv_command(fd);

            if (command.mode == kModeQuit)
            {
                return;
            }

            if (command.size == 0 ||
                command.iterations == 0 ||
                command.size >
                    static_cast<uint64_t>(
                        std::numeric_limits<size_t>::max()))
            {

                throw std::runtime_error(
                    "invalid command received");
            }

            buffer.resize(
                static_cast<size_t>(command.size));

            // Signal that allocation/setup is done.
            send_all(fd, &ready, sizeof(ready));

            if (command.mode == kModePingPong)
            {
                for (uint64_t i = 0;
                     i < command.iterations;
                     ++i)
                {

                    recv_all(
                        fd,
                        buffer.data(),
                        buffer.size());

                    send_all(
                        fd,
                        buffer.data(),
                        buffer.size());
                }
            }
            else if (command.mode == kModeStream)
            {

                for (uint64_t i = 0;
                     i < command.iterations;
                     ++i)
                {

                    recv_all(
                        fd,
                        buffer.data(),
                        buffer.size());
                }

                // Completion acknowledgement.
                send_all(fd, &ready, sizeof(ready));
            }
            else
            {
                throw std::runtime_error(
                    "unknown benchmark mode");
            }
        }
    }

    // -----------------------------------------------------------------------------
    // Client benchmarks
    // -----------------------------------------------------------------------------

    Result run_pingpong(
        int fd,
        uint64_t size)
    {

        const uint64_t iterations =
            pingpong_iterations(size);

        std::vector<uint8_t> buffer(
            static_cast<size_t>(size),
            0xA5);

        send_command(
            fd,
            kModePingPong,
            size,
            iterations);

        uint8_t ready = 0;

        recv_all(fd, &ready, sizeof(ready));

        const auto start =
            std::chrono::steady_clock::now();

        for (uint64_t i = 0;
             i < iterations;
             ++i)
        {

            send_all(
                fd,
                buffer.data(),
                buffer.size());

            recv_all(
                fd,
                buffer.data(),
                buffer.size());
        }

        const auto end =
            std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration<double>(
                end - start)
                .count();

        // Ping-pong moves size bytes in each direction.
        const double total_bytes =
            static_cast<double>(size) *
            static_cast<double>(iterations) *
            2.0;

        Result result;

        result.mode = "pingpong";
        result.size = size;
        result.iterations = iterations;
        result.seconds = seconds;

        result.rtt_us =
            seconds * 1e6 /
            static_cast<double>(iterations);

        result.gbps =
            total_bytes * 8.0 /
            seconds /
            1e9;

        result.messages_per_second =
            static_cast<double>(iterations) /
            seconds;

        return result;
    }

    Result run_stream(
        int fd,
        uint64_t size)
    {

        const uint64_t iterations =
            stream_iterations(size);

        std::vector<uint8_t> buffer(
            static_cast<size_t>(size),
            0x5A);

        send_command(
            fd,
            kModeStream,
            size,
            iterations);

        uint8_t ready = 0;

        recv_all(fd, &ready, sizeof(ready));

        const auto start =
            std::chrono::steady_clock::now();

        for (uint64_t i = 0;
             i < iterations;
             ++i)
        {

            send_all(
                fd,
                buffer.data(),
                buffer.size());
        }

        // Wait until the receiver has consumed
        // every byte before stopping the timer.
        recv_all(fd, &ready, sizeof(ready));

        const auto end =
            std::chrono::steady_clock::now();

        const double seconds =
            std::chrono::duration<double>(
                end - start)
                .count();

        const double total_bytes =
            static_cast<double>(size) *
            static_cast<double>(iterations);

        Result result;

        result.mode = "stream";
        result.size = size;
        result.iterations = iterations;
        result.seconds = seconds;

        result.gbps =
            total_bytes * 8.0 /
            seconds /
            1e9;

        result.messages_per_second =
            static_cast<double>(iterations) /
            seconds;

        return result;
    }

    // -----------------------------------------------------------------------------
    // Output
    // -----------------------------------------------------------------------------

    std::string human_size(uint64_t bytes)
    {
        const char *units[] = {
            "B",
            "KiB",
            "MiB",
            "GiB"};

        double value =
            static_cast<double>(bytes);

        size_t unit = 0;

        while (value >= 1024.0 &&
               unit < 3)
        {

            value /= 1024.0;
            ++unit;
        }

        std::ostringstream output;

        if (value >= 10.0 || unit == 0)
        {
            output
                << std::fixed
                << std::setprecision(0);
        }
        else
        {
            output
                << std::fixed
                << std::setprecision(1);
        }

        output
            << value
            << ' '
            << units[unit];

        return output.str();
    }

    void print_result(const Result &result)
    {
        std::cout
            << std::left
            << std::setw(10)
            << result.mode

            << std::right
            << std::setw(10)
            << human_size(result.size)

            << "  n="
            << std::setw(7)
            << result.iterations;

        if (result.mode == "pingpong")
        {
            std::cout
                << "  RTT="
                << std::fixed
                << std::setprecision(2)
                << std::setw(10)
                << result.rtt_us
                << " us";
        }
        else
        {
            std::cout
                << "  RTT="
                << std::setw(13)
                << "-";
        }

        std::cout
            << "  "
            << std::fixed
            << std::setprecision(3)
            << std::setw(8)
            << result.gbps
            << " Gbit/s"

            << "  "
            << std::setprecision(0)
            << std::setw(10)
            << result.messages_per_second
            << " msg/s"

            << '\n';
    }

    void write_csv(
        const std::string &path,
        const std::vector<Result> &results)
    {

        if (path.empty())
        {
            return;
        }

        const std::filesystem::path output_path(path);

        std::error_code ec;

        if (output_path.has_parent_path())
        {
            std::filesystem::create_directories(
                output_path.parent_path(), ec);

            if (ec)
            {
                throw std::runtime_error(
                    "failed to create output directory '" +
                    output_path.parent_path().string() +
                    "': " +
                    ec.message());
            }
        }

        std::ofstream file(path);

        if (!file)
        {
            throw std::runtime_error(
                "failed to open output file '" +
                path +
                "': " +
                std::string(std::strerror(errno)));
        }

        file
            << "mode,"
            << "size_bytes,"
            << "iterations,"
            << "seconds,"
            << "rtt_us,"
            << "gbps,"
            << "messages_per_second\n";

        file << std::setprecision(12);

        for (const auto &result : results)
        {
            file
                << result.mode << ','
                << result.size << ','
                << result.iterations << ','
                << result.seconds << ','
                << result.rtt_us << ','
                << result.gbps << ','
                << result.messages_per_second
                << '\n';
        }
    }

    // -----------------------------------------------------------------------------
    // CLI
    // -----------------------------------------------------------------------------

    void print_usage(const char *program)
    {
        std::cerr
            << "Usage:\n\n"

            << "Server:\n"
            << "  " << program
            << " server "
            << "[--bind IPv4] "
            << "[--port PORT] "
            << "[--busy-poll MICROSECONDS]\n\n"

            << "Client:\n"
            << "  " << program
            << " client "
            << "--host IPv4 "
            << "[--port PORT] "
            << "[--mode all|pingpong|stream] "
            << "[--sizes LIST] "
            << "[--output FILE] "
            << "[--busy-poll MICROSECONDS]\n\n"

            << "Options:\n"
            << "  --busy-poll MICROSECONDS\n"
            << "      Linux SO_BUSY_POLL duration. "
            << "0 disables it. Default: 0.\n\n"

            << "Examples:\n"

            << "  "
            << program
            << " server --bind 192.168.3.2\n"

            << "  "
            << program
            << " server --bind 192.168.3.2 "
            << "--busy-poll 100\n"

            << "  "
            << program
            << " client --host 192.168.3.2\n"

            << "  "
            << program
            << " client --host 192.168.3.2 "
            << "--busy-poll 100\n"

            << "  "
            << program
            << " client --host 192.168.3.2 "
            << "--sizes 64,4K,1M,16M\n"

            << "  "
            << program
            << " client --host 192.168.3.2 "
            << "--output results/run.csv\n";
    }

    Options parse_options(
        int argc,
        char **argv)
    {

        if (argc < 2)
        {
            print_usage(argv[0]);
            throw std::runtime_error(
                "missing server/client role");
        }

        Options options;

        options.role = argv[1];

        for (int i = 2; i < argc; ++i)
        {
            const std::string argument = argv[i];

            auto require_value =
                [&](const std::string &option)
            {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error(
                        "missing value for " + option);
                }

                return std::string(argv[++i]);
            };

            if (argument == "--bind")
            {
                options.bind_address =
                    require_value(argument);
            }
            else if (argument == "--host")
            {
                options.host =
                    require_value(argument);
            }
            else if (argument == "--port")
            {
                const unsigned long value =
                    std::stoul(
                        require_value(argument));

                if (value == 0 ||
                    value > 65535)
                {

                    throw std::runtime_error(
                        "invalid port");
                }

                options.port =
                    static_cast<uint16_t>(value);
            }
            else if (argument == "--mode")
            {
                options.mode =
                    require_value(argument);

                if (options.mode != "all" &&
                    options.mode != "pingpong" &&
                    options.mode != "stream")
                {

                    throw std::runtime_error(
                        "mode must be "
                        "all, pingpong, or stream");
                }
            }
            else if (argument == "--sizes")
            {
                options.sizes =
                    parse_sizes(
                        require_value(argument));
            }
            else if (argument == "--output")
            {
                options.output =
                    require_value(argument);
            }
            else if (argument == "--busy-poll")
            {
                const std::string value_text =
                    require_value(argument);

                long value = 0;
                size_t consumed = 0;

                try
                {
                    value = std::stol(value_text, &consumed);
                }
                catch (const std::exception &)
                {
                    throw std::runtime_error(
                        "invalid --busy-poll value: " +
                        value_text);
                }

                if (consumed != value_text.size() ||
                    value < 0 ||
                    value > 1000000)
                {

                    throw std::runtime_error(
                        "invalid --busy-poll value: " +
                        value_text +
                        " (must be an integer between "
                        "0 and 1000000)");
                }

                options.busy_poll_us =
                    static_cast<int>(value);
            }
            else if (argument == "--help" ||
                     argument == "-h")
            {

                print_usage(argv[0]);
                std::exit(0);
            }
            else
            {
                throw std::runtime_error(
                    "unknown argument: " + argument);
            }
        }

        if (options.role != "server" &&
            options.role != "client")
        {

            throw std::runtime_error(
                "role must be server or client");
        }

        if (options.role == "client" &&
            options.host.empty())
        {

            throw std::runtime_error(
                "--host is required in client mode");
        }

        return options;
    }

    // -----------------------------------------------------------------------------
    // Entry points
    // -----------------------------------------------------------------------------

    int run_server(const Options &options)
    {
        const int listener =
            create_server_socket(
                options.bind_address,
                options.port);

        std::cout
            << "Listening on "
            << options.bind_address
            << ':'
            << options.port
            << '\n';

        while (true)
        {
            sockaddr_in peer{};
            socklen_t peer_length = sizeof(peer);

            const int client =
                ::accept(
                    listener,
                    reinterpret_cast<sockaddr *>(&peer),
                    &peer_length);

            if (client < 0)
            {
                if (errno == EINTR)
                {
                    continue;
                }

                ::close(listener);

                throw std::runtime_error(
                    "accept failed: " +
                    std::string(std::strerror(errno)));
            }

            char peer_address[INET_ADDRSTRLEN]{};

            ::inet_ntop(
                AF_INET,
                &peer.sin_addr,
                peer_address,
                sizeof(peer_address));

            std::cout
                << "Connected: "
                << peer_address
                << ':'
                << ntohs(peer.sin_port)
                << '\n';

            try
            {
                set_tcp_nodelay(client);

                const bool busy_poll_applied =
                    set_busy_poll(client, options.busy_poll_us);

                std::cout << "TCP_NODELAY=on\n";
                print_busy_poll_status(
                    options.busy_poll_us,
                    busy_poll_applied);

                server_loop(client);
            }
            catch (const std::exception &error)
            {
                std::cerr
                    << "Client disconnected/error: "
                    << error.what()
                    << '\n';
            }

            ::close(client);

            std::cout
                << "Client disconnected. "
                << "Waiting for next connection...\n";
        }
    }

    int run_client(const Options &options)
    {
        const int fd =
            connect_socket(
                options.host,
                options.port);

        std::cout
            << "Connected to "
            << options.host
            << ':'
            << options.port
            << '\n';

        const bool busy_poll_applied =
            set_busy_poll(fd, options.busy_poll_us);

        std::cout << "TCP_NODELAY=on\n";
        print_busy_poll_status(
            options.busy_poll_us,
            busy_poll_applied);
        std::cout << '\n';

        std::vector<Result> results;

        for (const uint64_t size : options.sizes)
        {
            if (size >
                static_cast<uint64_t>(
                    std::numeric_limits<size_t>::max()))
            {

                throw std::runtime_error(
                    "message size too large");
            }

            if (options.mode == "all" ||
                options.mode == "pingpong")
            {

                const Result result =
                    run_pingpong(fd, size);

                print_result(result);

                results.push_back(result);
            }

            if (options.mode == "all" ||
                options.mode == "stream")
            {

                const Result result =
                    run_stream(fd, size);

                print_result(result);

                results.push_back(result);
            }
        }

        send_command(
            fd,
            kModeQuit,
            0,
            0);

        ::close(fd);

        write_csv(
            options.output,
            results);

        if (!options.output.empty())
        {
            std::cout
                << "\nWrote "
                << options.output
                << '\n';
        }

        return 0;
    }

} // namespace

int main(
    int argc,
    char **argv)
{

    try
    {
        const Options options =
            parse_options(argc, argv);

        if (options.role == "server")
        {
            return run_server(options);
        }

        return run_client(options);
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "error: "
            << error.what()
            << '\n';

        return 1;
    }
}