// Manual multi-process broadcast smoke test: bootstraps a World from
// CLI args and runs one or more tbccl::broadcast() calls from a fixed
// root, validating every rank's buffer against the expected pattern
// each iteration. Not a benchmark; see benchmarks/tcp_pingpong.cpp for
// transport-level performance testing.

#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

    struct Options
    {
        std::size_t rank = 0;
        std::vector<tbccl::PeerEndpoint> peers;
        std::string bind_address;
        int busy_poll_us = 0;
        std::size_t root = 0;
        std::size_t bytes = 64;
        int iterations = 1;
    };

    void print_usage(const char *program)
    {
        std::cerr
            << "Usage:\n"
            << "  " << program
            << " --rank R --peers HOST:PORT,HOST:PORT,... "
            << "[--bind ADDRESS] [--busy-poll MICROSECONDS] "
            << "[--root R] [--bytes N] [--iterations N]\n\n"
            << "Example:\n"
            << "  " << program
            << " --rank 0 --peers "
            << "127.0.0.1:18515,127.0.0.1:18516,127.0.0.1:18517 "
            << "--root 1 --bytes 1048576 --iterations 10\n";
    }

    tbccl::PeerEndpoint parse_peer(const std::string &text)
    {
        const std::size_t colon = text.rfind(':');

        if (colon == std::string::npos || colon == 0 ||
            colon == text.size() - 1)
        {
            throw std::runtime_error("invalid peer endpoint: " + text);
        }

        tbccl::PeerEndpoint peer;
        peer.host = text.substr(0, colon);

        const std::string port_text = text.substr(colon + 1);

        unsigned long port_value = 0;

        try
        {
            std::size_t consumed = 0;
            port_value = std::stoul(port_text, &consumed);

            if (consumed != port_text.size())
            {
                throw std::runtime_error("trailing characters");
            }
        }
        catch (const std::exception &)
        {
            throw std::runtime_error("invalid port in peer endpoint: " + text);
        }

        if (port_value == 0 || port_value > 65535)
        {
            throw std::runtime_error("invalid port in peer endpoint: " + text);
        }

        peer.port = static_cast<std::uint16_t>(port_value);

        return peer;
    }

    std::vector<tbccl::PeerEndpoint> parse_peers(const std::string &input)
    {
        std::vector<tbccl::PeerEndpoint> peers;

        std::size_t start = 0;

        while (start < input.size())
        {
            const std::size_t comma = input.find(',', start);
            const std::size_t end =
                comma == std::string::npos ? input.size() : comma;

            peers.push_back(parse_peer(input.substr(start, end - start)));

            if (comma == std::string::npos)
            {
                break;
            }

            start = comma + 1;
        }

        return peers;
    }

    unsigned long parse_unsigned(
        const std::string &name,
        const std::string &value_text)
    {
        unsigned long value = 0;

        try
        {
            std::size_t consumed = 0;
            value = std::stoul(value_text, &consumed);

            if (consumed != value_text.size())
            {
                throw std::runtime_error("trailing characters");
            }
        }
        catch (const std::exception &)
        {
            throw std::runtime_error("invalid " + name + " value: " + value_text);
        }

        return value;
    }

    Options parse_options(int argc, char **argv)
    {
        Options options;

        bool have_rank = false;
        bool have_peers = false;

        for (int i = 1; i < argc; ++i)
        {
            const std::string argument = argv[i];

            auto require_value = [&](const std::string &name)
            {
                if (i + 1 >= argc)
                {
                    throw std::runtime_error("missing value for " + name);
                }

                return std::string(argv[++i]);
            };

            if (argument == "--rank")
            {
                options.rank =
                    static_cast<std::size_t>(
                        parse_unsigned(argument, require_value(argument)));
                have_rank = true;
            }
            else if (argument == "--peers")
            {
                options.peers = parse_peers(require_value(argument));
                have_peers = true;
            }
            else if (argument == "--bind")
            {
                options.bind_address = require_value(argument);
            }
            else if (argument == "--root")
            {
                options.root =
                    static_cast<std::size_t>(
                        parse_unsigned(argument, require_value(argument)));
            }
            else if (argument == "--bytes")
            {
                options.bytes =
                    static_cast<std::size_t>(
                        parse_unsigned(argument, require_value(argument)));
            }
            else if (argument == "--busy-poll")
            {
                const std::string value_text = require_value(argument);

                long value = 0;
                std::size_t consumed = 0;

                try
                {
                    value = std::stol(value_text, &consumed);
                }
                catch (const std::exception &)
                {
                    throw std::runtime_error(
                        "invalid --busy-poll value: " + value_text);
                }

                if (consumed != value_text.size() || value < 0 ||
                    value > 1000000)
                {
                    throw std::runtime_error(
                        "invalid --busy-poll value: " + value_text +
                        " (must be an integer between 0 and 1000000)");
                }

                options.busy_poll_us = static_cast<int>(value);
            }
            else if (argument == "--iterations")
            {
                const std::string value_text = require_value(argument);

                long value = 0;
                std::size_t consumed = 0;

                try
                {
                    value = std::stol(value_text, &consumed);
                }
                catch (const std::exception &)
                {
                    throw std::runtime_error(
                        "invalid --iterations value: " + value_text);
                }

                if (consumed != value_text.size() || value <= 0)
                {
                    throw std::runtime_error(
                        "invalid --iterations value: " + value_text +
                        " (must be a positive integer)");
                }

                options.iterations = static_cast<int>(value);
            }
            else if (argument == "--help" || argument == "-h")
            {
                print_usage(argv[0]);
                std::exit(0);
            }
            else
            {
                throw std::runtime_error("unknown argument: " + argument);
            }
        }

        if (!have_rank)
        {
            throw std::runtime_error("--rank is required");
        }

        if (!have_peers)
        {
            throw std::runtime_error("--peers is required");
        }

        return options;
    }

    std::vector<std::uint8_t> deterministic_buffer(
        std::size_t size,
        std::uint32_t seed)
    {
        std::vector<std::uint8_t> buffer(size);

        for (std::size_t i = 0; i < size; ++i)
        {
            buffer[i] = static_cast<std::uint8_t>(
                (static_cast<std::uint32_t>(i) * 2654435761u + seed) &
                0xFFu);
        }

        return buffer;
    }

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options options = parse_options(argc, argv);

        tbccl::TcpWorldOptions world_options;
        world_options.rank = options.rank;
        world_options.peers = options.peers;
        world_options.bind_address = options.bind_address;
        world_options.tcp.busy_poll_us = options.busy_poll_us;

        auto world = tbccl::create_tcp_world(world_options);

        std::cout
            << "rank=" << world->rank()
            << " world_size=" << world->size()
            << " root=" << options.root
            << " bytes=" << options.bytes << '\n';

        if (options.root >= world->size())
        {
            throw std::runtime_error(
                "--root " + std::to_string(options.root) +
                " is invalid for world size " +
                std::to_string(world->size()));
        }

        for (int iteration = 1; iteration <= options.iterations; ++iteration)
        {
            const auto pattern =
                deterministic_buffer(
                    options.bytes, static_cast<std::uint32_t>(iteration));

            std::vector<std::uint8_t> buffer =
                (world->rank() == options.root)
                    ? pattern
                    : deterministic_buffer(options.bytes, 0xBAADF00Du);

            tbccl::broadcast(*world, buffer.data(), buffer.size(), options.root);

            if (buffer != pattern)
            {
                throw std::runtime_error(
                    "iteration " + std::to_string(iteration) +
                    ": buffer does not match root's pattern");
            }

            std::cout
                << "rank " << world->rank() << " iteration " << iteration
                << "/" << options.iterations << ": OK\n";
        }

        std::cout << "rank " << world->rank() << ": PASS\n";

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
