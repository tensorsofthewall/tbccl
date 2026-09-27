// Manual multi-process barrier smoke test: bootstraps a World from CLI
// args and executes one or more tbccl::barrier() calls, printing
// entry/exit around each so a human (or scripts/run_local_world.py-style
// launcher) can confirm every rank actually reached and left the
// barrier. Not a benchmark; see benchmarks/tcp_pingpong.cpp for
// transport-level performance testing, and §28 of the barrier plan for
// why this tool deliberately doesn't try to measure barrier latency.

#include <tbccl/collectives.hpp>
#include <tbccl/tcp_world.hpp>

#include <chrono>
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
        int iterations = 1;
    };

    void print_usage(const char *program)
    {
        std::cerr
            << "Usage:\n"
            << "  " << program
            << " --rank R --peers HOST:PORT,HOST:PORT,... "
            << "[--bind ADDRESS] [--busy-poll MICROSECONDS] "
            << "[--iterations N]\n\n"
            << "Example:\n"
            << "  " << program
            << " --rank 1 --peers "
            << "127.0.0.1:18515,127.0.0.1:18516,127.0.0.1:18517 "
            << "--iterations 100\n";
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
                const std::string value_text = require_value(argument);

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
                    throw std::runtime_error("invalid --rank value: " + value_text);
                }

                options.rank = static_cast<std::size_t>(value);
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
            << " world_size=" << world->size() << '\n';

        const auto start = std::chrono::steady_clock::now();

        for (int iteration = 1; iteration <= options.iterations; ++iteration)
        {
            std::cout
                << "rank " << world->rank() << " entered barrier ("
                << iteration << "/" << options.iterations << ")\n";

            tbccl::barrier(*world);

            std::cout
                << "rank " << world->rank() << " exited barrier ("
                << iteration << "/" << options.iterations << ")\n";
        }

        const auto elapsed =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start)
                .count();

        // Debug-only timing, not a formal benchmark: this includes
        // scheduling/thread-wakeup noise and the coordinator's O(N)
        // gather/release, not just network time.
        std::cout
            << "rank " << world->rank() << ": " << options.iterations
            << " barrier(s) in " << elapsed << " s\n";

        std::cout << "rank " << world->rank() << ": PASS\n";

        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
