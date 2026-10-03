// Manual multi-process reduce-scatter smoke test: bootstraps a World
// from CLI args and runs one or more tbccl::reduce_scatter() calls,
// with every rank computing its own expected segment independently
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
        std::size_t recv_count = 16;
        tbccl::DataType datatype = tbccl::DataType::Float32;
        tbccl::ReduceOp op = tbccl::ReduceOp::Sum;
        int iterations = 1;
    };

    void print_usage(const char *program)
    {
        std::cerr
            << "Usage:\n"
            << "  " << program
            << " --rank R --peers HOST:PORT,HOST:PORT,... "
            << "[--bind ADDRESS] [--busy-poll MICROSECONDS] "
            << "[--recv-count N] [--datatype int32|int64|float32|float64] "
            << "[--op sum|product|min|max] [--iterations N]\n\n"
            << "Example:\n"
            << "  " << program
            << " --rank 1 --peers "
            << "127.0.0.1:18515,127.0.0.1:18516,127.0.0.1:18517 "
            << "--recv-count 262144 --datatype float32 --op sum "
            << "--iterations 10\n";
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

    tbccl::DataType parse_datatype(const std::string &text)
    {
        if (text == "int32") return tbccl::DataType::Int32;
        if (text == "int64") return tbccl::DataType::Int64;
        if (text == "float32") return tbccl::DataType::Float32;
        if (text == "float64") return tbccl::DataType::Float64;

        throw std::runtime_error("invalid --datatype value: " + text);
    }

    tbccl::ReduceOp parse_op(const std::string &text)
    {
        if (text == "sum") return tbccl::ReduceOp::Sum;
        if (text == "product") return tbccl::ReduceOp::Product;
        if (text == "min") return tbccl::ReduceOp::Min;
        if (text == "max") return tbccl::ReduceOp::Max;

        throw std::runtime_error("invalid --op value: " + text);
    }

    const char *datatype_name(tbccl::DataType datatype)
    {
        switch (datatype)
        {
        case tbccl::DataType::Int32: return "int32";
        case tbccl::DataType::Int64: return "int64";
        case tbccl::DataType::Float32: return "float32";
        case tbccl::DataType::Float64: return "float64";
        default: break;
        }

        return "unknown";
    }

    const char *op_name(tbccl::ReduceOp op)
    {
        switch (op)
        {
        case tbccl::ReduceOp::Sum: return "sum";
        case tbccl::ReduceOp::Product: return "product";
        case tbccl::ReduceOp::Min: return "min";
        case tbccl::ReduceOp::Max: return "max";
        }

        return "unknown";
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
            else if (argument == "--recv-count")
            {
                options.recv_count =
                    static_cast<std::size_t>(
                        parse_unsigned(argument, require_value(argument)));
            }
            else if (argument == "--datatype")
            {
                options.datatype = parse_datatype(require_value(argument));
            }
            else if (argument == "--op")
            {
                options.op = parse_op(require_value(argument));
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

    // Deterministic values kept small ([1, 7]) so Product cannot
    // explode numerically, matching tests/reduce_scatter_test.cpp's
    // approach. `global_index` is this element's position in the full
    // world_size*recv_count reduced array (i.e. rank*recv_count + i).
    template <typename T>
    T generate_value(
        std::size_t rank,
        std::size_t global_index,
        int iteration)
    {
        const long value =
            (static_cast<long>(rank) + static_cast<long>(global_index) +
             iteration) %
                7 +
            1;

        return static_cast<T>(value);
    }

    template <typename T>
    void combine(T &acc, T value, tbccl::ReduceOp op)
    {
        switch (op)
        {
        case tbccl::ReduceOp::Sum:
            acc = acc + value;
            break;
        case tbccl::ReduceOp::Product:
            acc = acc * value;
            break;
        case tbccl::ReduceOp::Min:
            acc = acc < value ? acc : value;
            break;
        case tbccl::ReduceOp::Max:
            acc = acc > value ? acc : value;
            break;
        }
    }

    template <typename T>
    void run_iteration(
        tbccl::World &world,
        std::size_t world_size,
        const Options &options,
        int iteration)
    {
        const std::size_t rank = world.rank();
        const std::size_t total_count = world_size * options.recv_count;

        std::vector<T> send(total_count);

        for (std::size_t i = 0; i < total_count; ++i)
        {
            send[i] = generate_value<T>(rank, i, iteration);
        }

        std::vector<T> recv(options.recv_count);

        tbccl::reduce_scatter(
            world, send.data(), recv.data(), options.recv_count,
            options.datatype, options.op);

        for (std::size_t i = 0; i < options.recv_count; ++i)
        {
            const std::size_t global_index = rank * options.recv_count + i;

            T expected = generate_value<T>(0, global_index, iteration);

            for (std::size_t r = 1; r < world_size; ++r)
            {
                combine(
                    expected, generate_value<T>(r, global_index, iteration),
                    options.op);
            }

            if (recv[i] != expected)
            {
                throw std::runtime_error(
                    "iteration " + std::to_string(iteration) + ": element " +
                    std::to_string(i) +
                    " does not match independently computed expected "
                    "result");
            }
        }
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
            << " recv_count=" << options.recv_count
            << " datatype=" << datatype_name(options.datatype)
            << " op=" << op_name(options.op) << '\n';

        for (int iteration = 1; iteration <= options.iterations; ++iteration)
        {
            switch (options.datatype)
            {
            case tbccl::DataType::Int32:
                run_iteration<std::int32_t>(
                    *world, world->size(), options, iteration);
                break;
            case tbccl::DataType::Int64:
                run_iteration<std::int64_t>(
                    *world, world->size(), options, iteration);
                break;
            case tbccl::DataType::Float32:
                run_iteration<float>(*world, world->size(), options, iteration);
                break;
            case tbccl::DataType::Float64:
                run_iteration<double>(
                    *world, world->size(), options, iteration);
                break;
            default: throw std::runtime_error("this tool does not support that datatype");
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
