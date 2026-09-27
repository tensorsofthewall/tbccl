#pragma once

// Shared multi-rank test scaffolding, factored out of the collective
// test files (barrier_test.cpp, broadcast_test.cpp, all_gather_test.cpp,
// reduce_test.cpp all carry near-identical copies of this). New test
// files may include this instead of duplicating it further; existing
// ones are left as-is rather than churned for the sake of it.

#include <tbccl/tcp_world.hpp>
#include <tbccl/world.hpp>

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace tbccl_test
{

    inline void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    inline std::vector<tbccl::PeerEndpoint> make_local_peers(
        std::uint16_t base_port,
        std::size_t count)
    {
        std::vector<tbccl::PeerEndpoint> peers;

        for (std::size_t i = 0; i < count; ++i)
        {
            peers.push_back(
                {"127.0.0.1",
                 static_cast<std::uint16_t>(base_port + i)});
        }

        return peers;
    }

    inline tbccl::TcpWorldOptions make_options(
        std::size_t rank,
        const std::vector<tbccl::PeerEndpoint> &peers,
        int timeout_ms = 5000)
    {
        tbccl::TcpWorldOptions options;

        options.rank = rank;
        options.peers = peers;
        options.bootstrap_timeout = std::chrono::milliseconds(timeout_ms);

        return options;
    }

    inline void run_rank(
        const tbccl::TcpWorldOptions &options,
        const std::function<void(tbccl::World &)> &body,
        std::exception_ptr &out_exception)
    {
        try
        {
            auto world = tbccl::create_tcp_world(options);
            body(*world);
        }
        catch (...)
        {
            out_exception = std::current_exception();
        }
    }

    inline std::string what_or_empty(const std::exception_ptr &ptr)
    {
        if (!ptr)
        {
            return "";
        }

        try
        {
            std::rethrow_exception(ptr);
        }
        catch (const std::exception &error)
        {
            return error.what();
        }
        catch (...)
        {
            return "non-std::exception";
        }
    }

    inline void join_and_check(
        std::vector<std::thread> &threads,
        const std::vector<std::exception_ptr> &errors)
    {
        for (auto &thread : threads)
        {
            thread.join();
        }

        for (std::size_t rank = 0; rank < errors.size(); ++rank)
        {
            if (errors[rank])
            {
                throw std::runtime_error(
                    "rank " + std::to_string(rank) +
                    " failed: " + what_or_empty(errors[rank]));
            }
        }
    }

    inline std::vector<std::uint8_t> deterministic_buffer(
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

} // namespace tbccl_test
