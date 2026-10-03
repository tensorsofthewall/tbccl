// multi-peer, full-duplex asynchronous point-to-point over the N-rank Communicator (world_size 2, 3 and 4, loopback, dynamic ports).
//
// Large payloads (several MiB, far beyond the socket buffers) are the point: a send cannot complete until its receiver posts the matching
// recv, so a ring only finishes if every rank can send to one peer and receive from another at the same time, and an exchange between
// two ranks only finishes if send and recv on ONE peer pair are independent lanes. A watchdog turns any hang into a failure.

#include "mesh_test_support.hpp"

#include <tbccl/communicator.hpp>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;

namespace
{

    BufferView view(std::vector<std::uint8_t> &v) { return BufferView{MemoryKind::Host, v.data(), v.size(), 0}; }

    std::vector<std::uint8_t> pattern(std::size_t from, std::size_t tag, std::size_t n)
    {
        std::vector<std::uint8_t> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>(from * 131 + tag * 17 + i * 7 + (i >> 8));
        return v;
    }

    void post_and_wait(std::vector<tbccl::Work> &works)
    {
        for (auto &w : works)
        {
            w.wait();
            expect(!w.has_error(), "work failed: " + w.error());
        }
    }

    tbccl::Work send(tbccl::Communicator &c, std::vector<std::uint8_t> &v, std::size_t peer) { return c.send(view(v), v.size(), DataType::UInt8, peer); }
    tbccl::Work recv(tbccl::Communicator &c, std::vector<std::uint8_t> &v, std::size_t peer) { return c.recv(view(v), v.size(), DataType::UInt8, peer); }

    // rank r sends to r+1 and receives from r-1 at the same time.
    void test_ring(std::size_t world, std::size_t bytes)
    {
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            const std::size_t next = (rank + 1) % world, prev = (rank + world - 1) % world;
            auto out = pattern(rank, 1, bytes);
            std::vector<std::uint8_t> in(bytes, 0xEE);
            std::vector<tbccl::Work> works;
            // Alternate the posting order so both "send first" and "recv first" are covered.
            if (rank % 2 == 0)
            {
                works.push_back(send(comm, out, next));
                works.push_back(recv(comm, in, prev));
            }
            else
            {
                works.push_back(recv(comm, in, prev));
                works.push_back(send(comm, out, next));
            }
            post_and_wait(works);
            expect(in == pattern(prev, 1, bytes), "ring payload from rank " + std::to_string(prev));
        });
        std::cout << "[PASS] ring world_size=" << world << " bytes=" << bytes << "\n";
    }

    // Two ranks send to each other simultaneously: only works if send and recv on one peer are independent.
    void test_pair_exchange(std::size_t world, std::size_t bytes)
    {
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            if (rank >= world - world % 2) return; // an odd rank out sits idle
            const std::size_t peer = rank ^ 1;
            auto out = pattern(rank, 2, bytes);
            std::vector<std::uint8_t> in(bytes, 0xEE);
            std::vector<tbccl::Work> works;
            works.push_back(send(comm, out, peer)); // both sides send FIRST
            works.push_back(recv(comm, in, peer));
            post_and_wait(works);
            expect(in == pattern(peer, 2, bytes), "pair exchange payload");
        });
        std::cout << "[PASS] bidirectional pairs world_size=" << world << " bytes=" << bytes << "\n";
    }

    // Every rank sends a distinct payload to every other rank and receives from every other rank: 2(N-1) operations outstanding at once.
    void test_all_to_all(std::size_t world, std::size_t bytes)
    {
        run_world(world, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::vector<std::uint8_t>> out(world), in(world);
            std::vector<tbccl::Work> works;
            for (std::size_t peer = 0; peer < world; ++peer)
            {
                if (peer == rank) continue;
                out[peer] = pattern(rank, 100 + peer, bytes + peer); // sizes differ per pair
                in[peer].assign(bytes + rank, 0xEE);
            }
            for (std::size_t peer = 0; peer < world; ++peer)
            {
                if (peer == rank) continue;
                works.push_back(send(comm, out[peer], peer));
                works.push_back(recv(comm, in[peer], peer));
            }
            post_and_wait(works);
            for (std::size_t peer = 0; peer < world; ++peer)
            {
                if (peer == rank) continue;
                expect(in[peer] == pattern(peer, 100 + rank, bytes + rank), "all-to-all payload from rank " + std::to_string(peer));
            }
        });
        std::cout << "[PASS] all-to-all P2P world_size=" << world << " bytes=" << bytes << "\n";
    }

    // Rank 0 sends to three peers at once; rank 3 receives from three peers, posting the receives in the REVERSE of the order the data
    // is sent in. Peers 1 and 2 each receive from 0 and forward a different payload to 3.
    void test_one_to_two_and_two_to_one()
    {
        const std::size_t bytes = 2u << 20;
        run_world(4, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<tbccl::Work> works;
            if (rank == 0)
            {
                std::vector<std::vector<std::uint8_t>> out;
                for (std::size_t peer = 1; peer < 4; ++peer) out.push_back(pattern(0, peer, bytes));
                for (std::size_t peer = 1; peer < 4; ++peer) works.push_back(send(comm, out[peer - 1], peer));
                post_and_wait(works);
            }
            else if (rank == 3)
            {
                std::vector<std::vector<std::uint8_t>> in(3, std::vector<std::uint8_t>(bytes, 0xEE));
                for (int peer = 2; peer >= 0; --peer) works.push_back(recv(comm, in[peer], peer)); // reverse order
                post_and_wait(works);
                expect(in[0] == pattern(0, 3, bytes), "payload from rank 0");
                expect(in[1] == pattern(1, 7, bytes), "payload from rank 1");
                expect(in[2] == pattern(2, 8, bytes), "payload from rank 2");
            }
            else
            {
                std::vector<std::uint8_t> in(bytes, 0xEE);
                auto forward = pattern(rank, rank == 1 ? 7 : 8, bytes);
                works.push_back(recv(comm, in, 0));
                works.push_back(send(comm, forward, 3));
                post_and_wait(works);
                expect(in == pattern(0, rank, bytes), "payload from rank 0");
            }
        });
        std::cout << "[PASS] one rank sends to three peers, one rank receives from three peers in reverse order\n";
    }

    // Messages to one peer in one direction stay in posting order, whatever their sizes.
    void test_fifo_per_peer()
    {
        const std::size_t sizes[] = {1, 100000, 0, 17, 3u << 20, 4096, 5};
        run_world(3, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<tbccl::Work> works;
            std::vector<std::vector<std::uint8_t>> bufs;
            if (rank == 0)
            {
                for (std::size_t i = 0; i < 7; ++i) bufs.push_back(pattern(0, 50 + i, sizes[i]));
                for (std::size_t i = 0; i < 7; ++i) works.push_back(send(comm, bufs[i], 2));
                post_and_wait(works);
            }
            else if (rank == 2)
            {
                for (std::size_t i = 0; i < 7; ++i) bufs.emplace_back(sizes[i], 0xEE);
                for (std::size_t i = 0; i < 7; ++i) works.push_back(recv(comm, bufs[i], 0));
                post_and_wait(works);
                for (std::size_t i = 0; i < 7; ++i) expect(bufs[i] == pattern(0, 50 + i, sizes[i]), "FIFO message " + std::to_string(i));
            }
        });
        std::cout << "[PASS] per-peer FIFO ordering with mixed sizes, including zero bytes\n";
    }

    // A receive that cannot complete yet must not delay traffic with another peer: rank 0 waits on rank 1 (which sends late) and, meanwhile, exchanges with rank 2.
    void test_no_head_of_line_blocking_across_peers()
    {
        static std::atomic<bool> rank2_done{false};
        rank2_done = false;
        run_world(3, [&](std::size_t rank, tbccl::Communicator &comm) {
            const std::size_t bytes = 1u << 20;
            if (rank == 0)
            {
                std::vector<std::uint8_t> late(bytes, 0xEE), to2 = pattern(0, 9, bytes), from2(bytes, 0xEE);
                auto pending = recv(comm, late, 1); // posted first, completes last
                std::vector<tbccl::Work> works;
                works.push_back(send(comm, to2, 2));
                works.push_back(recv(comm, from2, 2));
                post_and_wait(works);
                expect(!pending.is_completed(), "the receive from rank 1 is still pending while the exchange with rank 2 finished");
                rank2_done = true;
                pending.wait();
                expect(!pending.has_error() && late == pattern(1, 9, bytes), "the late payload from rank 1 arrived intact");
            }
            else if (rank == 1)
            {
                while (!rank2_done) std::this_thread::sleep_for(std::chrono::milliseconds(5));
                auto out = pattern(1, 9, bytes);
                send(comm, out, 0).wait();
            }
            else
            {
                std::vector<std::uint8_t> in(bytes, 0xEE), out = pattern(2, 9, bytes);
                std::vector<tbccl::Work> works;
                works.push_back(recv(comm, in, 0));
                works.push_back(send(comm, out, 0));
                post_and_wait(works);
                expect(in == pattern(0, 9, bytes), "payload from rank 0");
            }
        });
        std::cout << "[PASS] an unfinished receive from one peer does not block another peer\n";
    }

    void test_invalid_peers()
    {
        run_world(3, [&](std::size_t rank, tbccl::Communicator &comm) {
            std::vector<std::uint8_t> b(16, 0);
            auto throws = [&](std::size_t peer, bool use_send, const std::string &needle) {
                try
                {
                    if (use_send) send(comm, b, peer);
                    else recv(comm, b, peer);
                }
                catch (const std::exception &e)
                {
                    return std::string(e.what()).find(needle) != std::string::npos;
                }
                return false;
            };
            expect(throws(rank, true, "invalid_argument"), "send to self is rejected");
            expect(throws(rank, false, "invalid_argument"), "recv from self is rejected");
            expect(throws(3, true, "invalid_argument"), "send to a rank outside the world is rejected");
            expect(throws(99, false, "invalid_argument"), "recv from a rank outside the world is rejected");
            expect(!comm.failed(), "argument errors do not poison the communicator");
        });
        std::cout << "[PASS] self and out-of-range peers are rejected without poisoning\n";
    }

    // The receiver posts a different size than the sender sent. It must fail clearly, never overrun its buffer, and not hang.
    void test_size_mismatch(bool receiver_larger)
    {
        const std::size_t sent = receiver_larger ? 1000 : 3000, posted = receiver_larger ? 3000 : 1000;
        run_world(2, [&](std::size_t rank, tbccl::Communicator &comm) {
            if (rank == 0)
            {
                auto out = pattern(0, 4, sent);
                auto w = send(comm, out, 1);
                w.wait(); // the sender cannot know; it either completed or the communicator was aborted around it
            }
            else
            {
                std::vector<std::uint8_t> in(posted + 64, 0xEE); // 64 guard bytes behind the posted size
                BufferView v{MemoryKind::Host, in.data(), posted, 0};
                auto w = comm.recv(v, posted, DataType::UInt8, 0);
                w.wait();
                expect(w.has_error() && w.error().find("protocol_mismatch") != std::string::npos && w.error().find("size mismatch") != std::string::npos,
                       "the receiver reports the size mismatch: " + w.error());
                for (std::size_t i = posted; i < in.size(); ++i) expect(in[i] == 0xEE, "no write beyond the posted receive size");
                expect(comm.failed(), "a framing mismatch poisons the communicator (the stream position is unknown)");
            }
        });
        std::cout << "[PASS] send/recv size mismatch (receiver " << (receiver_larger ? "larger" : "smaller") << ") fails clearly without overrun\n";
    }

} // namespace

int main()
{
    std::thread([] {
        std::this_thread::sleep_for(std::chrono::seconds(180));
        std::cerr << "[FAIL] watchdog: a multi-peer test hung\n";
        std::_Exit(2);
    }).detach();
    try
    {
        for (std::size_t world : {std::size_t{2}, std::size_t{3}, std::size_t{4}})
        {
            test_ring(world, 17);
            test_ring(world, 96u << 20);
            test_pair_exchange(world, 96u << 20);
            test_all_to_all(world, 1u << 20);
        }
        test_one_to_two_and_two_to_one();
        test_fifo_per_peer();
        test_no_head_of_line_blocking_across_peers();
        test_invalid_peers();
        test_size_mismatch(true);
        test_size_mismatch(false);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All N-rank P2P tests passed.\n";
    return 0;
}
