// Phase 43: Host-memory correctness tests for the byte-generic public Communicator::broadcast() and
// Communicator::all_gather() (roots 0 and 1, odd sizes, repeated rounds, FIFO ordering against
// all_reduce, capability queries, argument errors). CUDA coverage is in
// communicator_collectives_cuda_test.cpp.

#include <tbccl/communicator.hpp>

#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace
{

void expect(bool condition, const std::string &message)
{
    if (!condition) throw std::runtime_error("assertion failed: " + message);
}

void run_pair(
    std::uint16_t port,
    const std::function<void(tbccl::Communicator &)> &rank0_fn,
    const std::function<void(tbccl::Communicator &)> &rank1_fn)
{
    tbccl::CommunicatorOptions opts0;
    opts0.rank = 0;
    opts0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
    tbccl::CommunicatorOptions opts1 = opts0;
    opts1.rank = 1;

    std::exception_ptr err0, err1;
    std::thread t1([&] {
        try
        {
            auto comm1 = tbccl::Communicator::create(opts1);
            rank1_fn(*comm1);
        }
        catch (...)
        {
            err1 = std::current_exception();
        }
    });
    try
    {
        auto comm0 = tbccl::Communicator::create(opts0);
        rank0_fn(*comm0);
    }
    catch (...)
    {
        err0 = std::current_exception();
    }
    t1.join();
    if (err0) std::rethrow_exception(err0);
    if (err1) std::rethrow_exception(err1);
}

std::vector<std::uint8_t> pattern(std::size_t bytes, unsigned seed)
{
    std::vector<std::uint8_t> v(bytes);
    for (std::size_t i = 0; i < bytes; ++i) v[i] = static_cast<std::uint8_t>((i * 131u + seed * 17u + 3u) & 0xffu);
    return v;
}

void wait_ok(tbccl::Work work, const std::string &what)
{
    work.wait();
    expect(!work.has_error(), what + ": " + work.error());
}

constexpr std::uint16_t kBasePort = 29550;
std::uint16_t next_port()
{
    static std::uint16_t port = kBasePort;
    port = static_cast<std::uint16_t>(port + 4);
    return port;
}

void test_broadcast_roots_and_sizes()
{
    for (std::size_t root : {std::size_t{0}, std::size_t{1}})
    {
        for (std::size_t bytes : {std::size_t{1}, std::size_t{7}, std::size_t{4096}, std::size_t{1} << 20})
        {
            std::vector<std::vector<std::uint8_t>> got(2);
            run_pair(
                next_port(),
                [&](tbccl::Communicator &c) {
                    for (unsigned round = 0; round < 3; ++round)
                    {
                        auto buf = root == 0 ? pattern(bytes, round + 1) : std::vector<std::uint8_t>(bytes, 0xEE);
                        tbccl::BufferView v{tbccl::MemoryKind::Host, buf.data(), bytes, 0};
                        wait_ok(c.broadcast(v, root), "rank0 broadcast");
                        expect(buf == pattern(bytes, round + 1), "rank0 content, root=" + std::to_string(root));
                        got[0] = buf;
                    }
                },
                [&](tbccl::Communicator &c) {
                    for (unsigned round = 0; round < 3; ++round)
                    {
                        auto buf = root == 1 ? pattern(bytes, round + 1) : std::vector<std::uint8_t>(bytes, 0xEE);
                        tbccl::BufferView v{tbccl::MemoryKind::Host, buf.data(), bytes, 0};
                        wait_ok(c.broadcast(v, root), "rank1 broadcast");
                        expect(buf == pattern(bytes, round + 1), "rank1 content, root=" + std::to_string(root));
                        got[1] = buf;
                    }
                });
        }
    }
    std::cout << "[PASS] test_broadcast_roots_and_sizes\n";
}

void test_all_gather_sizes()
{
    for (std::size_t bytes : {std::size_t{8}, std::size_t{7}, std::size_t{4096}, std::size_t{1} << 20})
    {
        auto in0 = pattern(bytes, 11), in1 = pattern(bytes, 29);
        auto check = [&](std::vector<std::uint8_t> &o0, std::vector<std::uint8_t> &o1, const char *who) {
            expect(o0 == in0, std::string(who) + " outputs[0] must be rank0 input");
            expect(o1 == in1, std::string(who) + " outputs[1] must be rank1 input");
        };
        run_pair(
            next_port(),
            [&](tbccl::Communicator &c) {
                std::vector<std::uint8_t> o0(bytes, 0xEE), o1(bytes, 0xEE);
                tbccl::BufferView in{tbccl::MemoryKind::Host, in0.data(), bytes, 0};
                std::vector<tbccl::BufferView> outs = {
                    {tbccl::MemoryKind::Host, o0.data(), bytes, 0}, {tbccl::MemoryKind::Host, o1.data(), bytes, 0}};
                wait_ok(c.all_gather(in, outs), "rank0 all_gather");
                check(o0, o1, "rank0");
            },
            [&](tbccl::Communicator &c) {
                std::vector<std::uint8_t> o0(bytes, 0xEE), o1(bytes, 0xEE);
                tbccl::BufferView in{tbccl::MemoryKind::Host, in1.data(), bytes, 0};
                std::vector<tbccl::BufferView> outs = {
                    {tbccl::MemoryKind::Host, o0.data(), bytes, 0}, {tbccl::MemoryKind::Host, o1.data(), bytes, 0}};
                wait_ok(c.all_gather(in, outs), "rank1 all_gather");
                check(o0, o1, "rank1");
            });
    }
    std::cout << "[PASS] test_all_gather_sizes\n";
}

// DDP's parameter-count verification: one int64 per rank.
void test_all_gather_int64_metadata_and_aliasing()
{
    run_pair(
        next_port(),
        [&](tbccl::Communicator &c) {
            std::int64_t mine = 11, out[2] = {-1, -1};
            std::vector<tbccl::BufferView> outs = {
                {tbccl::MemoryKind::Host, &out[0], 8, 0}, {tbccl::MemoryKind::Host, &out[1], 8, 0}};
            // input aliases outputs[rank]: the local copy is skipped.
            out[0] = mine;
            wait_ok(c.all_gather({tbccl::MemoryKind::Host, &out[0], 8, 0}, outs), "rank0");
            expect(out[0] == 11 && out[1] == 22, "rank0 saw [11,22]");
        },
        [&](tbccl::Communicator &c) {
            std::int64_t mine = 22, out[2] = {-1, -1};
            std::vector<tbccl::BufferView> outs = {
                {tbccl::MemoryKind::Host, &out[0], 8, 0}, {tbccl::MemoryKind::Host, &out[1], 8, 0}};
            wait_ok(c.all_gather({tbccl::MemoryKind::Host, &mine, 8, 0}, outs), "rank1");
            expect(out[0] == 11 && out[1] == 22, "rank1 saw [11,22]");
        });
    std::cout << "[PASS] test_all_gather_int64_metadata_and_aliasing\n";
}

// One FIFO ordering domain: queue three different collectives back to back, wait afterwards.
void test_mixed_collectives_fifo_order()
{
    const std::size_t n = 4096;
    run_pair(
        next_port(),
        [&](tbccl::Communicator &c) {
            std::vector<float> a(n, 1.0f), r0(n), r1(n);
            std::vector<std::uint8_t> bc = pattern(333, 5), g_in = pattern(64, 1), g0(64), g1(64);
            std::vector<tbccl::BufferView> outs = {
                {tbccl::MemoryKind::Host, g0.data(), 64, 0}, {tbccl::MemoryKind::Host, g1.data(), 64, 0}};
            tbccl::BufferView av{tbccl::MemoryKind::Host, a.data(), n * 4, 0};
            auto w1 = c.all_reduce(av, av, n, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
            auto w2 = c.broadcast({tbccl::MemoryKind::Host, bc.data(), bc.size(), 0}, 0);
            auto w3 = c.all_gather({tbccl::MemoryKind::Host, g_in.data(), 64, 0}, outs);
            wait_ok(w3, "w3");
            wait_ok(w2, "w2");
            wait_ok(w1, "w1");
            expect(a[0] == 3.0f && a[n - 1] == 3.0f, "all_reduce result");
            expect(g0 == pattern(64, 1) && g1 == pattern(64, 2), "all_gather result");
        },
        [&](tbccl::Communicator &c) {
            std::vector<float> a(n, 2.0f);
            std::vector<std::uint8_t> bc(333, 0), g_in = pattern(64, 2), g0(64), g1(64);
            std::vector<tbccl::BufferView> outs = {
                {tbccl::MemoryKind::Host, g0.data(), 64, 0}, {tbccl::MemoryKind::Host, g1.data(), 64, 0}};
            tbccl::BufferView av{tbccl::MemoryKind::Host, a.data(), n * 4, 0};
            auto w1 = c.all_reduce(av, av, n, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
            auto w2 = c.broadcast({tbccl::MemoryKind::Host, bc.data(), bc.size(), 0}, 0);
            auto w3 = c.all_gather({tbccl::MemoryKind::Host, g_in.data(), 64, 0}, outs);
            wait_ok(w1, "w1");
            wait_ok(w2, "w2");
            wait_ok(w3, "w3");
            expect(a[0] == 3.0f, "all_reduce result");
            expect(bc == pattern(333, 5), "broadcast result");
            expect(g0 == pattern(64, 1) && g1 == pattern(64, 2), "all_gather result");
        });
    std::cout << "[PASS] test_mixed_collectives_fifo_order\n";
}

void test_capabilities_and_errors()
{
    run_pair(
        next_port(),
        [&](tbccl::Communicator &c) {
            expect(c.capabilities().supports_collective_broadcast(tbccl::MemoryKind::Host), "broadcast cap");
            expect(c.capabilities().supports_collective_all_gather(tbccl::MemoryKind::Host), "all_gather cap");
            std::uint8_t b[4] = {0};
            auto throws = [](const std::function<void()> &f, const std::string &needle) {
                try { f(); } catch (const std::runtime_error &e) { return std::string(e.what()).find(needle) != std::string::npos; }
                return false;
            };
            expect(throws([&] { c.broadcast({tbccl::MemoryKind::Host, b, 4, 0}, 2); }, "invalid_argument"), "root range");
            expect(throws([&] { c.broadcast({tbccl::MemoryKind::Host, nullptr, 4, 0}, 0); }, "invalid_argument"), "null buffer");
            expect(throws([&] { c.all_gather({tbccl::MemoryKind::Host, b, 4, 0}, {{tbccl::MemoryKind::Host, b, 4, 0}}); }, "invalid_argument"), "outputs count");
            expect(throws([&] { c.all_gather({tbccl::MemoryKind::Host, b, 4, 0}, {{tbccl::MemoryKind::Host, b, 4, 0}, {tbccl::MemoryKind::Host, b, 2, 0}}); }, "invalid_argument"), "output size");
            wait_ok(c.broadcast({tbccl::MemoryKind::Host, nullptr, 0, 0}, 0), "zero-byte broadcast");
        },
        [&](tbccl::Communicator &c) { wait_ok(c.broadcast({tbccl::MemoryKind::Host, nullptr, 0, 0}, 0), "zero-byte broadcast"); });
    std::cout << "[PASS] test_capabilities_and_errors\n";
}

} // namespace

int main()
{
    try
    {
        test_broadcast_roots_and_sizes();
        test_all_gather_sizes();
        test_all_gather_int64_metadata_and_aliasing();
        test_mixed_collectives_fifo_order();
        test_capabilities_and_errors();
    }
    catch (const std::exception &e)
    {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
    std::cout << "All communicator collective tests passed.\n";
    return 0;
}
