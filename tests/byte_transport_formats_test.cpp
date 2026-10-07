// Libtbccl moves OPAQUE bytes. Quantized and low-precision payloads that the runtime cannot (and must not) reduce -- FP8 E4M3/E5M2 words, 16-bit
// float and int8 buffers, a packed 4-bit weight bundle with its scales and zero-points, and arbitrary random bytes that belong to no dtype at all --
// must cross send/recv, broadcast and all_gather bit-exactly, in odd sizes, between Host and (when built) CUDA memory.
//
// TBCCL sees buffers only: nothing here names GPTQ/AWQ/NF4/MXFP4, group sizes, scales or zero points as TBCCL concepts. The packers below are test code (the
// owning framework would do this); the three bundle members are transported as three independent buffers. P2P describes every payload as UInt8 elements, one
// per byte (the datatype and count of send/recv are only a size check; the transfer moves BufferView::bytes).

#include <tbccl/communicator.hpp>

#ifdef BYTE_TEST_WITH_CUDA
#include <tbccl/cuda_support.hpp>

#include <cuda_runtime.h>
#endif

#include <atomic>
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

    // expect() is called from both rank threads, so the counters are atomic.
    std::atomic<std::uint64_t> g_checks{0};        // number of byte-exactness assertions evaluated
    std::atomic<std::uint64_t> g_bytes_checked{0}; // total payload bytes those assertions covered

    void expect(bool condition, const std::string &message)
    {
        ++g_checks;
        if (!condition) throw std::runtime_error("assertion failed: " + message);
    }

    // Control ports 29254-29265 (pairs of two), data plane = control + 1000 = 30254-30265: both below the kernel's ephemeral range (32768+). Cases run one after
    // another and the listener sets SO_REUSEADDR, so the six slots are reused in turn.
    std::uint16_t g_next_port = 29254;
    std::uint16_t take_port()
    {
        const std::uint16_t p = g_next_port;
        g_next_port = static_cast<std::uint16_t>(g_next_port + 2);
        if (g_next_port >= 29266) g_next_port = 29254;
        return p;
    }

    void run_pair(const std::function<void(tbccl::Communicator &)> &rank0_fn, const std::function<void(tbccl::Communicator &)> &rank1_fn)
    {
        const std::uint16_t port = take_port();
        tbccl::CommunicatorOptions o0;
        o0.rank = 0;
        o0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
        tbccl::CommunicatorOptions o1 = o0;
        o1.rank = 1;
        std::exception_ptr e0, e1;
        std::thread t([&] {
            try { auto c = tbccl::Communicator::create(o1); rank1_fn(*c); } catch (...) { e1 = std::current_exception(); }
        });
        try { auto c = tbccl::Communicator::create(o0); rank0_fn(*c); } catch (...) { e0 = std::current_exception(); }
        t.join();
        if (e0) std::rethrow_exception(e0);
        if (e1) std::rethrow_exception(e1);
    }

    // ----- payload builders (test code; TBCCL never sees these formats) ------------------------------------------------------------------------------------
    std::uint32_t next_random(std::uint32_t &state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    // Every byte value 0..255 appears (so every FP8 encoding -- NaNs, zeros, subnormals, extremes -- is exercised), then pseudo-random bytes.
    std::vector<std::uint8_t> raw_bytes(std::size_t n, std::uint32_t seed)
    {
        std::vector<std::uint8_t> v(n);
        std::uint32_t state = seed | 1u;
        for (std::size_t i = 0; i < n; ++i) v[i] = i < 256 ? static_cast<std::uint8_t>(i) : static_cast<std::uint8_t>(next_random(state) >> 13);
        return v;
    }

    struct Payload
    {
        std::string name;
        std::vector<std::uint8_t> bytes;
    };

    std::vector<Payload> payloads()
    {
        std::vector<Payload> out;
        for (std::size_t n : {std::size_t{1}, std::size_t{3}, std::size_t{17}, std::size_t{1021}, std::size_t{4099}, (std::size_t{1} << 20) + 3})
        {
            out.push_back({"fp8-e4m3-words/" + std::to_string(n), raw_bytes(n, 11u + static_cast<std::uint32_t>(n))});
            out.push_back({"fp8-e5m2-words/" + std::to_string(n), raw_bytes(n, 23u + static_cast<std::uint32_t>(n))});
            out.push_back({"random-bytes/" + std::to_string(n), raw_bytes(n, 7u + static_cast<std::uint32_t>(n))});
        }
        // 16-bit float words (arbitrary patterns incl. NaN/inf) and int8: even byte counts, same buffers as any other bytes.
        out.push_back({"bf16-or-fp16-words/4096", raw_bytes(4096, 5)});
        out.push_back({"int8/65537", raw_bytes(65537, 9)});
        return out;
    }

    // Synthetic AWQ/GPTQ-style layer: packed 4-bit weights (two values per byte, low nibble first), 16-bit scales and signed zero points, three independent buffers.
    struct Bundle
    {
        std::vector<std::uint8_t> packed;  // [rows, cols / 2]
        std::vector<std::uint8_t> scales;  // [rows, cols / group] 16-bit words (raw)
        std::vector<std::uint8_t> zeros;   // [rows, cols / group] int8
        std::vector<std::uint8_t> values;  // the unpacked 4-bit values (one per byte), for reconstruction checks
    };

    Bundle make_bundle(std::size_t rows, std::size_t cols, std::size_t group, std::uint32_t seed)
    {
        Bundle b;
        std::uint32_t state = seed | 1u;
        b.values.resize(rows * cols);
        for (auto &v : b.values) v = static_cast<std::uint8_t>(next_random(state) & 0xF);
        b.packed.resize(rows * cols / 2);
        for (std::size_t i = 0; i < b.packed.size(); ++i) b.packed[i] = static_cast<std::uint8_t>(b.values[2 * i] | (b.values[2 * i + 1] << 4));
        const std::size_t groups = rows * cols / group;
        b.scales.resize(groups * 2);
        for (auto &v : b.scales) v = static_cast<std::uint8_t>(next_random(state) >> 11);
        b.zeros.resize(groups);
        for (auto &v : b.zeros) v = static_cast<std::uint8_t>(static_cast<std::int8_t>(static_cast<int>(next_random(state) % 16) - 8));
        return b;
    }

    std::vector<std::uint8_t> unpack(const std::vector<std::uint8_t> &packed)
    {
        std::vector<std::uint8_t> v(packed.size() * 2);
        for (std::size_t i = 0; i < packed.size(); ++i)
        {
            v[2 * i] = packed[i] & 0xF;
            v[2 * i + 1] = packed[i] >> 4;
        }
        return v;
    }

    // ----- memory kinds -----------------------------------------------------------------------------------------------------------------------------------------
    enum class Kind { Host, Cuda };

    // A buffer of `bytes` bytes in host or device memory with upload/download helpers.
    class Buf
    {
    public:
        Buf(Kind kind, std::size_t bytes) : kind_(kind), bytes_(bytes), host_(kind == Kind::Host ? bytes : 0)
        {
#ifdef BYTE_TEST_WITH_CUDA
            if (kind == Kind::Cuda && bytes > 0 && cudaMalloc(&dev_, bytes) != cudaSuccess) throw std::runtime_error("cudaMalloc failed");
#endif
        }
        ~Buf()
        {
#ifdef BYTE_TEST_WITH_CUDA
            if (dev_) cudaFree(dev_);
#endif
        }
        Buf(const Buf &) = delete;
        Buf &operator=(const Buf &) = delete;

        void set(const std::vector<std::uint8_t> &v)
        {
            expect(v.size() == bytes_, "Buf::set size");
            if (kind_ == Kind::Host) { host_ = v; return; }
#ifdef BYTE_TEST_WITH_CUDA
            if (bytes_ > 0 && cudaMemcpy(dev_, v.data(), bytes_, cudaMemcpyHostToDevice) != cudaSuccess) throw std::runtime_error("H2D failed");
#endif
        }
        std::vector<std::uint8_t> get() const
        {
            if (kind_ == Kind::Host) return host_;
            std::vector<std::uint8_t> v(bytes_);
#ifdef BYTE_TEST_WITH_CUDA
            if (bytes_ > 0 && cudaMemcpy(v.data(), dev_, bytes_, cudaMemcpyDeviceToHost) != cudaSuccess) throw std::runtime_error("D2H failed");
#endif
            return v;
        }
        tbccl::BufferView view()
        {
            if (kind_ == Kind::Host) return {tbccl::MemoryKind::Host, host_.data(), bytes_, -1};
            return {tbccl::MemoryKind::Cuda, dev_, bytes_, 0};
        }

    private:
        Kind kind_;
        std::size_t bytes_;
        std::vector<std::uint8_t> host_;
        void *dev_ = nullptr;
    };

    std::vector<std::uint8_t> other_bytes(std::size_t n)
    {
        std::vector<std::uint8_t> v(n, 0xA5);
        return v;
    }

    void wait_ok(tbccl::Work w, const std::string &what)
    {
        w.wait();
        expect(!w.has_error(), what + ": " + w.error());
    }

    // ----- operations ---------------------------------------------------------------------------------------------------------------------------------------------
    // Both ranks run the same sequence; `kinds[r]` is where rank r's buffers live.
    void test_send_recv(const Kind kinds[2], const std::vector<Payload> &set)
    {
        for (const Payload &p : set)
        {
            for (std::size_t sender : {std::size_t{0}, std::size_t{1}})
            {
                std::vector<std::uint8_t> got;
                run_pair(
                    [&](tbccl::Communicator &c) {
                        Buf b(kinds[0], p.bytes.size());
                        b.set(sender == 0 ? p.bytes : other_bytes(p.bytes.size()));
                        wait_ok(sender == 0 ? c.send(b.view(), p.bytes.size(), tbccl::DataType::UInt8, 1) : c.recv(b.view(), p.bytes.size(), tbccl::DataType::UInt8, 1), "rank0 p2p");
                        if (sender == 1) got = b.get();
                    },
                    [&](tbccl::Communicator &c) {
                        Buf b(kinds[1], p.bytes.size());
                        b.set(sender == 1 ? p.bytes : other_bytes(p.bytes.size()));
                        wait_ok(sender == 1 ? c.send(b.view(), p.bytes.size(), tbccl::DataType::UInt8, 0) : c.recv(b.view(), p.bytes.size(), tbccl::DataType::UInt8, 0), "rank1 p2p");
                        if (sender == 0) got = b.get();
                    });
                g_bytes_checked += p.bytes.size();
                expect(got == p.bytes, p.name + ": send/recv from rank " + std::to_string(sender) + " not byte exact");
            }
        }
    }

    void test_broadcast_all_gather(const Kind kinds[2], const std::vector<Payload> &set)
    {
        for (const Payload &p : set)
        {
            if (p.bytes.size() > (std::size_t{1} << 18)) continue; // broadcast/all_gather are tuned for small metadata; keep the large cases to send/recv
            for (std::size_t root : {std::size_t{0}, std::size_t{1}})
            {
                std::vector<std::uint8_t> out[2];
                auto body = [&](std::size_t rank, tbccl::Communicator &c) {
                    Buf b(kinds[rank], p.bytes.size());
                    b.set(rank == root ? p.bytes : other_bytes(p.bytes.size()));
                    wait_ok(c.broadcast(b.view(), root), "broadcast");
                    out[rank] = b.get();
                };
                run_pair([&](tbccl::Communicator &c) { body(0, c); }, [&](tbccl::Communicator &c) { body(1, c); });
                g_bytes_checked += 2 * p.bytes.size();
                expect(out[0] == p.bytes && out[1] == p.bytes, p.name + ": broadcast from root " + std::to_string(root) + " not byte exact");
            }
            std::vector<std::uint8_t> mine[2] = {p.bytes, raw_bytes(p.bytes.size(), 4242u + static_cast<std::uint32_t>(p.bytes.size()))};
            std::vector<std::uint8_t> slots[2][2];
            auto gather = [&](std::size_t rank, tbccl::Communicator &c) {
                Buf in(kinds[rank], p.bytes.size()), o0(kinds[rank], p.bytes.size()), o1(kinds[rank], p.bytes.size());
                in.set(mine[rank]);
                o0.set(other_bytes(p.bytes.size()));
                o1.set(other_bytes(p.bytes.size()));
                std::vector<tbccl::BufferView> outs = {o0.view(), o1.view()};
                wait_ok(c.all_gather(in.view(), outs), "all_gather");
                slots[rank][0] = o0.get();
                slots[rank][1] = o1.get();
            };
            run_pair([&](tbccl::Communicator &c) { gather(0, c); }, [&](tbccl::Communicator &c) { gather(1, c); });
            for (std::size_t r = 0; r < 2; ++r)
                g_bytes_checked += 2 * p.bytes.size(), expect(slots[r][0] == mine[0] && slots[r][1] == mine[1], p.name + ": all_gather result on rank " + std::to_string(r) + " not byte exact");
        }
    }

    void test_bundle(const Kind kinds[2])
    {
        constexpr std::size_t kRows = 64, kCols = 256, kGroup = 32;
        for (std::size_t sender : {std::size_t{0}, std::size_t{1}})
        {
            const Bundle src = make_bundle(kRows, kCols, kGroup, 100u + static_cast<std::uint32_t>(sender));
            Bundle got;
            const std::vector<std::uint8_t> *members[3] = {&src.packed, &src.scales, &src.zeros};
            std::vector<std::uint8_t> recvd[3];
            auto body = [&](std::size_t rank, tbccl::Communicator &c) {
                for (int m = 0; m < 3; ++m)
                {
                    const std::size_t n = members[m]->size();
                    Buf b(kinds[rank], n);
                    b.set(rank == sender ? *members[m] : other_bytes(n));
                    wait_ok(rank == sender ? c.send(b.view(), n, tbccl::DataType::UInt8, 1 - rank) : c.recv(b.view(), n, tbccl::DataType::UInt8, 1 - rank), "bundle member");
                    if (rank != sender) recvd[m] = b.get();
                }
            };
            run_pair([&](tbccl::Communicator &c) { body(0, c); }, [&](tbccl::Communicator &c) { body(1, c); });
            got.packed = recvd[0];
            got.scales = recvd[1];
            got.zeros = recvd[2];
            expect(got.packed == src.packed, "packed 4-bit weight bytes changed in transit");
            expect(got.scales == src.scales, "16-bit scale bytes changed in transit");
            expect(got.zeros == src.zeros, "zero-point bytes changed in transit");
            expect(unpack(got.packed) == src.values, "reconstructed 4-bit values differ from the source");
        }
    }

    void run_all(const char *label, const Kind kinds[2])
    {
        const auto set = payloads();
        test_send_recv(kinds, set);
        test_broadcast_all_gather(kinds, set);
        test_bundle(kinds);
        std::cout << "[PASS] " << label << "\n";
    }

} // namespace

int main()
{
#ifdef BYTE_TEST_WITH_CUDA
    tbccl::register_cuda_support();
#endif
    try
    {
        const Kind hh[2] = {Kind::Host, Kind::Host};
        run_all("host <-> host: FP8 words, 16-bit/int8 buffers, random bytes, packed-int4 bundle (send/recv, broadcast, all_gather)", hh);
#ifdef BYTE_TEST_WITH_CUDA
        const Kind ch[2] = {Kind::Cuda, Kind::Host};
        const Kind hc[2] = {Kind::Host, Kind::Cuda};
        const Kind cc[2] = {Kind::Cuda, Kind::Cuda};
        run_all("cuda <-> host", ch);
        run_all("host <-> cuda", hc);
        run_all("cuda <-> cuda", cc);
#endif
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
    std::cout << "All byte transport format tests passed (" << g_checks.load() << " assertions, " << g_bytes_checked.load() << " payload bytes compared).\n";
    return 0;
}
