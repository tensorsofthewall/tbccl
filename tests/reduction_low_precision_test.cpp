// Host correctness of the 16-bit floating-point reduction types (Float16, BFloat16) and, from the integer commit on, Int8/UInt8.
//
//   1. Conversions are proven by construction, with no oracle: every one of the 65536 bit patterns decodes to an independently
//      computed exact value; for every pair of adjacent positive patterns the exact midpoint rounds to the EVEN neighbor and the floats
//      just above/below it round to the upper/lower neighbor (ties-to-even, overflow-to-infinity and the subnormal/normal boundary all
//      fall out of that rule). Hand-selected vectors cover +/-0, subnormals, smallest normal, largest finite, overflow, inf and NaN.
//      Where the compiler provides _Float16 / __bf16 they are used as an extra cross-check (test code only; never in a public header).
//   2. SUM = widen to float32, one float32 add, round once (ties-to-even), element-wise: edge-pattern cross product, deterministic
//      random vectors of assorted lengths.
//   3. The same arithmetic through the real N=2 collectives on TCP loopback: World reference/ring/pipelined all_reduce and the
//      production Communicator::all_reduce on host buffers (in place and out of place).

#include <tbccl/collectives.hpp>
#include <tbccl/communicator.hpp>
#include <tbccl/reduction.hpp>

#include "all_reduce_internal.hpp"
#include "low_precision.hpp"
#include "reduction_internal.hpp"
#include "test_utils.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using tbccl_test::expect;
namespace lowp = tbccl::detail::lowp;

namespace
{

    // ----- format traits -------------------------------------------------------------------------------------------------------
    struct Fp16
    {
        static constexpr const char *name = "float16";
        static constexpr tbccl::DataType dt = tbccl::DataType::Float16;
        using Elem = lowp::Half;
        static constexpr std::uint16_t kInf = 0x7C00, kLastFinite = 0x7BFF;
        static float dec(std::uint16_t b) { return lowp::fp16_bits_to_float(b); }
        static std::uint16_t enc(float f) { return lowp::float_to_fp16_bits_rne(f); }
        static bool is_nan(std::uint16_t b) { return (b & 0x7C00) == 0x7C00 && (b & 0x03FF) != 0; }
        // Independent exact value of a non-NaN pattern.
        static double exact(std::uint16_t b)
        {
            const double sign = (b & 0x8000) ? -1.0 : 1.0;
            const int e = (b >> 10) & 0x1F, m = b & 0x3FF;
            if (e == 0x1F) return sign * std::numeric_limits<double>::infinity();
            return sign * (e == 0 ? std::ldexp(static_cast<double>(m), -24) : std::ldexp(static_cast<double>(1024 + m), e - 25));
        }
    };

    struct Bf16
    {
        static constexpr const char *name = "bfloat16";
        static constexpr tbccl::DataType dt = tbccl::DataType::BFloat16;
        using Elem = lowp::BFloat16;
        static constexpr std::uint16_t kInf = 0x7F80, kLastFinite = 0x7F7F;
        static float dec(std::uint16_t b) { return lowp::bf16_bits_to_float(b); }
        static std::uint16_t enc(float f) { return lowp::float_to_bf16_bits_rne(f); }
        static bool is_nan(std::uint16_t b) { return (b & 0x7F80) == 0x7F80 && (b & 0x007F) != 0; }
        static double exact(std::uint16_t b)
        {
            const double sign = (b & 0x8000) ? -1.0 : 1.0;
            const int e = (b >> 7) & 0xFF, m = b & 0x7F;
            if (e == 0xFF) return sign * std::numeric_limits<double>::infinity();
            return sign * (e == 0 ? std::ldexp(static_cast<double>(m), -133) : std::ldexp(static_cast<double>(128 + m), e - 134));
        }
    };

    std::uint32_t next_random(std::uint32_t &state)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return state;
    }

    // Equal bits, or both NaN (NaN payloads are not compared; see low_precision.hpp).
    template <class F>
    bool same(std::uint16_t got, std::uint16_t want)
    {
        if (F::is_nan(want)) return F::is_nan(got);
        return got == want;
    }

    template <class F>
    std::uint16_t expected_sum(std::uint16_t a, std::uint16_t b)
    {
        return F::enc(F::dec(a) + F::dec(b));
    }

    // ----- 1. conversions ------------------------------------------------------------------------------------------------------
    template <class F>
    void test_decode_exhaustive()
    {
        for (std::uint32_t b = 0; b <= 0xFFFFu; ++b)
        {
            const auto bits = static_cast<std::uint16_t>(b);
            const float got = F::dec(bits);
            if (F::is_nan(bits))
            {
                expect(std::isnan(got), std::string(F::name) + ": NaN pattern must decode to NaN");
                continue;
            }
            const double want = F::exact(bits);
            expect(static_cast<double>(got) == want && std::signbit(static_cast<double>(got)) == std::signbit(want),
                   std::string(F::name) + ": decode mismatch for pattern " + std::to_string(b));
            expect(F::enc(got) == bits, std::string(F::name) + ": decode->encode must round trip, pattern " + std::to_string(b));
        }
        std::cout << "[PASS] test_decode_exhaustive<" << F::name << ">\n";
    }

    template <class F>
    void test_encode_rounding_exhaustive()
    {
        for (std::uint16_t p = 0; p < F::kInf; ++p)
        {
            const std::uint16_t next = static_cast<std::uint16_t>(p + 1);
            const double lo = F::exact(p);
            const double hi = next == F::kInf ? lo + (lo - F::exact(static_cast<std::uint16_t>(p - 1))) : F::exact(next);
            const double mid_exact = lo + (hi - lo) / 2.0;
            const float mid = static_cast<float>(mid_exact);
            expect(static_cast<double>(mid) == mid_exact, std::string(F::name) + ": midpoint must be exactly representable in float32");
            const std::uint16_t even = (p & 1u) ? next : p;
            for (std::uint16_t sign : {std::uint16_t{0}, std::uint16_t{0x8000}})
            {
                const float s = sign ? -1.0f : 1.0f;
                expect(F::enc(s * mid) == (sign | even), std::string(F::name) + ": midpoint above pattern " + std::to_string(p) + " must round to even");
                expect(F::enc(s * std::nextafter(mid, std::numeric_limits<float>::infinity())) == (sign | next),
                       std::string(F::name) + ": just above midpoint must round up, pattern " + std::to_string(p));
                expect(F::enc(s * std::nextafter(mid, 0.0f)) == (sign | p),
                       std::string(F::name) + ": just below midpoint must round down, pattern " + std::to_string(p));
            }
        }
        std::cout << "[PASS] test_encode_rounding_exhaustive<" << F::name << ">\n";
    }

    struct EncodeCase
    {
        std::uint32_t float_bits;
        std::uint16_t want;
    };

    void expect_encode_vectors(const char *name, std::uint16_t (*enc)(float), bool (*is_nan)(std::uint16_t), const std::vector<EncodeCase> &cases)
    {
        for (const EncodeCase &c : cases)
        {
            float value;
            std::memcpy(&value, &c.float_bits, sizeof(value));
            const std::uint16_t got = enc(value);
            if (is_nan(c.want))
                expect(is_nan(got), std::string(name) + ": NaN input must encode to NaN, float bits " + std::to_string(c.float_bits));
            else
                expect(got == c.want, std::string(name) + ": encode vector mismatch, float bits 0x" + std::to_string(c.float_bits) + " got " + std::to_string(got) +
                                          " want " + std::to_string(c.want));
        }
    }

    void test_special_vectors()
    {
        // float32 bit pattern -> expected binary16 bits
        expect_encode_vectors("float16", Fp16::enc, Fp16::is_nan, {
            {0x00000000, 0x0000}, // +0
            {0x80000000, 0x8000}, // -0
            {0x3F800000, 0x3C00}, // 1.0
            {0xBF800000, 0xBC00}, // -1.0
            {0x477FE000, 0x7BFF}, // 65504, the largest finite half
            {0x477FEFFF, 0x7BFF}, // largest float below 65520 still rounds down
            {0x477FF000, 0x7C00}, // 65520: exact tie, rounds to even = infinity
            {0x47800000, 0x7C00}, // 65536 overflows
            {0xC7800000, 0xFC00}, // -65536 overflows
            {0x7F800000, 0x7C00}, // +inf
            {0xFF800000, 0xFC00}, // -inf
            {0x7FC00000, 0x7E00}, // quiet NaN
            {0xFFC00000, 0xFE00}, // -NaN keeps its sign
            {0x7F800001, 0x7E00}, // signalling-style NaN stays NaN
            {0x33800000, 0x0001}, // 2^-24: smallest subnormal
            {0x33000000, 0x0000}, // 2^-25: exact tie between 0 and 2^-24, even = 0
            {0x33000001, 0x0001}, // just above 2^-25 rounds up
            {0x33C00000, 0x0002}, // 1.5 * 2^-24: tie between 1 and 2 ulps, even = 2
            {0x387FFFFF, 0x0400}, // largest float below 2^-14 rounds up into the smallest normal
            {0x38800000, 0x0400}, // 2^-14: smallest normal
            {0x387FC000, 0x03FF}, // largest subnormal (1023 * 2^-24)
            {0x3F801000, 0x3C00}, // 1 + 2^-11: halfway between 0x3C00 and 0x3C01, even = 0x3C00
            {0x3F803000, 0x3C02}, // 1 + 3 * 2^-11: halfway between 0x3C01 and 0x3C02, even = 0x3C02
            {0x3F801001, 0x3C01}, // just above the first tie rounds up
            {0x00000001, 0x0000}, // tiny float32 subnormal flushes to 0
        });
        // float32 bit pattern -> expected bfloat16 bits
        expect_encode_vectors("bfloat16", Bf16::enc, Bf16::is_nan, {
            {0x00000000, 0x0000}, {0x80000000, 0x8000}, {0x3F800000, 0x3F80}, {0xBF800000, 0xBF80},
            {0x3F808000, 0x3F80}, // halfway between 0x3F80 and 0x3F81, even = 0x3F80
            {0x3F818000, 0x3F82}, // halfway between 0x3F81 and 0x3F82, even = 0x3F82 (truncation would give 0x3F81)
            {0x3F808001, 0x3F81}, // just above a tie rounds up
            {0x3F807FFF, 0x3F80}, // just below a tie rounds down
            {0x7F7FFFFF, 0x7F80}, // FLT_MAX rounds to +inf in bfloat16
            {0xFF7FFFFF, 0xFF80}, {0x7F7F8000, 0x7F80}, // tie at the top of the range: even = infinity
            {0x7F800000, 0x7F80}, {0xFF800000, 0xFF80},
            {0x7FC00000, 0x7FC0}, {0x7F800001, 0x7FC0}, {0xFFC00000, 0xFFC0}, // NaNs stay NaN, sign kept
            {0x00010000, 0x0001}, // smallest bfloat16 subnormal region value
            {0x00008000, 0x0000}, // exact tie between 0 and the smallest bfloat16 subnormal, even = 0
            {0x00018000, 0x0002}, // tie between 1 and 2, even = 2
        });
        // decode spot checks
        expect(Fp16::dec(0x3C00) == 1.0f && Fp16::dec(0xC000) == -2.0f && Fp16::dec(0x7BFF) == 65504.0f, "fp16 decode spot values");
        expect(Fp16::dec(0x0001) == std::ldexp(1.0f, -24) && Fp16::dec(0x0400) == std::ldexp(1.0f, -14), "fp16 subnormal/normal decode");
        expect(std::signbit(Fp16::dec(0x8000)) && Fp16::dec(0x8000) == 0.0f, "fp16 -0 decodes to -0");
        expect(std::isinf(Fp16::dec(0xFC00)) && Fp16::dec(0xFC00) < 0, "fp16 -inf");
        expect(Bf16::dec(0x3F80) == 1.0f && Bf16::dec(0xC000) == -2.0f, "bf16 decode spot values");
        expect(std::isnan(Bf16::dec(0x7FC0)) && std::isinf(Bf16::dec(0x7F80)), "bf16 NaN/inf decode");
        std::cout << "[PASS] test_special_vectors\n";
    }

#if defined(__FLT16_MAX__)
    std::uint16_t oracle_f2h(float f)
    {
        const _Float16 h = static_cast<_Float16>(f);
        std::uint16_t b;
        std::memcpy(&b, &h, sizeof(b));
        return b;
    }
    float oracle_h2f(std::uint16_t b)
    {
        _Float16 h;
        std::memcpy(&h, &b, sizeof(h));
        return static_cast<float>(h);
    }
#endif
#if defined(__BFLT16_MAX__)
    std::uint16_t oracle_f2b(float f)
    {
        const __bf16 h = static_cast<__bf16>(f);
        std::uint16_t b;
        std::memcpy(&b, &h, sizeof(b));
        return b;
    }
    float oracle_b2f(std::uint16_t b)
    {
        __bf16 h;
        std::memcpy(&h, &b, sizeof(h));
        return static_cast<float>(h);
    }
#endif

    // Extra cross-check against the compiler's own conversions, when it has the types (g++ and Apple clang have _Float16; g++ has __bf16).
    void test_compiler_oracle()
    {
        int checked = 0;
        auto sweep = [&](const char *name, auto enc, auto dec, auto oracle_enc, auto oracle_dec, auto is_nan) {
            for (std::uint32_t b = 0; b <= 0xFFFFu; ++b)
            {
                const auto bits = static_cast<std::uint16_t>(b);
                if (is_nan(bits)) continue;
                expect(dec(bits) == oracle_dec(bits), std::string(name) + ": decode differs from the compiler type, pattern " + std::to_string(b));
            }
            for (std::uint64_t x = 0; x <= 0xFFFFFFFFull; x += 4093)
            {
                const auto fb = static_cast<std::uint32_t>(x);
                float f;
                std::memcpy(&f, &fb, sizeof(f));
                const std::uint16_t got = enc(f), want = oracle_enc(f);
                if (std::isnan(f)) expect(is_nan(got) && is_nan(want), std::string(name) + ": NaN handling differs");
                else expect(got == want, std::string(name) + ": encode differs from the compiler type for float bits " + std::to_string(fb));
            }
            ++checked;
        };
#if defined(__FLT16_MAX__)
        sweep("float16", Fp16::enc, Fp16::dec, oracle_f2h, oracle_h2f, Fp16::is_nan);
#endif
#if defined(__BFLT16_MAX__)
        sweep("bfloat16", Bf16::enc, Bf16::dec, oracle_f2b, oracle_b2f, Bf16::is_nan);
#endif
        std::cout << "[PASS] test_compiler_oracle (" << checked << " format(s) cross-checked against compiler types)\n";
    }

    // ----- 2. element-wise SUM ----------------------------------------------------------------------------------------------------
    template <class F>
    std::vector<std::uint16_t> edge_patterns();

    template <>
    std::vector<std::uint16_t> edge_patterns<Fp16>()
    {
        return {0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x0401, 0x3555, 0x3C00, 0xBC00, 0x3C01, 0x4000, 0x5640, 0x7BFF, 0xFBFF,
                0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x0200, 0x0800, 0x6400, 0x6401, 0xE400};
    }

    template <>
    std::vector<std::uint16_t> edge_patterns<Bf16>()
    {
        return {0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x0080, 0x0081, 0x3F80, 0xBF80, 0x3F81, 0x4000, 0x4B00, 0x7F7F, 0xFF7F,
                0x7F80, 0xFF80, 0x7FC0, 0x7F81, 0x3F00, 0x4380, 0x4381, 0x3C00, 0xC3C0, 0x3FC0};
    }

    template <class F>
    void test_sum_edge_matrix()
    {
        using E = typename F::Elem;
        const auto patterns = edge_patterns<F>();
        std::vector<E> a, b;
        std::vector<std::uint16_t> want;
        for (std::uint16_t x : patterns)
        {
            for (std::uint16_t y : patterns)
            {
                a.push_back(E{x});
                b.push_back(E{y});
                want.push_back(expected_sum<F>(x, y));
            }
        }
        tbccl::detail::apply_reduction(a.data(), b.data(), a.size(), tbccl::ReduceOp::Sum);
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            expect(same<F>(a[i].bits, want[i]), std::string(F::name) + ": edge sum mismatch at pair " + std::to_string(i));
        }
        // a few exact cases by hand
        const auto sum = [](std::uint16_t x, std::uint16_t y) {
            E l{x};
            const E r{y};
            tbccl::detail::apply_reduction(&l, &r, 1, tbccl::ReduceOp::Sum);
            return l.bits;
        };
        expect(sum(F::enc(1.0f), F::enc(-1.0f)) == 0x0000, std::string(F::name) + ": 1 + -1 must be +0");
        expect(sum(0x8000, 0x8000) == 0x8000, std::string(F::name) + ": -0 + -0 must be -0");
        expect(sum(0x8000, 0x0000) == 0x0000, std::string(F::name) + ": -0 + +0 must be +0");
        expect(sum(F::kLastFinite, F::kLastFinite) == F::kInf, std::string(F::name) + ": largest finite + itself overflows to +inf");
        expect(sum(static_cast<std::uint16_t>(F::kLastFinite | 0x8000), static_cast<std::uint16_t>(F::kLastFinite | 0x8000)) == (F::kInf | 0x8000),
               std::string(F::name) + ": negative overflow to -inf");
        expect(F::is_nan(sum(F::kInf, static_cast<std::uint16_t>(F::kInf | 0x8000))), std::string(F::name) + ": +inf + -inf is NaN");
        std::cout << "[PASS] test_sum_edge_matrix<" << F::name << ">\n";
    }

    template <class F>
    std::vector<std::uint16_t> random_patterns(std::size_t count, std::uint32_t seed)
    {
        std::vector<std::uint16_t> v(count);
        std::uint32_t state = seed;
        for (auto &x : v) x = static_cast<std::uint16_t>(next_random(state) >> 8);
        return v;
    }

    template <class F>
    void test_sum_random_vectors()
    {
        using E = typename F::Elem;
        for (std::size_t count : {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{17}, std::size_t{255}, std::size_t{1023}, std::size_t{1025}, std::size_t{1} << 18})
        {
            const auto x = random_patterns<F>(count, 12345u + static_cast<std::uint32_t>(count));
            const auto y = random_patterns<F>(count, 99991u + static_cast<std::uint32_t>(count));
            std::vector<E> acc(count);
            std::vector<E> other(count);
            for (std::size_t i = 0; i < count; ++i)
            {
                acc[i] = E{x[i]};
                other[i] = E{y[i]};
            }
            tbccl::detail::apply_reduction(acc.data(), other.data(), count, tbccl::ReduceOp::Sum);
            for (std::size_t i = 0; i < count; ++i)
            {
                expect(same<F>(acc[i].bits, expected_sum<F>(x[i], y[i])), std::string(F::name) + ": random sum mismatch, count " + std::to_string(count) + " index " + std::to_string(i));
            }
        }
        std::cout << "[PASS] test_sum_random_vectors<" << F::name << ">\n";
    }

    // ----- 3. real N=2 collectives -------------------------------------------------------------------------------------------------
    // Port zones used by no other test: World API cases take base, base+1 from 31000-31399; Communicator cases take base, base+1 from
    // 31850-31899 (their data plane is control port + 1000 = 32850-32899).
    std::uint16_t g_next_world_port = 31000;
    std::uint16_t g_next_comm_port = 31850;

    std::uint16_t take_world_port()
    {
        const std::uint16_t p = g_next_world_port;
        g_next_world_port = static_cast<std::uint16_t>(g_next_world_port + 4);
        expect(g_next_world_port < 31400, "test world port budget exhausted");
        return p;
    }

    std::uint16_t take_comm_port()
    {
        const std::uint16_t p = g_next_comm_port;
        g_next_comm_port = static_cast<std::uint16_t>(g_next_comm_port + 2);
        expect(g_next_comm_port < 31900, "test communicator port budget exhausted");
        return p;
    }

    using WorldAllReduce = std::function<void(tbccl::World &, const void *, void *, std::size_t, tbccl::DataType)>;

    template <class F>
    void run_world_all_reduce(const char *label, const WorldAllReduce &fn, std::size_t count)
    {
        const auto in0 = random_patterns<F>(count, 7u + static_cast<std::uint32_t>(count));
        const auto in1 = random_patterns<F>(count, 1301u + static_cast<std::uint32_t>(count));
        auto peers = tbccl_test::make_local_peers(take_world_port(), 2);
        std::vector<std::vector<std::uint16_t>> out(2, std::vector<std::uint16_t>(count, 0));
        std::vector<std::exception_ptr> errors(2);
        std::vector<std::thread> threads;
        for (std::size_t rank = 0; rank < 2; ++rank)
        {
            threads.emplace_back(
                tbccl_test::run_rank, tbccl_test::make_options(rank, peers),
                [&, rank](tbccl::World &world) { fn(world, (rank == 0 ? in0 : in1).data(), out[rank].data(), count, F::dt); }, std::ref(errors[rank]));
        }
        tbccl_test::join_and_check(threads, errors);
        for (std::size_t i = 0; i < count; ++i)
        {
            const std::uint16_t want = expected_sum<F>(in0[i], in1[i]);
            expect(same<F>(out[0][i], want) && same<F>(out[1][i], want),
                   std::string(F::name) + " " + label + ": World all_reduce mismatch, count " + std::to_string(count) + " index " + std::to_string(i));
        }
    }

    template <class F>
    void test_world_all_reduce()
    {
        const WorldAllReduce selected = [](tbccl::World &w, const void *s, void *r, std::size_t n, tbccl::DataType t) { tbccl::all_reduce(w, s, r, n, t, tbccl::ReduceOp::Sum); };
        const WorldAllReduce reference = [](tbccl::World &w, const void *s, void *r, std::size_t n, tbccl::DataType t) { tbccl::detail::all_reduce_reference(w, s, r, n, t, tbccl::ReduceOp::Sum); };
        const WorldAllReduce ring = [](tbccl::World &w, const void *s, void *r, std::size_t n, tbccl::DataType t) { tbccl::detail::all_reduce_ring(w, s, r, n, t, tbccl::ReduceOp::Sum); };
        const WorldAllReduce pipelined = [](tbccl::World &w, const void *s, void *r, std::size_t n, tbccl::DataType t) {
            tbccl::detail::all_reduce_pipelined(w, s, r, n, t, tbccl::ReduceOp::Sum, 1024);
        };
        for (std::size_t count : {std::size_t{1}, std::size_t{17}, std::size_t{4096}, std::size_t{100000}})
        {
            run_world_all_reduce<F>("selected", selected, count);
            run_world_all_reduce<F>("reference", reference, count);
        }
        for (std::size_t count : {std::size_t{2}, std::size_t{18}, std::size_t{4096}, std::size_t{100000}})
        {
            run_world_all_reduce<F>("ring", ring, count);
            run_world_all_reduce<F>("pipelined", pipelined, count);
        }
        std::cout << "[PASS] test_world_all_reduce<" << F::name << ">\n";
    }

    void run_pair(std::uint16_t port, const std::function<void(tbccl::Communicator &)> &rank0_fn, const std::function<void(tbccl::Communicator &)> &rank1_fn)
    {
        tbccl::CommunicatorOptions opts0;
        opts0.rank = 0;
        opts0.peers = {{"127.0.0.1", port}, {"127.0.0.1", static_cast<std::uint16_t>(port + 1)}};
        tbccl::CommunicatorOptions opts1 = opts0;
        opts1.rank = 1;
        std::exception_ptr err0, err1;
        std::thread t1([&] {
            try { auto c = tbccl::Communicator::create(opts1); rank1_fn(*c); } catch (...) { err1 = std::current_exception(); }
        });
        try { auto c = tbccl::Communicator::create(opts0); rank0_fn(*c); } catch (...) { err0 = std::current_exception(); }
        t1.join();
        if (err0) std::rethrow_exception(err0);
        if (err1) std::rethrow_exception(err1);
    }

    template <class F>
    void test_communicator_all_reduce()
    {
        for (std::size_t count : {std::size_t{1}, std::size_t{17}, std::size_t{2048}, std::size_t{524288}})
        {
            for (bool in_place : {true, false})
            {
                const auto in0 = random_patterns<F>(count, 31u + static_cast<std::uint32_t>(count));
                const auto in1 = random_patterns<F>(count, 977u + static_cast<std::uint32_t>(count));
                std::vector<std::uint16_t> out0 = in0, out1 = in1, dst0(count, 0), dst1(count, 0);
                auto body = [&](tbccl::Communicator &comm, std::vector<std::uint16_t> &inout, std::vector<std::uint16_t> &dst) {
                    tbccl::BufferView in_view{tbccl::MemoryKind::Host, inout.data(), inout.size() * 2, -1};
                    tbccl::BufferView out_view{tbccl::MemoryKind::Host, dst.data(), dst.size() * 2, -1};
                    tbccl::Work work = in_place ? comm.all_reduce(in_view, in_view, count, F::dt, tbccl::ReduceOp::Sum)
                                                : comm.all_reduce(in_view, out_view, count, F::dt, tbccl::ReduceOp::Sum);
                    work.wait();
                    expect(!work.has_error(), std::string(F::name) + ": Communicator::all_reduce failed: " + work.error());
                };
                run_pair(take_comm_port(), [&](tbccl::Communicator &c) { body(c, out0, dst0); }, [&](tbccl::Communicator &c) { body(c, out1, dst1); });
                const auto &r0 = in_place ? out0 : dst0;
                const auto &r1 = in_place ? out1 : dst1;
                for (std::size_t i = 0; i < count; ++i)
                {
                    const std::uint16_t want = expected_sum<F>(in0[i], in1[i]);
                    expect(same<F>(r0[i], want) && same<F>(r1[i], want),
                           std::string(F::name) + ": Communicator all_reduce mismatch, count " + std::to_string(count) + (in_place ? " in-place" : " out-of-place") +
                               " index " + std::to_string(i));
                }
            }
        }
        std::cout << "[PASS] test_communicator_all_reduce<" << F::name << ">\n";
    }

    void test_capabilities()
    {
        run_pair(take_comm_port(),
                 [](tbccl::Communicator &c) {
                     const auto &caps = c.capabilities();
                     for (tbccl::DataType t : {tbccl::DataType::Float16, tbccl::DataType::BFloat16})
                     {
                         expect(caps.supports_collective_all_reduce(tbccl::MemoryKind::Host, t, tbccl::ReduceOp::Sum), "host must support 16-bit float sum");
                         expect(!caps.supports_collective_all_reduce(tbccl::MemoryKind::Host, t, tbccl::ReduceOp::Product), "product is not supported for 16-bit floats");
                     }
                     expect(caps.supports_collective_all_reduce(tbccl::MemoryKind::Host, tbccl::DataType::Float32, tbccl::ReduceOp::Sum), "float32 sum unchanged");
                 },
                 [](tbccl::Communicator &) {});
        std::cout << "[PASS] test_capabilities\n";
    }

} // namespace

int main()
{
    try
    {
        test_decode_exhaustive<Fp16>();
        test_decode_exhaustive<Bf16>();
        test_encode_rounding_exhaustive<Fp16>();
        test_encode_rounding_exhaustive<Bf16>();
        test_special_vectors();
        test_compiler_oracle();
        test_sum_edge_matrix<Fp16>();
        test_sum_edge_matrix<Bf16>();
        test_sum_random_vectors<Fp16>();
        test_sum_random_vectors<Bf16>();
        test_world_all_reduce<Fp16>();
        test_world_all_reduce<Bf16>();
        test_communicator_all_reduce<Fp16>();
        test_communicator_all_reduce<Bf16>();
        test_capabilities();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
    std::cout << "All low-precision reduction tests passed.\n";
    return 0;
}
