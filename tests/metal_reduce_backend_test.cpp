// A dedicated Metal-shared local-reduction correctness
// test, real Mac hardware only. Verifies HostReduceBackend (reused
// unchanged for Metal-shared -- Metal-shared memory is already
// CPU-addressable, so no separate GPU-side Metal reduction kernel is
// needed) against TWO real Metal-shared allocations' own CPU-visible
// memory -- not a plain host std::vector standing in for it.
//
// Plain .cpp (not .mm): TensorBackend's public interface
// (tensor_backend.hpp) is ordinary C++; only its Metal .mm implementation
// needs Objective-C++, and that stays linked-in but unseen here, matching
// this file's sibling tensor_metal_test.mm's own scope (which tests the
// TensorBackend contract itself, not local reduction).

#include "tensor/host_reduce_backend.hpp"
#include "tensor/tensor_backend.hpp"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using tbccl_bench::tensor::BackendKind;
    using tbccl_bench::tensor::HostReduceBackend;
    using tbccl_bench::tensor::make_backend;

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    float value_for(std::size_t i, std::uint32_t seed, std::uint32_t modulus)
    {
        return static_cast<float>((i + seed) % modulus);
    }

    // Writes `count` deterministic float32 values directly into a real
    // Metal-shared backend's (mutable, CPU-visible) destination buffer --
    // used here purely as "a real Metal-shared allocation to hold a test
    // value," independent of the normal network-receive role that buffer
    // plays in n2_all_reduce_tensor().
    void fill_metal_shared(
        tbccl_bench::tensor::TensorBackend &backend,
        std::uint32_t seed, std::uint32_t modulus, std::size_t count)
    {
        auto *data = static_cast<float *>(backend.destination_staging_data());
        for (std::size_t i = 0; i < count; ++i)
        {
            data[i] = value_for(i, seed, modulus);
        }
    }

    void run_one_case(std::size_t count, std::uint32_t local_seed,
                       std::uint32_t local_mod, std::uint32_t peer_seed,
                       std::uint32_t peer_mod)
    {
        const std::size_t bytes = count * sizeof(float);

        expect(tbccl_bench::tensor::backend_kind_available(BackendKind::MetalShared),
               "metal-shared backend must be available in this build");

        auto backend_local = make_backend(BackendKind::MetalShared);
        backend_local->allocate(bytes);
        auto backend_peer = make_backend(BackendKind::MetalShared);
        backend_peer->allocate(bytes);

        fill_metal_shared(*backend_local, local_seed, local_mod, count);
        fill_metal_shared(*backend_peer, peer_seed, peer_mod, count);

        std::vector<float> expected(count);
        const auto *local_before =
            static_cast<const float *>(backend_local->destination_staging_data());
        const auto *peer_before =
            static_cast<const float *>(backend_peer->destination_staging_data());
        for (std::size_t i = 0; i < count; ++i)
        {
            expected[i] = local_before[i] + peer_before[i];
        }

        HostReduceBackend reduce(
            backend_local->destination_staging_data(),
            backend_peer->destination_staging_data());
        reduce.reduce_sum(count, tbccl::DataType::Float32);

        const auto *result =
            static_cast<const float *>(backend_local->destination_staging_data());
        for (std::size_t i = 0; i < count; ++i)
        {
            expect(result[i] == expected[i],
                   "element " + std::to_string(i) + " mismatch for count=" +
                       std::to_string(count));
        }
    }

    void test_initial_values_are_distinct()
    {
        auto backend_local = make_backend(BackendKind::MetalShared);
        backend_local->allocate(1024 * sizeof(float));
        auto backend_peer = make_backend(BackendKind::MetalShared);
        backend_peer->allocate(1024 * sizeof(float));

        fill_metal_shared(*backend_local, 11, 127, 1024);
        fill_metal_shared(*backend_peer, 97, 113, 1024);

        const auto *local =
            static_cast<const float *>(backend_local->destination_staging_data());
        const auto *peer =
            static_cast<const float *>(backend_peer->destination_staging_data());

        bool any_different = false;
        for (std::size_t i = 0; i < 1024; ++i)
        {
            if (local[i] != peer[i])
            {
                any_different = true;
                break;
            }
        }
        expect(any_different, "local and peer buffers must actually hold distinct values before reduction");

        std::cout << "[PASS] test_initial_values_are_distinct\n";
    }

    void test_basic_sum()
    {
        run_one_case(1024, 11, 127, 97, 113);
        std::cout << "[PASS] test_basic_sum\n";
    }

    void test_single_element()
    {
        run_one_case(1, 3, 127, 9, 113);
        std::cout << "[PASS] test_single_element\n";
    }

    void test_odd_count()
    {
        run_one_case(1001, 5, 127, 19, 113);
        std::cout << "[PASS] test_odd_count\n";
    }

    void test_repeated_reuse()
    {
        // Same two backend allocations, reused across several rounds with
        // distinct seeds each time -- the "repeated reuse" check, proving
        // no leftover state contaminates a later round.
        constexpr std::size_t kCount = 4096;
        const std::size_t bytes = kCount * sizeof(float);

        auto backend_local = make_backend(BackendKind::MetalShared);
        backend_local->allocate(bytes);
        auto backend_peer = make_backend(BackendKind::MetalShared);
        backend_peer->allocate(bytes);

        for (std::uint32_t round = 0; round < 5; ++round)
        {
            fill_metal_shared(*backend_local, round, 127, kCount);
            fill_metal_shared(*backend_peer, round + 50, 113, kCount);

            std::vector<float> expected(kCount);
            const auto *local_before =
                static_cast<const float *>(backend_local->destination_staging_data());
            const auto *peer_before =
                static_cast<const float *>(backend_peer->destination_staging_data());
            for (std::size_t i = 0; i < kCount; ++i)
            {
                expected[i] = local_before[i] + peer_before[i];
            }

            HostReduceBackend reduce(
                backend_local->destination_staging_data(),
                backend_peer->destination_staging_data());
            reduce.reduce_sum(kCount, tbccl::DataType::Float32);

            const auto *result =
                static_cast<const float *>(backend_local->destination_staging_data());
            for (std::size_t i = 0; i < kCount; ++i)
            {
                expect(result[i] == expected[i],
                       "round " + std::to_string(round) + " element " +
                           std::to_string(i) + " mismatch");
            }
        }

        std::cout << "[PASS] test_repeated_reuse\n";
    }

} // namespace

int main()
{
    try
    {
        test_initial_values_are_distinct();
        test_basic_sum();
        test_single_element();
        test_odd_count();
        test_repeated_reuse();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";
    return 0;
}
