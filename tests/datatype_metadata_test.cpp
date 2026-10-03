// Phase 49: reduction datatype metadata. The numeric enum values are fixed (appended-only), datatype_size() is the single size
// table, reduction_supported()/validate_reduction() are the single support predicate, and an unsupported (dtype, op) pair is
// rejected before ANY communication on every public reducing entry point (a world whose send/recv throw proves it).

#include <tbccl/collectives.hpp>
#include <tbccl/reduction.hpp>
#include <tbccl/world.hpp>

#include "test_utils.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using tbccl_test::expect;

namespace
{

    static_assert(static_cast<int>(tbccl::DataType::Int32) == 0, "DataType values are ABI-visible");
    static_assert(static_cast<int>(tbccl::DataType::Int64) == 1, "DataType values are ABI-visible");
    static_assert(static_cast<int>(tbccl::DataType::Float32) == 2, "DataType values are ABI-visible");
    static_assert(static_cast<int>(tbccl::DataType::Float64) == 3, "DataType values are ABI-visible");
    static_assert(static_cast<int>(tbccl::DataType::Int8) == 4, "appended in Phase 49");
    static_assert(static_cast<int>(tbccl::DataType::UInt8) == 5, "appended in Phase 49");
    static_assert(static_cast<int>(tbccl::DataType::Float16) == 6, "appended in Phase 49");
    static_assert(static_cast<int>(tbccl::DataType::BFloat16) == 7, "appended in Phase 49");

    using tbccl::DataType;
    using tbccl::ReduceOp;

    constexpr std::array<DataType, 8> kAllTypes = {
        DataType::Int32, DataType::Int64, DataType::Float32, DataType::Float64,
        DataType::Int8, DataType::UInt8, DataType::Float16, DataType::BFloat16};
    constexpr std::array<DataType, 4> kOriginalTypes = {DataType::Int32, DataType::Int64, DataType::Float32, DataType::Float64};
    constexpr std::array<DataType, 4> kNewTypes = {DataType::Int8, DataType::UInt8, DataType::Float16, DataType::BFloat16};
    constexpr std::array<ReduceOp, 4> kAllOps = {ReduceOp::Sum, ReduceOp::Product, ReduceOp::Min, ReduceOp::Max};

    // A World whose send()/recv() throw if a collective ever reaches them.
    class FailingWorld : public tbccl::World
    {
    public:
        FailingWorld(std::size_t rank, std::size_t size) : rank_(rank), size_(size) {}
        std::size_t rank() const noexcept override { return rank_; }
        std::size_t size() const noexcept override { return size_; }
        void send(std::size_t, const void *, std::size_t) override { throw std::runtime_error("FailingWorld::send reached"); }
        void recv(std::size_t, void *, std::size_t) override { throw std::runtime_error("FailingWorld::recv reached"); }

    private:
        std::size_t rank_;
        std::size_t size_;
    };

    void test_sizes_and_labels()
    {
        expect(tbccl::datatype_size(DataType::Int8) == 1, "int8 size");
        expect(tbccl::datatype_size(DataType::UInt8) == 1, "uint8 size");
        expect(tbccl::datatype_size(DataType::Float16) == 2, "float16 size");
        expect(tbccl::datatype_size(DataType::BFloat16) == 2, "bfloat16 size");
        expect(tbccl::datatype_size(DataType::Int32) == 4, "int32 size");
        expect(tbccl::datatype_size(DataType::Float32) == 4, "float32 size");
        expect(tbccl::datatype_size(DataType::Int64) == 8, "int64 size");
        expect(tbccl::datatype_size(DataType::Float64) == 8, "float64 size");

        expect(std::string(tbccl::datatype_label(DataType::BFloat16)) == "bfloat16", "bfloat16 label");
        expect(std::string(tbccl::datatype_label(DataType::UInt8)) == "uint8", "uint8 label");
        expect(std::string(tbccl::reduce_op_label(ReduceOp::Product)) == "product", "product label");
        expect(std::string(tbccl::datatype_label(static_cast<DataType>(99))) == "unknown", "out-of-range label");

        bool threw = false;
        try { tbccl::datatype_size(static_cast<DataType>(99)); } catch (const std::runtime_error &) { threw = true; }
        expect(threw, "datatype_size must throw on an unknown DataType");
        std::cout << "[PASS] test_sizes_and_labels\n";
    }

    void test_support_matrix()
    {
        for (DataType type : kOriginalTypes)
        {
            for (ReduceOp op : kAllOps)
            {
                expect(tbccl::reduction_supported(type, op), "original types keep every ReduceOp");
                tbccl::validate_reduction(type, op);
            }
        }
        for (DataType type : kNewTypes)
        {
            for (ReduceOp op : {ReduceOp::Product, ReduceOp::Min, ReduceOp::Max})
            {
                expect(!tbccl::reduction_supported(type, op), "new types support Sum only");
            }
        }
        expect(!tbccl::reduction_supported(static_cast<DataType>(99), ReduceOp::Sum), "unknown dtype is unsupported");
        std::cout << "[PASS] test_support_matrix\n";
    }

    void test_validation_message()
    {
        bool threw = false;
        try
        {
            tbccl::validate_reduction(DataType::Float16, ReduceOp::Product);
        }
        catch (const std::runtime_error &error)
        {
            threw = true;
            const std::string message = error.what();
            expect(message.rfind("unsupported: reduction dtype=float16 op=product", 0) == 0, "message prefix: " + message);
            expect(message.find("supported ops for this dtype:") != std::string::npos, "message lists supported ops: " + message);
        }
        expect(threw, "float16 + product must be rejected");
        std::cout << "[PASS] test_validation_message\n";
    }

    // Float16/BFloat16/Int8/UInt8 + Product/Min/Max must fail identically and immediately on every public reducing entry point,
    // without touching the world, so a peer that never calls cannot be left blocked.
    void test_rejected_before_communication()
    {
        FailingWorld world(/*rank=*/0, /*size=*/2);
        std::array<std::uint8_t, 64> send{};
        std::array<std::uint8_t, 64> recv{};

        for (DataType type : kNewTypes)
        {
            for (ReduceOp op : {ReduceOp::Product, ReduceOp::Min, ReduceOp::Max})
            {
                const std::string what = std::string(tbccl::datatype_label(type)) + "/" + tbccl::reduce_op_label(op);
                auto expect_unsupported = [&](const char *entry, auto &&call) {
                    bool threw = false;
                    try { call(); }
                    catch (const std::runtime_error &error)
                    {
                        threw = true;
                        expect(std::string(error.what()).rfind("unsupported: reduction", 0) == 0,
                               std::string(entry) + " " + what + ": wrong error: " + error.what());
                    }
                    expect(threw, std::string(entry) + " " + what + " must be rejected");
                };
                expect_unsupported("reduce", [&] { tbccl::reduce(world, send.data(), recv.data(), 4, type, op, 0); });
                expect_unsupported("all_reduce", [&] { tbccl::all_reduce(world, send.data(), recv.data(), 4, type, op); });
                expect_unsupported("reduce_scatter", [&] { tbccl::reduce_scatter(world, send.data(), recv.data(), 2, type, op); });
            }
        }
        std::cout << "[PASS] test_rejected_before_communication\n";
    }

} // namespace

int main()
{
    try
    {
        test_sizes_and_labels();
        test_support_matrix();
        test_validation_message();
        test_rejected_before_communication();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
    std::cout << "All datatype metadata tests passed.\n";
    return 0;
}
