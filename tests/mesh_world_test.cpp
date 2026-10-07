// mesh_test::World: independent worlds coexist, and one world serves many bodies on the same communicators (the property the
// heavy matrix tests rely on to keep their loopback connection count, and so their TIME_WAIT footprint, bounded).

#include "mesh_test_support.hpp"

#include <cstdint>
#include <iostream>
#include <set>

using namespace mesh_test;
using tbccl::BufferView;
using tbccl::DataType;
using tbccl::MemoryKind;
using tbccl::ReduceOp;

namespace
{
    constexpr std::size_t kWorlds = 6;
    constexpr std::size_t kWorldSize = 3;
    constexpr int kBodies = 20;

    void exercise(std::size_t index, std::vector<std::string> &errors)
    {
        try
        {
            World world(kWorldSize);
            std::set<const tbccl::Communicator *> seen;
            std::mutex m;
            for (int body = 0; body < kBodies; ++body)
            {
                world.run([&](std::size_t rank, tbccl::Communicator &comm) {
                    {
                        std::lock_guard<std::mutex> lock(m);
                        seen.insert(&comm);
                    }
                    std::vector<std::int32_t> v(8, static_cast<std::int32_t>(rank + 1 + body));
                    BufferView view{MemoryKind::Host, v.data(), v.size() * sizeof(std::int32_t), 0};
                    tbccl::Work w = comm.all_reduce(view, view, v.size(), DataType::Int32, ReduceOp::Sum);
                    w.wait();
                    expect(!w.has_error(), "all_reduce: " + w.error());
                    const std::int32_t want = static_cast<std::int32_t>(1 + 2 + 3 + 3 * body);
                    for (auto x : v) expect(x == want, "all_reduce result in world " + std::to_string(index) + " body " + std::to_string(body));
                });
            }
            expect(seen.size() == kWorldSize, "every body ran on the same communicators");
        }
        catch (const std::exception &e)
        {
            errors[index] = e.what();
        }
    }
} // namespace

int main()
{
    std::vector<std::string> errors(kWorlds);
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < kWorlds; ++i) threads.emplace_back(exercise, i, std::ref(errors));
    for (auto &t : threads) t.join();
    for (std::size_t i = 0; i < kWorlds; ++i)
    {
        if (!errors[i].empty())
        {
            std::cerr << "[FAIL] world " << i << ": " << errors[i] << "\n";
            return 1;
        }
    }
    std::cout << "[PASS] " << kWorlds << " concurrent worlds, each reused for " << kBodies << " bodies\n";
    return 0;
}
