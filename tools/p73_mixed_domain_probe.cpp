// Direct libtbccl probe of P2P + collective traffic on ONE communicator, below every framework adapter (Host buffers, loopback, W2).
//
// Each scenario gives every rank an ordered list of operations submitted back to back WITHOUT waiting (C = a collective of the chosen family, S = send to the
// peer, R = recv from the peer), with seeded random submission jitter, and then waits for all the Works under a watchdog. Payloads are deterministic sentinels, so a
// wrong result says whose bytes arrived where. Scenarios cover the same relative order on both ranks and the opposite order, which the intended contract says
// must be legal: collectives and P2P are independent ordering domains.
//
//   p73_mixed_domain_probe [--gate] [--seed N] [--family AR|BC|AG|BAR] [--bytes N]...
//
// Without --gate it only reports (exit 0): this is how the pre-repair behaviour is recorded. With --gate any scenario that is not PASS fails the run.
#include "../tests/mesh_test_support.hpp"

#include <tbccl/communicator.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <random>
#include <sstream>

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
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint8_t>(from * 131 + tag * 17 + i * 7 + (i >> 8) + 1);
    return v;
}

std::int32_t ar_value(std::size_t rank, std::size_t i) { return static_cast<std::int32_t>(i * 7 + rank * 1000003 + 12345); }

struct Scenario
{
    std::string name;
    std::string ops[2]; // tokens per rank: C S R
};

struct Outcome
{
    std::string status = "PASS"; // PASS | WRONG | ERROR | HANG
    std::string detail;
};

std::string classify(const std::vector<std::uint8_t> &got, std::size_t rank_peer, std::size_t tag, const std::vector<std::uint8_t> &want, const std::vector<std::uint8_t> &ar_in_other)
{
    std::size_t bad = 0, first = got.size();
    for (std::size_t i = 0; i < got.size(); ++i)
        if (got[i] != want[i])
        {
            ++bad;
            first = std::min(first, i);
        }
    std::ostringstream o;
    o << bad << "/" << got.size() << " bytes wrong, first at " << first;
    if (got == ar_in_other) o << " (the buffer holds the other rank's all_reduce input)";
    (void)rank_peer;
    (void)tag;
    return o.str();
}

Outcome run_scenario(const Scenario &sc, const std::string &family, std::size_t bytes, unsigned seed, std::chrono::milliseconds watchdog)
{
    Outcome out;
    std::vector<Outcome> per(2);
    std::atomic<bool> any_hang{false};
    auto results = bootstrap(healthy_slots(2), std::chrono::seconds(20));
    for (std::size_t r = 0; r < 2; ++r)
        if (!results[r].comm) return {"ERROR", "bootstrap: " + results[r].error};
    std::vector<std::thread> threads;
    for (std::size_t rank = 0; rank < 2; ++rank)
    {
        threads.emplace_back([&, rank] {
            tbccl::Communicator &comm = *results[rank].comm;
            const std::size_t peer = 1 - rank;
            std::mt19937 rng(seed * 7919u + static_cast<unsigned>(rank) * 104729u);
            auto jitter = [&] { std::this_thread::sleep_for(std::chrono::microseconds(rng() % 1500)); };

            // one all_reduce buffer per collective in the scenario: a second in-place all_reduce on the same buffer would legitimately return twice the sum
            const std::size_t ncoll = static_cast<std::size_t>(std::count(sc.ops[rank].begin(), sc.ops[rank].end(), 'C'));
            std::vector<std::vector<std::int32_t>> ars(std::max<std::size_t>(ncoll, 1), std::vector<std::int32_t>(bytes / 4));
            for (auto &a : ars)
                for (std::size_t i = 0; i < a.size(); ++i) a[i] = ar_value(rank, i);
            std::size_t next_ar = 0;
            std::vector<std::int32_t> &ar = ars[0];
            std::vector<std::uint8_t> bc = rank == 0 ? pattern(0, 5, bytes) : std::vector<std::uint8_t>(bytes, 0xEE);
            std::vector<std::uint8_t> ag_in = pattern(rank, 6, bytes), ag_out0(bytes, 0xEE), ag_out1(bytes, 0xEE);
            std::vector<std::uint8_t> tx = pattern(rank, 9, bytes), rx(bytes, 0xEE);
            std::vector<tbccl::Work> works;
            std::vector<std::string> names;
            try
            {
                for (char tok : sc.ops[rank])
                {
                    if (tok == ' ') continue;
                    if (tok == 'C')
                    {
                        if (family == "AR")
                        {
                            auto &a = ars[next_ar++];
                            works.push_back(comm.all_reduce(BufferView{MemoryKind::Host, a.data(), a.size() * 4, 0}, BufferView{MemoryKind::Host, a.data(), a.size() * 4, 0}, a.size(), DataType::Int32, tbccl::ReduceOp::Sum));
                        }
                        else if (family == "BC") works.push_back(comm.broadcast(view(bc), 0));
                        else if (family == "AG") works.push_back(comm.all_gather(view(ag_in), {view(ag_out0), view(ag_out1)}));
                        else works.push_back(comm.barrier());
                        names.push_back("collective");
                    }
                    else if (tok == 'S')
                    {
                        works.push_back(comm.send(view(tx), tx.size(), DataType::UInt8, peer));
                        names.push_back("send");
                    }
                    else
                    {
                        works.push_back(comm.recv(view(rx), rx.size(), DataType::UInt8, peer));
                        names.push_back("recv");
                    }
                    jitter();
                }
                const auto deadline = std::chrono::steady_clock::now() + watchdog;
                for (std::size_t i = 0; i < works.size(); ++i)
                {
                    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
                    if (!works[i].wait_for(std::max(left, std::chrono::milliseconds(1))))
                    {
                        per[rank] = {"HANG", names[i] + " Work #" + std::to_string(i) + " did not finish in " + std::to_string(watchdog.count()) + " ms"};
                        any_hang = true;
                        comm.abort("probe watchdog");
                        return;
                    }
                    if (works[i].has_error())
                    {
                        per[rank] = {"ERROR", names[i] + " Work #" + std::to_string(i) + ": " + works[i].error()};
                        comm.abort("probe: a Work failed");
                        return;
                    }
                }
                // verify only what this rank was supposed to hold
                std::vector<std::string> wrong;
                const bool has_c = sc.ops[rank].find('C') != std::string::npos;
                const bool has_r = sc.ops[rank].find('R') != std::string::npos;
                if (has_r && rx != pattern(peer, 9, bytes))
                {
                    std::vector<std::uint8_t> ar_bytes(bytes);
                    std::memcpy(ar_bytes.data(), ars[0].data(), bytes);
                    wrong.push_back("recv buffer: " + classify(rx, peer, 9, pattern(peer, 9, bytes), ar_bytes));
                }
                if (has_c && family == "AR")
                    for (std::size_t k = 0; k < ncoll; ++k)
                        for (std::size_t i = 0; i < ars[k].size(); ++i)
                            if (ars[k][i] != ar_value(0, i) + ar_value(1, i))
                            {
                                wrong.push_back("all_reduce #" + std::to_string(k) + " result wrong at element " + std::to_string(i));
                                break;
                            }
                if (has_c && family == "BC" && bc != pattern(0, 5, bytes)) wrong.push_back("broadcast buffer wrong");
                if (has_c && family == "AG" && (ag_out0 != pattern(0, 6, bytes) || ag_out1 != pattern(1, 6, bytes))) wrong.push_back("all_gather outputs wrong");
                if (!wrong.empty())
                {
                    std::string d;
                    for (auto &w : wrong) d += (d.empty() ? "" : "; ") + w;
                    per[rank] = {"WRONG", d};
                }
            }
            catch (const std::exception &e)
            {
                per[rank] = {"ERROR", std::string("exception: ") + e.what()};
                comm.abort("probe exception");
            }
        });
    }
    for (auto &t : threads) t.join();
    results.clear();
    for (std::size_t r = 0; r < 2; ++r)
    {
        if (per[r].status != "PASS" && (out.status == "PASS" || per[r].status == "WRONG"))
        {
            out.status = per[r].status;
        }
        if (per[r].status != "PASS") out.detail += (out.detail.empty() ? "" : " | ") + std::string("r") + std::to_string(r) + ": " + per[r].detail;
    }
    return out;
}

std::vector<Scenario> scenarios()
{
    return {
        {"same order: C then S / C then R", {"C S", "C R"}},
        {"same order: S then C / R then C", {"S C", "R C"}},
        {"HEADLINE opposite: C,S / R,C", {"C S", "R C"}},
        {"mirror opposite: S,C / C,R", {"S C", "C R"}},
        {"reverse direction headline: C,R / S,C", {"C R", "S C"}},
        {"full duplex, same order: S,R,C / S,R,C", {"S R C", "S R C"}},
        {"full duplex, opposite: S,R,C / C,S,R", {"S R C", "C S R"}},
        {"multi P2P, same order: S,C,S / R,C,R", {"S C S", "R C R"}},
        {"multi P2P, opposite: S,C,S / R,R,C", {"S C S", "R R C"}},
        {"multi collective, same order: C,S,C / C,R,C", {"C S C", "C R C"}},
        {"multi collective, opposite: C,S,C / R,C,C", {"C S C", "R C C"}},
    };
}

} // namespace

int main(int argc, char **argv)
{
    bool gate = false;
    unsigned seed = 1;
    std::vector<std::string> families = {"AR", "BC", "AG", "BAR"};
    std::vector<std::size_t> sizes;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--gate") gate = true;
        else if (a == "--seed" && i + 1 < argc) seed = static_cast<unsigned>(std::atoi(argv[++i]));
        else if (a == "--family" && i + 1 < argc) families = {argv[++i]};
        else if (a == "--bytes" && i + 1 < argc) sizes.push_back(static_cast<std::size_t>(std::atoll(argv[++i])));
    }
    if (sizes.empty()) sizes = {4096, 1u << 20};
    std::size_t total = 0, pass = 0;
    std::map<std::string, std::size_t> by_status;
    for (const auto &fam : families)
        for (const std::size_t bytes : sizes)
            for (const auto &sc : scenarios())
            {
                const auto o = run_scenario(sc, fam, fam == "BAR" ? bytes : bytes, seed, std::chrono::milliseconds(8000));
                ++total;
                ++by_status[o.status];
                if (o.status == "PASS") ++pass;
                std::cout << o.status << "  " << fam << " " << bytes << "B  " << sc.name << (o.detail.empty() ? "" : "  -> " + o.detail) << "\n" << std::flush;
            }
    std::cout << "SUMMARY seed=" << seed << " total=" << total << " pass=" << pass;
    for (const auto &kv : by_status)
        if (kv.first != "PASS") std::cout << " " << kv.first << "=" << kv.second;
    std::cout << "\n";
    return gate && pass != total ? 1 : 0;
}
