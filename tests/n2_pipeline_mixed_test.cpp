// Two ranks that run DIFFERENT two-rank all_reduce schedules must still agree and finish: the pipelined schedule (TBCCL_ALLREDUCE_PIPELINE_BYTES=n, any chunk size) and the
// sequential one (unset or 0) put identical bytes on the wire in each direction, and neither waits for the other's schedule. The setting is latched per process, so each case
// starts two child processes (this binary re-executed) with their own environment; the parent kills them and fails if they do not finish in time (a deadlock between
// mismatched ranks would otherwise only show up as a hang). The same pairings are run against the previous release's binary by the benchmark tooling.

#include <tbccl/communicator.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace
{
// A rank's listening port p. The static (legacy) peer list also uses p + 1000 for its data connection, so both must be free and below 65536; ephemeral ports (the kernel's
// choice) can be too high on macOS. Random candidates in a fixed range are probed by binding.
bool bindable(std::uint16_t port)
{
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port = htons(port);
    const bool ok = ::bind(fd, reinterpret_cast<sockaddr *>(&a), sizeof a) == 0;
    ::close(fd);
    return ok;
}

std::uint16_t free_port(std::uint16_t avoid)
{
    static unsigned counter = static_cast<unsigned>(::getpid()) * 7919u;
    for (;;)
    {
        counter = counter * 1103515245u + 12345u;
        const std::uint16_t port = static_cast<std::uint16_t>(20000 + (counter >> 8) % 30000);
        if (port != avoid && port != avoid + 1000 && port + 1000 != avoid && bindable(port) && bindable(static_cast<std::uint16_t>(port + 1000))) return port;
    }
}

float expected_value(std::size_t i) { return static_cast<float>((i % 13 + 1) + (i % 7 + 20)); } // rank 0 holds i % 13 + 1, rank 1 holds i % 7 + 20

int child(std::size_t rank, std::uint16_t port0, std::uint16_t port1)
{
    tbccl::CommunicatorOptions o;
    o.rank = rank;
    o.peers.push_back({"127.0.0.1", port0});
    o.peers.push_back({"127.0.0.1", port1});
    o.bootstrap_timeout = std::chrono::milliseconds(60000);
    auto comm = tbccl::Communicator::create(o);
    // 24 MiB: far beyond the socket buffers, so a rank that sends everything before it receives (the sequential non-root) fills the peer's buffers while the peer keeps working
    for (std::size_t count : {std::size_t{1}, std::size_t{16384}, std::size_t{100003}, std::size_t{262145}, std::size_t{1000003}, (std::size_t{24} << 20) / 4 + 3})
        for (int round = 0; round < 3; ++round)
        {
            std::vector<float> data(count);
            for (std::size_t i = 0; i < count; ++i) data[i] = rank == 0 ? static_cast<float>(i % 13 + 1) : static_cast<float>(i % 7 + 20);
            const tbccl::BufferView v{tbccl::MemoryKind::Host, data.data(), count * 4, -1};
            auto w = comm->all_reduce(v, v, count, tbccl::DataType::Float32, tbccl::ReduceOp::Sum);
            if (!w.wait_for(120s) || w.has_error())
            {
                std::fprintf(stderr, "rank %zu: all_reduce of %zu elements failed: %s\n", rank, count, w.has_error() ? w.error().c_str() : "timeout");
                return 3;
            }
            for (std::size_t i = 0; i < count; ++i)
                if (data[i] != expected_value(i))
                {
                    std::fprintf(stderr, "rank %zu: element %zu of %zu is %f, expected %f\n", rank, i, count, static_cast<double>(data[i]), static_cast<double>(expected_value(i)));
                    return 4;
                }
        }
    comm->barrier().wait();
    return 0;
}

bool run_case(const char *self, const char *env0, const char *env1)
{
    const std::uint16_t p0 = free_port(0), p1 = free_port(p0);
    pid_t pids[2];
    for (int r = 0; r < 2; ++r)
    {
        pids[r] = ::fork();
        if (pids[r] == 0)
        {
            const char *env = r == 0 ? env0 : env1;
            if (*env) ::setenv("TBCCL_ALLREDUCE_PIPELINE_BYTES", env, 1);
            else ::unsetenv("TBCCL_ALLREDUCE_PIPELINE_BYTES");
            const std::string rank = std::to_string(r), a = std::to_string(p0), b = std::to_string(p1);
            ::execl(self, self, "--child", rank.c_str(), a.c_str(), b.c_str(), static_cast<char *>(nullptr));
            std::_Exit(127);
        }
    }
    bool ok = true;
    const auto deadline = std::chrono::steady_clock::now() + 240s;
    int done = 0;
    bool finished[2] = {false, false};
    while (done < 2)
    {
        for (int r = 0; r < 2; ++r)
        {
            if (finished[r]) continue;
            int status = 0;
            if (::waitpid(pids[r], &status, WNOHANG) == pids[r])
            {
                finished[r] = true;
                ++done;
                if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
                {
                    std::fprintf(stderr, "case [rank0='%s' rank1='%s']: rank %d ended with status %d\n", env0, env1, r, status);
                    ok = false;
                }
            }
        }
        if (done < 2 && std::chrono::steady_clock::now() > deadline)
        {
            std::fprintf(stderr, "case [rank0='%s' rank1='%s']: DEADLOCK or stall, ranks did not finish within 240 s\n", env0, env1);
            for (int r = 0; r < 2; ++r)
                if (!finished[r])
                {
                    ::kill(pids[r], SIGKILL);
                    ::waitpid(pids[r], nullptr, 0);
                }
            return false;
        }
        std::this_thread::sleep_for(20ms);
    }
    std::printf("%s case [rank0='%s' rank1='%s']\n", ok ? "PASS" : "FAIL", env0, env1);
    std::fflush(stdout);
    return ok;
}
} // namespace

int main(int argc, char **argv)
{
    if (argc == 5 && std::strcmp(argv[1], "--child") == 0) return child(std::strtoull(argv[2], nullptr, 10), static_cast<std::uint16_t>(std::atoi(argv[3])), static_cast<std::uint16_t>(std::atoi(argv[4])));
    bool ok = true;
    // (rank 0 = the root, rank 1 = the non-root)
    const char *cases[][2] = {{"", ""},         {"0", "0"},          {"65536", ""},    {"", "65536"},     {"65536", "0"},     {"0", "65536"},
                              {"4096", "65536"}, {"262144", "8192"}, {"65536", "65536"}, {"1048576", ""}, {"", "1048576"}};
    for (const auto &c : cases) ok = run_case(argv[0], c[0], c[1]) && ok;
    if (!ok) return 1;
    std::printf("n2_pipeline_mixed_test passed\n");
    return 0;
}
