#pragma once

// Benchmark-only, opt-in telemetry. No transport API or wire-format changes.
#include <tbccl/tcp.hpp>
#include <sys/resource.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <time.h>
#ifdef __linux__
#include <dirent.h>
#endif
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace tbccl_bench::diagnostics
{
inline double seconds(const timeval &value)
{
    return static_cast<double>(value.tv_sec) + static_cast<double>(value.tv_usec) / 1e6;
}

struct CpuSnapshot
{
    rusage usage{};
    std::chrono::steady_clock::time_point wall;
    double thread_seconds = -1.0;

    static CpuSnapshot capture()
    {
        CpuSnapshot result;
        if (::getrusage(RUSAGE_SELF, &result.usage) != 0)
            throw std::runtime_error("getrusage failed");
#ifdef CLOCK_THREAD_CPUTIME_ID
        timespec thread{};
        if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &thread) == 0)
            result.thread_seconds = static_cast<double>(thread.tv_sec) +
                                    static_cast<double>(thread.tv_nsec) / 1e9;
#endif
        result.wall = std::chrono::steady_clock::now();
        return result;
    }
};

class CpuWindow
{
public:
    explicit CpuWindow(bool enabled) : enabled_(enabled) {}
    void begin() { if (enabled_) start_ = CpuSnapshot::capture(); }
    void end() { if (enabled_) stop_ = CpuSnapshot::capture(); }
    void report(std::size_t rank, std::size_t size) const
    {
        if (!enabled_) return;
        const double wall = std::chrono::duration<double>(stop_.wall - start_.wall).count();
        const double user = seconds(stop_.usage.ru_utime) - seconds(start_.usage.ru_utime);
        const double system = seconds(stop_.usage.ru_stime) - seconds(start_.usage.ru_stime);
        std::ostringstream out;
        out << std::setprecision(12)
            << "TBCCL_DIAGNOSTIC {\"kind\":\"cpu\",\"rank\":" << rank
            << ",\"payload_bytes\":" << size
            << ",\"wall_seconds\":" << wall << ",\"user_seconds\":" << user
            << ",\"system_seconds\":" << system
            << ",\"process_cpu_pct\":" << (wall > 0 ? 100 * (user + system) / wall : 0)
            << ",\"thread_seconds\":";
        if (start_.thread_seconds >= 0 && stop_.thread_seconds >= 0)
            out << stop_.thread_seconds - start_.thread_seconds;
        else out << "null";
        out << ",\"voluntary_context_switches\":" << stop_.usage.ru_nvcsw - start_.usage.ru_nvcsw
            << ",\"involuntary_context_switches\":" << stop_.usage.ru_nivcsw - start_.usage.ru_nivcsw
            << ",\"cpu_migrations\":null}\n";
        std::cerr << out.str();
    }
private:
    bool enabled_;
    CpuSnapshot start_, stop_;
};

inline void samples(bool enabled, std::size_t rank, std::size_t size,
                    const char *metric, const std::vector<double> &values)
{
    if (!enabled) return;
    std::ostringstream out;
    out << std::setprecision(12) << "TBCCL_DIAGNOSTIC {\"kind\":\"samples\",\"rank\":"
        << rank << ",\"payload_bytes\":" << size << ",\"metric\":\"" << metric
        << "\",\"values_us\":[";
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i) out << ',';
        out << values[i];
    }
    out << "]}\n";
    std::cerr << out.str();
}

// Inspect this benchmark's connected IPv4 TCP descriptors after bootstrap.
// World does not expose descriptors; /proc/self/fd avoids changing that API.
// An observed value is kernel configuration, NOT proof of active NIC polling.
inline void sockets(int requested)
{
    std::ostringstream out;
    out << "TBCCL_DIAGNOSTIC {\"kind\":\"busy_poll\",\"requested_us\":" << requested;
#if defined(__linux__) && defined(SO_BUSY_POLL)
    out << ",\"status\":\"supported\",\"sockets\":[";
    DIR *directory = ::opendir("/proc/self/fd");
    if (!directory) throw std::runtime_error("cannot inspect /proc/self/fd");
    bool first = true;
    while (const auto *entry = ::readdir(directory))
    {
        char *end = nullptr;
        const long number = std::strtol(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0') continue;
        const int fd = static_cast<int>(number);
        sockaddr_in peer{}, local{};
        socklen_t length = sizeof(peer);
        if (::getpeername(fd, reinterpret_cast<sockaddr *>(&peer), &length) != 0 ||
            peer.sin_family != AF_INET) continue;
        int type = 0;
        length = sizeof(type);
        if (::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &length) != 0 || type != SOCK_STREAM)
            continue;
        length = sizeof(local);
        if (::getsockname(fd, reinterpret_cast<sockaddr *>(&local), &length) != 0) continue;
        int effective = 0;
        length = sizeof(effective);
        const bool observed = ::getsockopt(fd, SOL_SOCKET, SO_BUSY_POLL, &effective, &length) == 0;
        if (!first) out << ',';
        first = false;
        out << "{\"fd\":" << fd << ",\"local_port\":" << ntohs(local.sin_port)
            << ",\"peer_port\":" << ntohs(peer.sin_port) << ",\"effective_us\":";
        if (observed) out << effective;
        else out << "null";
        out << '}';
    }
    ::closedir(directory);
    out << ']';
#else
    out << ",\"status\":\"unsupported\",\"sockets\":[]";
#endif
    out << "}\n";
    std::cerr << out.str();
}
} // namespace tbccl_bench::diagnostics
