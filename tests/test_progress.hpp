// Progress markers and a watchdog for tests that run several communicators in one process, so that a test killed by its ctest timeout leaves evidence of where it stopped.
//
//   test_progress::start("NAME")      prints "NAME begin" and arms the watchdog
//   test_progress::mark("text")       prints "NAME <text> t=<seconds since start>s" to stderr (flushed) and remembers it as the last marker
//   test_progress::finish()           prints a summary (elapsed time, CPU, context switches, peak RSS, threads) and disarms the watchdog
//
// The watchdog fires after TBCCL_TEST_WATCHDOG_S seconds (default 100, i.e. shortly before the 120 s ctest timeout of the tests that use this header): it prints the last marker, and on
// Linux the state, kernel wait channel and a backtrace of every thread and the loopback TCP connections with queued bytes, then ends the process with status 124. It never
// turns a hang into a pass and leaves no process behind. It needs no privileges. Times are this process's own steady clock; nothing here is compared across processes.
// Backtraces carry function names when the executable is linked with -rdynamic; otherwise they are offsets into the executable.

#pragma once

#include <sys/resource.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

#ifdef __linux__
#include <dirent.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <fstream>
#include <sstream>
#endif

namespace test_progress
{
namespace detail
{
using Clock = std::chrono::steady_clock;

struct State
{
    std::string name;
    Clock::time_point begin;
    std::mutex mutex;
    std::string last;
    std::condition_variable cv;
    bool finished = false;
    std::thread watchdog;
};

inline State &state()
{
    static State s;
    return s;
}

inline double elapsed()
{
    return std::chrono::duration<double>(Clock::now() - state().begin).count();
}

#ifdef __linux__
inline std::atomic<int> g_bt_done{0};

inline void backtrace_handler(int)
{
    void *frames[48];
    const int n = ::backtrace(frames, 48);
    ::backtrace_symbols_fd(frames, n, 2);
    g_bt_done.store(1, std::memory_order_release);
}

inline std::string read_line(const std::string &path)
{
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

inline unsigned port_of(const std::string &addr_port)
{
    const auto colon = addr_port.find(':');
    return colon == std::string::npos ? 0u : static_cast<unsigned>(std::strtoul(addr_port.c_str() + colon + 1, nullptr, 16));
}

inline void dump_threads()
{
    std::fprintf(stderr, "WATCHDOG threads (tid state comm wchan):\n");
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_handler = backtrace_handler;
    sigemptyset(&sa.sa_mask);
    ::sigaction(SIGUSR2, &sa, nullptr);
    DIR *d = ::opendir("/proc/self/task");
    if (!d) return;
    const pid_t pid = ::getpid();
    const pid_t self = static_cast<pid_t>(::syscall(SYS_gettid));
    int shown = 0;
    while (dirent *e = ::readdir(d))
    {
        if (e->d_name[0] < '0' || e->d_name[0] > '9' || ++shown > 64) continue;
        const std::string dir = std::string("/proc/self/task/") + e->d_name + "/";
        const std::string stat = read_line(dir + "stat");
        const auto rp = stat.rfind(')');
        const std::string comm = read_line(dir + "comm");
        std::fprintf(stderr, "  tid=%s state=%c comm=%s wchan=%s\n", e->d_name, rp != std::string::npos && rp + 2 < stat.size() ? stat[rp + 2] : '?', comm.c_str(), read_line(dir + "wchan").c_str());
        const pid_t tid = static_cast<pid_t>(std::atoi(e->d_name));
        if (tid == self) continue;
        g_bt_done.store(0, std::memory_order_release);
        std::fprintf(stderr, "  backtrace tid=%d:\n", static_cast<int>(tid));
        std::fflush(stderr);
        if (::syscall(SYS_tgkill, pid, tid, SIGUSR2) == 0)
            for (int i = 0; i < 40 && !g_bt_done.load(std::memory_order_acquire); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ::closedir(d);
}

inline void dump_sockets()
{
    std::fprintf(stderr, "WATCHDOG loopback TCP sockets of this process's namespace with queued bytes (state tx_queue:rx_queue local_port remote_port):\n");
    int shown = 0;
    for (const char *file : {"/proc/self/net/tcp", "/proc/self/net/tcp6"})
    {
        std::ifstream f(file);
        std::string line;
        std::getline(f, line);
        while (std::getline(f, line) && shown < 40)
        {
            std::istringstream is(line);
            std::string sl, local, remote, st, q;
            is >> sl >> local >> remote >> st >> q;
            const auto colon = q.find(':');
            if (colon == std::string::npos) continue;
            const unsigned long tx = std::strtoul(q.c_str(), nullptr, 16), rx = std::strtoul(q.c_str() + colon + 1, nullptr, 16);
            const bool loopback = local.rfind("0100007F", 0) == 0 || local.rfind("00000000000000000000000001000000", 0) == 0;
            if (loopback && (tx != 0 || rx != 0))
            {
                std::fprintf(stderr, "  %s %lu:%lu %u %u\n", st.c_str(), tx, rx, port_of(local), port_of(remote));
                ++shown;
            }
        }
    }
}
#endif

inline void summary(const char *label)
{
    rusage ru{};
    ::getrusage(RUSAGE_SELF, &ru);
    long threads = -1;
#ifdef __linux__
    const std::string status = "/proc/self/status";
    std::ifstream f(status);
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("Threads:", 0) == 0) threads = std::atol(line.c_str() + 8);
#endif
    std::fprintf(stderr, "%s %s t=%.3fs utime=%.3fs stime=%.3fs maxrss_kb=%ld nvcsw=%ld nivcsw=%ld threads=%ld\n", state().name.c_str(), label, elapsed(),
                 static_cast<double>(ru.ru_utime.tv_sec) + ru.ru_utime.tv_usec * 1e-6, static_cast<double>(ru.ru_stime.tv_sec) + ru.ru_stime.tv_usec * 1e-6, ru.ru_maxrss, ru.ru_nvcsw,
                 ru.ru_nivcsw, threads);
    std::fflush(stderr);
}

inline void watchdog_main(int seconds)
{
    State &s = state();
    std::unique_lock<std::mutex> lock(s.mutex);
    if (s.cv.wait_for(lock, std::chrono::seconds(seconds), [&] { return s.finished; })) return;
    const std::string last = s.last;
    lock.unlock();
    std::fprintf(stderr, "WATCHDOG: no completion after %d s; last marker: %s\n", seconds, last.c_str());
    summary("WATCHDOG");
#ifdef __linux__
    dump_threads();
    dump_sockets();
#endif
    std::fprintf(stderr, "WATCHDOG: ending the process with status 124\n");
    std::fflush(stderr);
    std::_Exit(124);
}
} // namespace detail

inline void mark(const std::string &text)
{
    detail::State &s = detail::state();
    char head[64];
    std::snprintf(head, sizeof head, " t=%.3fs", detail::elapsed());
    const std::string line = s.name + " " + text + head;
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.last = line;
    }
    std::fprintf(stderr, "%s\n", line.c_str());
    std::fflush(stderr);
}

inline void start(const std::string &name)
{
    detail::State &s = detail::state();
    s.name = name;
    s.begin = detail::Clock::now();
    const char *e = std::getenv("TBCCL_TEST_WATCHDOG_S");
    const int seconds = e && *e ? std::atoi(e) : 100;
    mark("begin");
    if (seconds > 0) s.watchdog = std::thread(detail::watchdog_main, seconds);
}

inline void finish()
{
    detail::State &s = detail::state();
    {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.finished = true;
    }
    s.cv.notify_all();
    if (s.watchdog.joinable()) s.watchdog.join();
    detail::summary("complete");
}
} // namespace test_progress
