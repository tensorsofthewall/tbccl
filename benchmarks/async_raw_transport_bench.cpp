// Phase 33 diagnostic-only: measures raw Transport::send/recv in a
// loop with the identical ack-based timing scope as
// tbccl_async_transfer_bench, but WITHOUT TensorCommWorker at all --
// isolates whether the queue/thread-handoff mechanism itself costs
// anything beyond what a direct Transport call would cost on its own.
//
// Phase 34 extension: this binary now supports choosing, independently
// per rank, whether the Transport::send()/recv() calls are made from
// this process's main thread (--io-exec=main, the Phase 33 control) or
// from a minimal persistent benchmark-local worker thread
// (--io-exec=worker, RawIoWorker, tensor/raw_io_worker.hpp).
// RawIoWorker deliberately does NOT use
// TensorCommWorker/StagingPool/AsyncMemoryBackend/ChunkPlan/
// TransferWork -- it exists purely to answer "does moving the exact
// same Transport call onto a background std::thread reproduce the
// Phase 33 regression on its own?" (Part F/G of the Phase 34 plan).
//
// Not part of the async substrate's public surface; a standalone
// measurement tool only.

#include <tbccl/peer_capabilities.hpp>
#include <tbccl/tcp.hpp>
#include <tbccl/transport.hpp>

#include "tensor/raw_io_worker.hpp"

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <fstream>
#elif defined(__APPLE__)
#include <pthread.h>
#include <mach/mach.h>
#include <mach/thread_info.h>
#include <sys/qos.h>
#endif

namespace
{
    struct PeerEndpoint { std::string host; std::uint16_t port = 0; };

    PeerEndpoint parse_peer(const std::string &text)
    {
        const auto colon = text.rfind(':');
        return {text.substr(0, colon), static_cast<std::uint16_t>(std::stoul(text.substr(colon + 1)))};
    }

    std::string json_escape(const std::string &s)
    {
        std::string out;
        for (char c : s) { if (c == '"' || c == '\\') out.push_back('\\'); out.push_back(c); }
        return out;
    }

#if defined(__linux__)
    std::string read_file(const std::string &path)
    {
        std::ifstream f(path);
        std::ostringstream ss;
        ss << f.rdbuf();
        return ss.str();
    }

    std::string linux_sched_policy_name(int policy)
    {
        switch (policy)
        {
            case SCHED_OTHER: return "SCHED_OTHER";
            case SCHED_FIFO: return "SCHED_FIFO";
            case SCHED_RR: return "SCHED_RR";
#ifdef SCHED_BATCH
            case SCHED_BATCH: return "SCHED_BATCH";
#endif
#ifdef SCHED_IDLE
            case SCHED_IDLE: return "SCHED_IDLE";
#endif
            default: return "unknown(" + std::to_string(policy) + ")";
        }
    }

    // Diagnostics captured for one Linux thread by TID. Best-effort:
    // any field the running kernel does not expose is recorded as
    // "unavailable" rather than guessed (Phase 34 Part 46).
    struct LinuxThreadDiag
    {
        pid_t tid = 0;
        std::string sched_policy = "unavailable";
        int sched_priority = -1;
        int nice_value = 0;
        bool nice_available = false;
        std::string affinity_list = "unavailable";
        std::string voluntary_ctxt_switches = "unavailable";
        std::string nonvoluntary_ctxt_switches = "unavailable";
        int current_cpu = -1;
    };

    LinuxThreadDiag capture_linux_thread_diag(pid_t tid)
    {
        LinuxThreadDiag d;
        d.tid = tid;

        sched_param sp{};
        int policy = sched_getscheduler(tid);
        if (policy >= 0)
        {
            d.sched_policy = linux_sched_policy_name(policy);
        }
        if (sched_getparam(tid, &sp) == 0)
        {
            d.sched_priority = sp.sched_priority;
        }

        errno = 0;
        const int nice_val = getpriority(PRIO_PROCESS, static_cast<id_t>(tid));
        if (!(nice_val == -1 && errno != 0))
        {
            d.nice_value = nice_val;
            d.nice_available = true;
        }

        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(tid, sizeof(set), &set) == 0)
        {
            std::ostringstream oss;
            bool first = true;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            {
                if (CPU_ISSET(cpu, &set))
                {
                    if (!first) oss << ",";
                    oss << cpu;
                    first = false;
                }
            }
            d.affinity_list = oss.str();
        }

        const std::string status = read_file("/proc/self/task/" + std::to_string(tid) + "/status");
        std::istringstream iss(status);
        std::string line;
        while (std::getline(iss, line))
        {
            auto trim = [](std::string s) {
                while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.erase(s.begin());
                while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r')) s.pop_back();
                return s;
            };
            if (line.rfind("voluntary_ctxt_switches:", 0) == 0)
            {
                d.voluntary_ctxt_switches = trim(line.substr(line.find(':') + 1));
            }
            else if (line.rfind("nonvoluntary_ctxt_switches:", 0) == 0)
            {
                d.nonvoluntary_ctxt_switches = trim(line.substr(line.find(':') + 1));
            }
        }

        // sched_getcpu() reports only the CALLING thread's CPU, not an
        // arbitrary target tid's -- to find where `tid` last ran, parse
        // field 39 (processor) of /proc/self/task/<tid>/stat instead.
        const std::string stat = read_file("/proc/self/task/" + std::to_string(tid) + "/stat");
        const auto close_paren = stat.rfind(')');
        if (close_paren != std::string::npos)
        {
            std::istringstream fields(stat.substr(close_paren + 2));
            std::string field;
            int field_index = 3; // fields after "pid (comm)" start at index 3 (state)
            while (fields >> field)
            {
                if (field_index == 39)
                {
                    try { d.current_cpu = std::stoi(field); } catch (...) {}
                    break;
                }
                ++field_index;
            }
        }
        return d;
    }

    std::string diag_to_json(const LinuxThreadDiag &d, int indent_spaces)
    {
        std::string pad(indent_spaces, ' ');
        std::ostringstream oss;
        oss << "{\n"
            << pad << "  \"tid\": " << d.tid << ",\n"
            << pad << "  \"sched_policy\": \"" << d.sched_policy << "\",\n"
            << pad << "  \"sched_priority\": " << d.sched_priority << ",\n"
            << pad << "  \"nice\": " << (d.nice_available ? std::to_string(d.nice_value) : std::string("null")) << ",\n"
            << pad << "  \"affinity_list\": \"" << d.affinity_list << "\",\n"
            << pad << "  \"voluntary_ctxt_switches\": \"" << d.voluntary_ctxt_switches << "\",\n"
            << pad << "  \"nonvoluntary_ctxt_switches\": \"" << d.nonvoluntary_ctxt_switches << "\",\n"
            << pad << "  \"current_cpu\": " << d.current_cpu << "\n"
            << pad << "}";
        return oss.str();
    }

    // Thread CPU time in milliseconds via CLOCK_THREAD_CPUTIME_ID-style
    // per-thread clock. Returns -1.0 if unavailable for this thread.
    double linux_thread_cpu_time_ms(pthread_t thread)
    {
        clockid_t clockid;
        if (pthread_getcpuclockid(thread, &clockid) != 0) return -1.0;
        timespec ts{};
        if (clock_gettime(clockid, &ts) != 0) return -1.0;
        return static_cast<double>(ts.tv_sec) * 1000.0 + static_cast<double>(ts.tv_nsec) / 1e6;
    }
#endif

#if defined(__APPLE__)
    std::string mac_qos_class_name(qos_class_t qos)
    {
        switch (qos)
        {
            case QOS_CLASS_USER_INTERACTIVE: return "USER_INTERACTIVE";
            case QOS_CLASS_USER_INITIATED: return "USER_INITIATED";
            case QOS_CLASS_DEFAULT: return "DEFAULT";
            case QOS_CLASS_UTILITY: return "UTILITY";
            case QOS_CLASS_BACKGROUND: return "BACKGROUND";
            case QOS_CLASS_UNSPECIFIED: return "UNSPECIFIED";
            default: return "unknown";
        }
    }

    struct MacThreadDiag
    {
        std::string qos_class = "unavailable";
        int relative_priority = 0;
        bool relative_priority_available = false;
        double cpu_time_ms = -1.0; // measured via mach thread_info, -1 = unavailable
    };

    MacThreadDiag capture_mac_thread_diag(pthread_t thread)
    {
        MacThreadDiag d;
        qos_class_t qos = QOS_CLASS_UNSPECIFIED;
        int rel_prio = 0;
        if (pthread_get_qos_class_np(thread, &qos, &rel_prio) == 0)
        {
            d.qos_class = mac_qos_class_name(qos);
            d.relative_priority = rel_prio;
            d.relative_priority_available = true;
        }

        mach_port_t mach_thread = pthread_mach_thread_np(thread);
        thread_basic_info_data_t info{};
        mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
        if (thread_info(mach_thread, THREAD_BASIC_INFO,
                         reinterpret_cast<thread_info_t>(&info), &count) == KERN_SUCCESS)
        {
            const double user_ms = info.user_time.seconds * 1000.0 + info.user_time.microseconds / 1000.0;
            const double sys_ms = info.system_time.seconds * 1000.0 + info.system_time.microseconds / 1000.0;
            d.cpu_time_ms = user_ms + sys_ms;
        }
        return d;
    }

    std::string diag_to_json(const MacThreadDiag &d, int indent_spaces)
    {
        std::string pad(indent_spaces, ' ');
        std::ostringstream oss;
        oss << "{\n"
            << pad << "  \"qos_class\": \"" << d.qos_class << "\",\n"
            << pad << "  \"relative_priority\": "
            << (d.relative_priority_available ? std::to_string(d.relative_priority) : std::string("null")) << ",\n"
            << pad << "  \"cpu_time_ms\": " << (d.cpu_time_ms >= 0 ? std::to_string(d.cpu_time_ms) : std::string("null")) << "\n"
            << pad << "}";
        return oss.str();
    }
#endif
}

int main(int argc, char **argv)
{
    try
    {
        std::size_t rank = 0, bytes = 64 * 1024 * 1024, warmup = 5, iterations = 20, source_rank = 0;
        std::vector<PeerEndpoint> peers;
        std::string io_exec = "main"; // main | worker
        long worker_cpu = -1;          // Linux only; -1 = unpinned
        std::string worker_qos;        // mac only: "" (default) | "inherit-main"
        bool sample_cpu = false;       // Linux only, opt-in per-iteration sched_getcpu sampling
        bool print_thread_info = false;

        for (int i = 1; i < argc; ++i)
        {
            std::string arg = argv[i];
            auto next = [&]() { return std::string(argv[++i]); };
            if (arg == "--rank") rank = std::stoul(next());
            else if (arg == "--peers") {
                std::string v = next(); auto comma = v.find(',');
                peers = {parse_peer(v.substr(0, comma)), parse_peer(v.substr(comma + 1))};
            }
            else if (arg == "--bytes") bytes = std::stoull(next());
            else if (arg == "--warmup") warmup = std::stoull(next());
            else if (arg == "--iterations") iterations = std::stoull(next());
            else if (arg == "--source-rank") source_rank = std::stoul(next());
            else if (arg == "--io-exec") io_exec = next();
            else if (arg == "--worker-cpu") worker_cpu = std::stol(next());
            else if (arg == "--worker-qos") worker_qos = next();
            else if (arg == "--sample-cpu") sample_cpu = true;
            else if (arg == "--thread-info") print_thread_info = true;
        }
        if (io_exec != "main" && io_exec != "worker")
        {
            throw std::runtime_error("--io-exec must be 'main' or 'worker'");
        }
        const bool is_sender = (rank == source_rank);
        const bool use_worker = (io_exec == "worker");

        std::unique_ptr<tbccl::Connection> connection;
        if (rank == 0)
        {
            auto listener = tbccl::tcp_listen(peers[0].host, peers[0].port, {});
            // Part AO: machine-readable readiness marker printed right
            // after listen()/bind() succeeds (well, right after
            // tcp_listen returns -- the accept() below still blocks),
            // so an orchestrating script can gate the peer's connect()
            // on genuine listener readiness instead of a fixed sleep.
            std::fprintf(stderr, "READY\n");
            std::fflush(stderr);
            connection = listener->accept();
        }
        else
        {
            connection = tbccl::tcp_connect(peers[0].host, peers[0].port, {});
        }
        const auto local_caps = tbccl::local_capabilities();
        tbccl::exchange_capabilities(*connection, local_caps);
        tbccl::TcpTransport transport(std::move(connection));

        std::vector<std::uint8_t> buffer(bytes, is_sender ? 0x42 : 0);

        std::unique_ptr<tbccl_bench::RawIoWorker> worker;
        if (use_worker)
        {
#if defined(__APPLE__)
            std::optional<qos_class_t> requested_qos;
            if (worker_qos == "inherit-main")
            {
                qos_class_t main_qos = QOS_CLASS_UNSPECIFIED;
                int main_rel = 0;
                pthread_get_qos_class_np(pthread_self(), &main_qos, &main_rel);
                requested_qos = main_qos;
            }
            else if (!worker_qos.empty() && worker_qos != "default")
            {
                throw std::runtime_error("--worker-qos must be 'default' or 'inherit-main'");
            }
            worker = std::make_unique<tbccl_bench::RawIoWorker>(&transport, requested_qos);
#else
            worker = std::make_unique<tbccl_bench::RawIoWorker>(&transport);
#endif
#if defined(__linux__)
            if (worker_cpu >= 0)
            {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(static_cast<int>(worker_cpu), &set);
                pthread_setaffinity_np(worker->native_handle(), sizeof(set), &set);
            }
#else
            (void)worker_cpu;
#endif
#if !defined(__APPLE__)
            (void)worker_qos;
#endif
        }

        auto do_transport_op = [&](bool send, void *data, std::size_t n)
        {
            if (worker)
            {
                worker->run_op(send ? tbccl_bench::RawIoWorker::Op::Send : tbccl_bench::RawIoWorker::Op::Recv, data, n);
            }
            else
            {
                if (send) transport.send(data, n);
                else transport.recv(data, n);
            }
        };

        std::vector<double> completion_us;
        std::vector<int> start_cpus, end_cpus; // sample-cpu diagnostic, main-thread only meaningful when io_exec=main
        const std::size_t total = warmup + iterations;
        for (std::size_t round = 0; round < total; ++round)
        {
            const auto start = std::chrono::steady_clock::now();
#if defined(__linux__)
            int cpu_before = sample_cpu ? sched_getcpu() : -1;
#endif
            do_transport_op(is_sender, buffer.data(), buffer.size());
            std::uint8_t ack = 0;
            if (is_sender) do_transport_op(false, &ack, sizeof(ack));
            else { ack = 1; do_transport_op(true, &ack, sizeof(ack)); }
            const auto end = std::chrono::steady_clock::now();
#if defined(__linux__)
            if (sample_cpu && round >= warmup)
            {
                start_cpus.push_back(cpu_before);
                end_cpus.push_back(sched_getcpu());
            }
#endif
            if (round >= warmup)
                completion_us.push_back(std::chrono::duration<double, std::micro>(end - start).count());
        }

        // Thread diagnostics, captured after the measured loop so the
        // worker thread has definitely executed real Transport calls.
        std::ostringstream diag_json;
        diag_json << "  \"io_exec\": \"" << io_exec << "\",\n";
#if defined(__linux__)
        diag_json << "  \"worker_cpu_requested\": " << worker_cpu << ",\n";
        const pid_t main_tid = static_cast<pid_t>(::syscall(SYS_gettid));
        const auto main_diag = capture_linux_thread_diag(main_tid);
        diag_json << "  \"main_thread\": " << diag_to_json(main_diag, 2) << ",\n";
        double main_cpu_ms = linux_thread_cpu_time_ms(pthread_self());
        diag_json << "  \"main_thread_cpu_ms\": " << main_cpu_ms << ",\n";
        if (worker)
        {
            const auto wdiag = capture_linux_thread_diag(worker->linux_tid());
            diag_json << "  \"worker_thread\": " << diag_to_json(wdiag, 2) << ",\n";
            double worker_cpu_ms = linux_thread_cpu_time_ms(worker->native_handle());
            diag_json << "  \"worker_thread_cpu_ms\": " << worker_cpu_ms << ",\n";
        }
        if (sample_cpu && !start_cpus.empty())
        {
            int migrations = 0;
            for (std::size_t i = 0; i < start_cpus.size(); ++i)
                if (start_cpus[i] != end_cpus[i]) ++migrations;
            diag_json << "  \"cpu_sample_migrations\": " << migrations << ",\n";
            diag_json << "  \"cpu_sample_start_cpus\": [";
            for (std::size_t i = 0; i < start_cpus.size(); ++i) { if (i) diag_json << ","; diag_json << start_cpus[i]; }
            diag_json << "],\n";
            diag_json << "  \"cpu_sample_end_cpus\": [";
            for (std::size_t i = 0; i < end_cpus.size(); ++i) { if (i) diag_json << ","; diag_json << end_cpus[i]; }
            diag_json << "],\n";
        }
#elif defined(__APPLE__)
        diag_json << "  \"worker_qos_requested\": \"" << json_escape(worker_qos) << "\",\n";
        const auto main_diag = capture_mac_thread_diag(pthread_self());
        diag_json << "  \"main_thread\": " << diag_to_json(main_diag, 2) << ",\n";
        if (worker)
        {
            const auto wdiag = capture_mac_thread_diag(worker->pthread_handle());
            diag_json << "  \"worker_thread\": " << diag_to_json(wdiag, 2) << ",\n";
        }
#endif
        (void)print_thread_info;

        if (is_sender)
        {
            std::sort(completion_us.begin(), completion_us.end());
            double median = completion_us[completion_us.size() / 2];
            double gib_s = (bytes / (1024.0*1024.0*1024.0)) / (median / 1e6);
            std::cout << "{\n"
                       << "  \"raw_transport_median_us\": " << median << ",\n"
                       << "  \"raw_transport_gib_s\": " << gib_s << ",\n"
                       << diag_json.str()
                       << "  \"role\": \"sender\"\n"
                       << "}\n";
        }
        else
        {
            std::cout << "{\n"
                       << diag_json.str()
                       << "  \"role\": \"receiver\"\n"
                       << "}\n";
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
