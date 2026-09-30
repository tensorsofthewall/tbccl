#pragma once

// Benchmark-only, opt-in per-iteration tracing. Events are buffered in memory
// and emitted once after the measured batch; no hot-path file I/O occurs.
#include <time.h>

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tbccl_bench::trace
{

inline std::int64_t monotonic_ns()
{
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC, &value) != 0)
        throw std::runtime_error("clock_gettime(CLOCK_MONOTONIC) failed");
    return static_cast<std::int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

inline std::int64_t realtime_ns()
{
    timespec value{};
    if (::clock_gettime(CLOCK_REALTIME, &value) != 0)
        throw std::runtime_error("clock_gettime(CLOCK_REALTIME) failed");
    return static_cast<std::int64_t>(value.tv_sec) * 1000000000LL + value.tv_nsec;
}

inline std::string boot_id()
{
#ifdef __linux__
    std::ifstream input("/proc/sys/kernel/random/boot_id");
    std::string value;
    if (input && std::getline(input, value)) return value;
#endif
    return "unavailable";
}

inline std::string escape_json(const std::string &value)
{
    std::ostringstream output;
    for (const unsigned char c : value)
    {
        switch (c)
        {
        case '\\': output << "\\\\"; break;
        case '"': output << "\\\""; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (c < 0x20)
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(c) << std::dec;
            else
                output << c;
        }
    }
    return output.str();
}

struct Entry
{
    int iteration_index = -1;
    std::int64_t iteration_begin_ns = -1;
    std::int64_t source_ready_ns = -1;
    std::int64_t staging_begin_ns = -1;
    std::int64_t staging_end_ns = -1;
    std::int64_t send_begin_ns = -1;
    std::int64_t send_end_ns = -1;
    std::int64_t ack_wait_begin_ns = -1;
    std::int64_t ack_received_ns = -1;
    std::int64_t recv_begin_ns = -1;
    std::int64_t recv_end_ns = -1;
    std::int64_t destination_staging_begin_ns = -1;
    std::int64_t destination_staging_end_ns = -1;
    std::int64_t destination_sync_end_ns = -1;
    std::int64_t ack_send_begin_ns = -1;
    std::int64_t ack_send_end_ns = -1;
    std::int64_t iteration_end_ns = -1;
    std::int64_t observed_source_gap_ns = -1;
};

class Batch
{
public:
    Batch(bool enabled, std::size_t capacity, std::string run_id,
          std::string physical_machine, std::size_t local_rank,
          std::size_t source_rank, std::string direction,
          std::size_t payload_bytes, int busy_poll_us,
          std::string timing_scope, int source_gap_us)
        : enabled_(enabled), capacity_(capacity), run_id_(std::move(run_id)),
          physical_machine_(std::move(physical_machine)), local_rank_(local_rank),
          source_rank_(source_rank), direction_(std::move(direction)),
          payload_bytes_(payload_bytes), busy_poll_us_(busy_poll_us),
          timing_scope_(std::move(timing_scope)), source_gap_us_(source_gap_us)
    {
        if (!enabled_) return;
        entries_.reserve(capacity_);
        boot_id_ = boot_id();
        clock_sample_monotonic_ns_ = monotonic_ns();
        clock_sample_realtime_ns_ = realtime_ns();
        timespec resolution{};
        if (::clock_getres(CLOCK_MONOTONIC, &resolution) != 0)
            throw std::runtime_error("clock_getres(CLOCK_MONOTONIC) failed");
        clock_resolution_ns_ = static_cast<std::int64_t>(resolution.tv_sec) *
                                   1000000000LL +
                               resolution.tv_nsec;
    }

    bool enabled() const { return enabled_; }
    std::int64_t now() const { return enabled_ ? monotonic_ns() : -1; }

    void add(const Entry &entry)
    {
        if (!enabled_) return;
        if (entries_.size() >= capacity_)
            throw std::runtime_error("transfer trace capacity exceeded");
        entries_.push_back(entry);
    }

    void flush() const
    {
        if (!enabled_) return;
        if (entries_.size() != capacity_)
            throw std::runtime_error("transfer trace batch is incomplete");
        std::ostringstream output;
        output << "TBCCL_DIAGNOSTIC {\"kind\":\"transfer_trace\""
               << ",\"clock\":\"CLOCK_MONOTONIC\""
               << ",\"clock_units\":\"nanoseconds\""
               << ",\"clock_resolution_ns\":" << clock_resolution_ns_
               << ",\"clock_sample_monotonic_ns\":" << clock_sample_monotonic_ns_
               << ",\"clock_sample_realtime_ns\":" << clock_sample_realtime_ns_
               << ",\"boot_id\":\"" << escape_json(boot_id_) << "\""
               << ",\"capacity\":" << capacity_ << ",\"entries\":[";
        for (std::size_t index = 0; index < entries_.size(); ++index)
        {
            if (index) output << ',';
            write_entry(output, entries_[index]);
        }
        output << "]}\n";
        std::cerr << output.str();
    }

private:
    static void write_optional(std::ostringstream &output, std::int64_t value)
    {
        if (value < 0) output << "null";
        else output << value;
    }

    void write_entry(std::ostringstream &output, const Entry &entry) const
    {
        output << "{\"run_id\":\"" << escape_json(run_id_)
               << "\",\"iteration_index\":" << entry.iteration_index
               << ",\"local_rank\":" << local_rank_
               << ",\"physical_machine\":\"" << escape_json(physical_machine_)
               << "\",\"source_rank\":" << source_rank_
               << ",\"direction\":\"" << escape_json(direction_)
               << "\",\"payload_bytes\":" << payload_bytes_
               << ",\"busy_poll_us\":" << busy_poll_us_
               << ",\"timing_scope\":\"" << escape_json(timing_scope_)
               << "\",\"source_gap_us\":" << source_gap_us_;
#define TBCCL_TRACE_FIELD(name) \
        output << ",\"" #name "\":"; write_optional(output, entry.name)
        TBCCL_TRACE_FIELD(iteration_begin_ns);
        TBCCL_TRACE_FIELD(source_ready_ns);
        TBCCL_TRACE_FIELD(staging_begin_ns);
        TBCCL_TRACE_FIELD(staging_end_ns);
        TBCCL_TRACE_FIELD(send_begin_ns);
        TBCCL_TRACE_FIELD(send_end_ns);
        TBCCL_TRACE_FIELD(ack_wait_begin_ns);
        TBCCL_TRACE_FIELD(ack_received_ns);
        TBCCL_TRACE_FIELD(recv_begin_ns);
        TBCCL_TRACE_FIELD(recv_end_ns);
        TBCCL_TRACE_FIELD(destination_staging_begin_ns);
        TBCCL_TRACE_FIELD(destination_staging_end_ns);
        TBCCL_TRACE_FIELD(destination_sync_end_ns);
        TBCCL_TRACE_FIELD(ack_send_begin_ns);
        TBCCL_TRACE_FIELD(ack_send_end_ns);
        TBCCL_TRACE_FIELD(iteration_end_ns);
        TBCCL_TRACE_FIELD(observed_source_gap_ns);
#undef TBCCL_TRACE_FIELD
        output << '}';
    }

    bool enabled_ = false;
    std::size_t capacity_ = 0;
    std::string run_id_;
    std::string physical_machine_;
    std::size_t local_rank_ = 0;
    std::size_t source_rank_ = 0;
    std::string direction_;
    std::size_t payload_bytes_ = 0;
    int busy_poll_us_ = 0;
    std::string timing_scope_;
    int source_gap_us_ = 0;
    std::string boot_id_;
    std::int64_t clock_sample_monotonic_ns_ = -1;
    std::int64_t clock_sample_realtime_ns_ = -1;
    std::int64_t clock_resolution_ns_ = -1;
    std::vector<Entry> entries_;
};

} // namespace tbccl_bench::trace
