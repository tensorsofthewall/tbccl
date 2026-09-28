// Phase 18 timing-scope regression tests (Part H, items 42-44).
//
// The ready/produce timing-boundary logic lives inside
// tensor_transfer_bench.cpp's anonymous namespace (appropriately
// encapsulated -- it is CLI-tool internal, not part of any library
// surface), so it cannot be unit-tested by linking. The CLI itself is
// the testable surface: this file spawns the actual compiled
// tbccl_tensor_transfer_bench binary (its path injected via the
// TBCCL_TENSOR_BENCH_PATH compile definition, a standard CMake/CTest
// "test one executable from another" pattern) as two local loopback
// processes and inspects its observable behavior -- exit codes and
// CSV output -- exactly as a user of the CLI would.
//
// Uses the host backend only, so this test needs neither CUDA nor
// Metal: HostBackend::initialize_source() still does real (if fast)
// CPU work per byte, which is enough to exercise the ready/produce
// distinction meaningfully at a large-enough payload size.
//
// POSIX-only (fork/exec/waitpid) -- consistent with
// tensor_transfer_bench.cpp itself already being POSIX-only
// (<arpa/inet.h>).

#ifndef TBCCL_TENSOR_BENCH_PATH
#error "TBCCL_TENSOR_BENCH_PATH must be defined (see CMakeLists.txt)"
#endif

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <vector>

namespace
{

    void expect(bool condition, const std::string &message)
    {
        if (!condition)
        {
            throw std::runtime_error("assertion failed: " + message);
        }
    }

    struct ChildProcess
    {
        pid_t pid;
        std::string stdout_path;
    };

    // Spawns the benchmark binary with `args`, redirecting its stdout
    // to `stdout_path` (so the CSV can be inspected afterward) and its
    // stderr to /dev/null (diagnostics not needed here). Does not
    // wait -- callers spawn both ranks first, then wait for both, so
    // the two loopback processes actually run concurrently.
    ChildProcess spawn(const std::vector<std::string> &args, const std::string &stdout_path)
    {
        const pid_t pid = fork();

        if (pid < 0)
        {
            throw std::runtime_error("fork failed");
        }

        if (pid == 0)
        {
            const int fd = ::open(
                stdout_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

            if (fd < 0)
            {
                _exit(127);
            }

            ::dup2(fd, STDOUT_FILENO);

            const int devnull = ::open("/dev/null", O_WRONLY);

            if (devnull >= 0)
            {
                ::dup2(devnull, STDERR_FILENO);
            }

            std::vector<char *> argv;
            argv.push_back(const_cast<char *>(TBCCL_TENSOR_BENCH_PATH));

            for (const auto &argument : args)
            {
                argv.push_back(const_cast<char *>(argument.c_str()));
            }

            argv.push_back(nullptr);

            execv(TBCCL_TENSOR_BENCH_PATH, argv.data());

            // execv only returns on failure.
            _exit(127);
        }

        return ChildProcess{pid, stdout_path};
    }

    int wait_for(const ChildProcess &child)
    {
        int status = 0;
        waitpid(child.pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }

    std::string read_file(const std::string &path)
    {
        std::ifstream in(path);
        std::ostringstream contents;
        contents << in.rdbuf();
        return contents.str();
    }

    std::vector<std::string> split_csv_line(const std::string &line)
    {
        std::vector<std::string> fields;
        std::string field;

        for (char c : line)
        {
            if (c == ',')
            {
                fields.push_back(field);
                field.clear();
            }
            else if (c != '\r')
            {
                field.push_back(c);
            }
        }

        fields.push_back(field);
        return fields;
    }

    std::size_t column_index(
        const std::vector<std::string> &header,
        const std::string &name)
    {
        for (std::size_t i = 0; i < header.size(); ++i)
        {
            if (header[i] == name)
            {
                return i;
            }
        }

        throw std::runtime_error("CSV column not found: " + name);
    }

    // Runs a 2-rank loopback end-to-end benchmark (host backend) with
    // the given --timing-scope, and returns the source rank's single
    // data row split into fields, plus the header (for column lookup).
    struct CsvResult
    {
        std::vector<std::string> header;
        std::vector<std::string> row;
    };

    CsvResult run_end_to_end_host(
        const std::string &timing_scope,
        std::uint16_t base_port,
        std::size_t size_bytes,
        int warmup,
        int iterations)
    {
        const std::string stdout0 =
            "/tmp/tbccl_timing_scope_test_" + std::to_string(getpid()) + "_" +
            std::to_string(base_port) + "_r0.csv";
        const std::string stdout1 =
            "/tmp/tbccl_timing_scope_test_" + std::to_string(getpid()) + "_" +
            std::to_string(base_port) + "_r1.csv";

        const std::string peers =
            "127.0.0.1:" + std::to_string(base_port) + ",127.0.0.1:" +
            std::to_string(base_port + 1);

        const std::vector<std::string> common = {
            "--mode", "end-to-end", "--peers", peers, "--local-backend", "host",
            "--source-rank", "0", "--timing-scope", timing_scope, "--sizes",
            std::to_string(size_bytes), "--warmup", std::to_string(warmup),
            "--iterations", std::to_string(iterations)};

        std::vector<std::string> args0 = {"--rank", "0"};
        args0.insert(args0.end(), common.begin(), common.end());

        std::vector<std::string> args1 = {"--rank", "1"};
        args1.insert(args1.end(), common.begin(), common.end());

        const ChildProcess rank0 = spawn(args0, stdout0);
        const ChildProcess rank1 = spawn(args1, stdout1);

        const int exit0 = wait_for(rank0);
        const int exit1 = wait_for(rank1);

        expect(exit0 == 0, "rank 0 (timing_scope=" + timing_scope + ") exited nonzero");
        expect(exit1 == 0, "rank 1 (timing_scope=" + timing_scope + ") exited nonzero");

        const std::string csv_text = read_file(stdout0);
        std::istringstream csv_stream(csv_text);

        std::string header_line;
        std::string row_line;

        expect(
            static_cast<bool>(std::getline(csv_stream, header_line)),
            "missing CSV header for timing_scope=" + timing_scope);
        expect(
            static_cast<bool>(std::getline(csv_stream, row_line)),
            "missing CSV data row for timing_scope=" + timing_scope);

        std::remove(stdout0.c_str());
        std::remove(stdout1.c_str());

        return CsvResult{split_csv_line(header_line), split_csv_line(row_line)};
    }

    // -----------------------------------------------------------------------------
    // Test 1: --timing-scope ready runs successfully and reports its
    // own scope correctly.
    // -----------------------------------------------------------------------------

    void test_ready_scope_runs()
    {
        const CsvResult result =
            run_end_to_end_host("ready", 31950, 1 << 24, 3, 10);

        const std::size_t scope_col = column_index(result.header, "timing_scope");
        const std::size_t completion_col =
            column_index(result.header, "completion_confirmed_us");
        const std::size_t enqueue_col =
            column_index(result.header, "producer_enqueue_us");

        expect(result.row[scope_col] == "ready", "timing_scope column should read 'ready'");
        expect(
            std::stod(result.row[completion_col]) >= 0.0,
            "completion_confirmed_us should be a valid non-negative number");
        expect(
            result.row[enqueue_col] != "NA",
            "producer_enqueue_us should be populated for the host backend");

        std::cout << "[PASS] test_ready_scope_runs\n";
    }

    // -----------------------------------------------------------------------------
    // Test 2: --timing-scope produce runs successfully -- in
    // particular, this implicitly confirms the tool's own per-
    // iteration Part 22 invariant (produce completion >= producer
    // completion) never threw, since a violation would make the
    // source rank exit nonzero.
    // -----------------------------------------------------------------------------

    void test_produce_scope_runs()
    {
        const CsvResult result =
            run_end_to_end_host("produce", 31952, 1 << 24, 3, 10);

        const std::size_t scope_col = column_index(result.header, "timing_scope");
        const std::size_t completion_col =
            column_index(result.header, "completion_confirmed_us");

        expect(result.row[scope_col] == "produce", "timing_scope column should read 'produce'");
        expect(
            std::stod(result.row[completion_col]) >= 0.0,
            "completion_confirmed_us should be a valid non-negative number");

        std::cout << "[PASS] test_produce_scope_runs\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: an invalid --timing-scope is rejected before any
    // networking is attempted (parse_options validates it), so this
    // needs no real peer on the other end -- the process must exit
    // nonzero on its own.
    // -----------------------------------------------------------------------------

    void test_invalid_timing_scope_rejected()
    {
        const std::string stdout_path =
            "/tmp/tbccl_timing_scope_test_" + std::to_string(getpid()) + "_invalid.csv";

        const ChildProcess child = spawn(
            {"--mode", "end-to-end", "--rank", "0", "--peers",
             "127.0.0.1:31960,127.0.0.1:31961", "--local-backend", "host",
             "--timing-scope", "not-a-real-scope", "--sizes", "1024",
             "--warmup", "1", "--iterations", "1"},
            stdout_path);

        const int exit_code = wait_for(child);

        expect(exit_code != 0, "an invalid --timing-scope should be rejected");

        std::remove(stdout_path.c_str());

        std::cout << "[PASS] test_invalid_timing_scope_rejected\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4: repeated alternating ready/produce runs all succeed --
    // stress-tests the Part 22 invariant check under repetition rather
    // than relying on a single sample.
    // -----------------------------------------------------------------------------

    void test_repeated_alternating_scopes()
    {
        constexpr int kRepeats = 10;

        for (int i = 0; i < kRepeats; ++i)
        {
            const std::string scope = (i % 2 == 0) ? "ready" : "produce";
            const std::uint16_t base_port =
                static_cast<std::uint16_t>(31970 + i * 2);

            const CsvResult result =
                run_end_to_end_host(scope, base_port, 1 << 20, 2, 5);

            const std::size_t scope_col = column_index(result.header, "timing_scope");
            expect(
                result.row[scope_col] == scope,
                "iteration " + std::to_string(i) + ": timing_scope mismatch");
        }

        std::cout
            << "[PASS] test_repeated_alternating_scopes (" << kRepeats
            << " iterations)\n";
    }

} // namespace

int main()
{
    try
    {
        test_ready_scope_runs();
        test_produce_scope_runs();
        test_invalid_timing_scope_rejected();
        test_repeated_alternating_scopes();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
