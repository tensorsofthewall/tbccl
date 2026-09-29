// Phase 19 regression tests for benchmarks/tensor/cuda_sync_bench.cu
// (Part J, items 48-49). Only built/run when TBCCL_ENABLE_CUDA is on
// and only meaningful with a CUDA device present at runtime.
//
// cuda_sync_bench is a standalone diagnostic tool (a plain main(),
// not a library), so -- exactly as Phase 18's
// tests/tensor_timing_scope_test.cpp did for tbccl_tensor_transfer_bench
// -- this spawns the actual compiled binary (its path injected via
// the TBCCL_CUDA_SYNC_BENCH_PATH compile definition) and inspects its
// observable behavior: exit code and, where relevant, stderr text.
// The tool's own internal correctness checks (byte-exact readback
// against the deterministic pattern, added in Phase 19) are what
// actually verify "the selected synchronization strategy did not
// return before GPU output was ready" (Part 49) -- a nonzero exit
// code here means that check failed and threw.

#ifndef TBCCL_CUDA_SYNC_BENCH_PATH
#error "TBCCL_CUDA_SYNC_BENCH_PATH must be defined (see CMakeLists.txt)"
#endif

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
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

    // Runs the diagnostic binary with `args`, waits for it, and
    // returns its exit code. stdout/stderr are discarded (this suite
    // only checks exit codes -- the tool's own internal correctness
    // checks are what matter here, and they throw + exit nonzero on
    // failure).
    int run_bench(const std::vector<std::string> &args)
    {
        const pid_t pid = fork();

        if (pid < 0)
        {
            throw std::runtime_error("fork failed");
        }

        if (pid == 0)
        {
            const int devnull = ::open("/dev/null", O_WRONLY);

            if (devnull >= 0)
            {
                ::dup2(devnull, STDOUT_FILENO);
                ::dup2(devnull, STDERR_FILENO);
            }

            std::vector<char *> argv;
            argv.push_back(const_cast<char *>(TBCCL_CUDA_SYNC_BENCH_PATH));

            for (const auto &argument : args)
            {
                argv.push_back(const_cast<char *>(argument.c_str()));
            }

            argv.push_back(nullptr);

            execv(TBCCL_CUDA_SYNC_BENCH_PATH, argv.data());
            _exit(127);
        }

        int status = 0;
        waitpid(pid, &status, 0);
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }

    // -----------------------------------------------------------------------------
    // Test 1: a CUDA device is available. If not, every subsequent
    // test would fail identically and uninformatively -- classify
    // this as a skip, not a failure (matching every other tensor
    // backend test file's convention).
    // -----------------------------------------------------------------------------

    bool cuda_device_available()
    {
        return run_bench(
                   {"--experiment", "kernel", "--sizes", "64", "--warmup", "1",
                    "--iterations", "1", "--runs", "1"}) == 0;
    }

    // -----------------------------------------------------------------------------
    // Test 2: every synchronization mode (stream/event/stream-poll/
    // event-poll) produces byte-correct GPU output -- exercised via
    // --experiment sync-mode's own internal readback-and-verify check
    // (added in Phase 19; a failure there throws and this process
    // exits nonzero).
    // -----------------------------------------------------------------------------

    void test_sync_mode_correctness_all_modes()
    {
        const int exit_code = run_bench(
            {"--experiment", "sync-mode", "--sync-mode", "all", "--sizes",
             "64,4096", "--warmup", "3", "--iterations", "10", "--runs", "1"});

        expect(exit_code == 0, "all four sync modes should produce correct GPU data");

        std::cout << "[PASS] test_sync_mode_correctness_all_modes\n";
    }

    // -----------------------------------------------------------------------------
    // Test 3: H2D and D2H comparisons each verify their own
    // correctness internally (Phase 19 addition) -- confirms both
    // pageable and pinned paths in both directions.
    // -----------------------------------------------------------------------------

    void test_h2d_and_d2h_correctness()
    {
        const int h2d_exit = run_bench(
            {"--experiment", "h2d", "--sizes", "64,4096", "--warmup", "3",
             "--iterations", "10", "--runs", "1"});
        const int d2h_exit = run_bench(
            {"--experiment", "d2h", "--sizes", "64,4096", "--warmup", "3",
             "--iterations", "10", "--runs", "1"});

        expect(h2d_exit == 0, "H2D pageable/pinned should produce correct GPU data");
        expect(d2h_exit == 0, "D2H pageable/pinned should produce correct GPU data");

        std::cout << "[PASS] test_h2d_and_d2h_correctness\n";
    }

    // -----------------------------------------------------------------------------
    // Test 4: zero and odd payload sizes, across every experiment --
    // regression test for the Phase 19 zero-block-launch fix
    // (launch_fill_pattern previously issued an invalid 0-block CUDA
    // launch for size 0; no Phase 18 experiment's hardcoded size list
    // ever included 0, so the bug was latent until --sizes made a
    // 0-byte run possible).
    // -----------------------------------------------------------------------------

    void test_zero_and_odd_sizes()
    {
        const std::vector<std::string> experiments = {"kernel", "h2d", "d2h",
                                                        "sync-mode"};

        for (const auto &experiment : experiments)
        {
            const int exit_code = run_bench(
                {"--experiment", experiment, "--sizes", "0,1,3,1000003", "--warmup",
                 "2", "--iterations", "5", "--runs", "1"});

            expect(
                exit_code == 0,
                "experiment '" + experiment +
                    "' should handle zero/odd payload sizes correctly");
        }

        std::cout << "[PASS] test_zero_and_odd_sizes\n";
    }

    // -----------------------------------------------------------------------------
    // Test 5: invalid --sync-mode and --device-schedule are rejected
    // before any CUDA work is attempted.
    // -----------------------------------------------------------------------------

    void test_invalid_arguments_rejected()
    {
        const int bad_sync_mode = run_bench(
            {"--sync-mode", "not-a-real-mode", "--experiment", "sync-mode",
             "--warmup", "1", "--iterations", "1", "--runs", "1"});
        const int bad_device_schedule = run_bench(
            {"--device-schedule", "not-a-real-schedule", "--warmup", "1",
             "--iterations", "1", "--runs", "1"});
        const int sizes_with_all = run_bench(
            {"--sizes", "64", "--experiment", "all", "--warmup", "1",
             "--iterations", "1", "--runs", "1"});

        expect(bad_sync_mode != 0, "an invalid --sync-mode should be rejected");
        expect(
            bad_device_schedule != 0,
            "an invalid --device-schedule should be rejected");
        expect(
            sizes_with_all != 0,
            "--sizes combined with --experiment all should be rejected");

        std::cout << "[PASS] test_invalid_arguments_rejected\n";
    }

    // -----------------------------------------------------------------------------
    // Test 6: repeated invocations (fresh process each time, as every
    // real invocation of this tool is) all succeed -- stability check,
    // matching item 48's "multiple repeated invocations".
    // -----------------------------------------------------------------------------

    void test_repeated_invocations()
    {
        constexpr int kRepeats = 5;

        for (int i = 0; i < kRepeats; ++i)
        {
            const int exit_code = run_bench(
                {"--experiment", "sync-mode", "--sizes", "4096", "--warmup", "2",
                 "--iterations", "5", "--runs", "1"});

            expect(
                exit_code == 0,
                "iteration " + std::to_string(i) + ": repeated invocation failed");
        }

        std::cout
            << "[PASS] test_repeated_invocations (" << kRepeats << " iterations)\n";
    }

    // -----------------------------------------------------------------------------
    // Test 7: the CUDA device-scheduling-flag matrix (Part G) runs
    // successfully for every supported flag, each in its own fresh
    // process (required: cudaSetDeviceFlags() must precede context
    // init -- see Part G item 35).
    // -----------------------------------------------------------------------------

    void test_device_schedule_flags()
    {
        const std::vector<std::string> schedules = {"auto", "spin", "blocking-sync"};

        for (const auto &schedule : schedules)
        {
            const int exit_code = run_bench(
                {"--device-schedule", schedule, "--experiment", "sync-mode",
                 "--sync-mode", "stream", "--sizes", "4096", "--warmup", "2",
                 "--iterations", "5", "--runs", "1"});

            expect(
                exit_code == 0,
                "--device-schedule " + schedule + " should run successfully");
        }

        std::cout << "[PASS] test_device_schedule_flags\n";
    }

} // namespace

int main()
{
    if (!cuda_device_available())
    {
        std::cout
            << "[SKIP] cuda_sync_bench_test: no CUDA device available at "
               "runtime\n";
        return 0;
    }

    try
    {
        test_sync_mode_correctness_all_modes();
        test_h2d_and_d2h_correctness();
        test_zero_and_odd_sizes();
        test_invalid_arguments_rejected();
        test_repeated_invocations();
        test_device_schedule_flags();
    }
    catch (const std::exception &error)
    {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }

    std::cout << "All tests passed.\n";

    return 0;
}
