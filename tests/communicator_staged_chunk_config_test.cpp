// TBCCL_STAGED_CHUNK_BYTES validation. The setting is read once per process, so ctest runs this program once per value: `valid` must create a communicator,
// `invalid` must fail Communicator::create with invalid_argument naming the variable (never silently run unchunked).

#include "mesh_test_support.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using namespace mesh_test;

// `late-ignored`: a valid (or unset) value is latched by the first communicator; an invalid value set afterwards is not looked at again.
// `late-recovers`: an invalid value makes creation fail, and the next creation after the variable is fixed succeeds (a failed latch is retried).
int late_mode(const char *mode)
{
    try
    {
        if (std::strcmp(mode, "late-ignored") == 0)
        {
            { World w(2); w.run([](std::size_t, tbccl::Communicator &comm) { comm.barrier().wait(); }); }
            ::setenv("TBCCL_STAGED_CHUNK_BYTES", "128K", 1);
            { World w(2); w.run([](std::size_t, tbccl::Communicator &comm) { comm.barrier().wait(); }); }
            std::printf("a value set after the first communicator was ignored\n");
        }
        else
        {
            ::setenv("TBCCL_STAGED_CHUNK_BYTES", "128K", 1);
            bool rejected = false;
            try { World w(2); } catch (const std::exception &) { rejected = true; }
            if (!rejected) { std::fprintf(stderr, "FAILED: invalid value accepted\n"); return 1; }
            ::setenv("TBCCL_STAGED_CHUNK_BYTES", "131072", 1);
            { World w(2); w.run([](std::size_t, tbccl::Communicator &comm) { comm.barrier().wait(); }); }
            std::printf("creation succeeded after the variable was fixed\n");
        }
    }
    catch (const std::exception &e)
    {
        std::fprintf(stderr, "FAILED: %s\n", e.what());
        return 1;
    }
    std::printf("communicator_staged_chunk_config_test passed\n");
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 2 && (std::strcmp(argv[1], "late-ignored") == 0 || std::strcmp(argv[1], "late-recovers") == 0)) return late_mode(argv[1]);
    if (argc != 2 || (std::strcmp(argv[1], "valid") != 0 && std::strcmp(argv[1], "invalid") != 0))
    {
        std::fprintf(stderr, "usage: communicator_staged_chunk_config_test valid|invalid|late-ignored|late-recovers\n");
        return 2;
    }
    const bool want_valid = std::strcmp(argv[1], "valid") == 0;
    const char *value = std::getenv("TBCCL_STAGED_CHUNK_BYTES");
    std::printf("TBCCL_STAGED_CHUNK_BYTES=%s expecting %s\n", value ? ("'" + std::string(value) + "'").c_str() : "(unset)", argv[1]);
    try
    {
        World w(2);
        w.run([](std::size_t, tbccl::Communicator &comm) { comm.barrier().wait(); });
        if (!want_valid)
        {
            std::fprintf(stderr, "FAILED: an invalid value was accepted\n");
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        const std::string what = e.what();
        if (want_valid)
        {
            std::fprintf(stderr, "FAILED: a valid value was rejected: %s\n", what.c_str());
            return 1;
        }
        if (what.find("invalid_argument: TBCCL_STAGED_CHUNK_BYTES=") == std::string::npos)
        {
            std::fprintf(stderr, "FAILED: wrong error: %s\n", what.c_str());
            return 1;
        }
        std::printf("rejected as expected: %s\n", what.c_str());
    }
    std::printf("communicator_staged_chunk_config_test passed\n");
    return 0;
}
