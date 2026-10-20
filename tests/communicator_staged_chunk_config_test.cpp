// TBCCL_STAGED_CHUNK_BYTES validation. The setting is read once per process, so ctest runs this program once per value: `valid` must create a communicator,
// `invalid` must fail Communicator::create with invalid_argument naming the variable (never silently run unchunked).

#include "mesh_test_support.hpp"

#include <cstdio>
#include <cstring>
#include <string>

using namespace mesh_test;

int main(int argc, char **argv)
{
    if (argc != 2 || (std::strcmp(argv[1], "valid") != 0 && std::strcmp(argv[1], "invalid") != 0))
    {
        std::fprintf(stderr, "usage: communicator_staged_chunk_config_test valid|invalid\n");
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
