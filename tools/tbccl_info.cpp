// tbccl-info: prints the version, ABI and capabilities of the installed TBCCL. Read-only; opens no sockets and starts nothing.

#include <tbccl/rank_directory.hpp>
#include <tbccl/tbccl.h>
#include <tbccl/version.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace
{
    const char *platform_os()
    {
#if defined(__APPLE__)
        return "macos";
#elif defined(__linux__)
        return "linux";
#else
        return "unknown";
#endif
    }

    const char *platform_arch()
    {
#if defined(__x86_64__) || defined(_M_X64)
        return "x86_64";
#elif defined(__aarch64__) || defined(_M_ARM64)
        return "arm64";
#else
        return "unknown";
#endif
    }
} // namespace

int main(int argc, char **argv)
{
    bool json = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--json") == 0) json = true;
        else
        {
            std::fprintf(stderr, "usage: tbccl-info [--json]\n");
            return std::strcmp(argv[i], "--help") == 0 ? 0 : 2;
        }
    }

    std::uint32_t abi = 0, major = 0, minor = 0, patch = 0;
    if (tbcclGetAbiVersion(&abi) != TBCCL_SUCCESS || tbcclGetPackageVersion(&major, &minor, &patch) != TBCCL_SUCCESS)
    {
        std::fprintf(stderr, "tbccl-info: cannot read the library version\n");
        return 1;
    }
    const unsigned wire = tbccl::kWireProtocolVersion;
#ifdef TBCCL_INFO_CUDA
    const bool cuda = true;
#else
    const bool cuda = false;
#endif
#ifdef TBCCL_INFO_METAL
    const bool metal = true;
#else
    const bool metal = false;
#endif

    if (json)
    {
        std::printf("{\"public_version\": \"%s\", \"package_version\": \"%u.%u.%u\", \"c_abi\": %u, \"wire_protocol\": %u, \"os\": \"%s\", \"arch\": \"%s\", "
                    "\"providers\": {\"host\": true, \"cuda\": %s, \"metal_shared\": %s}}\n",
                    TBCCL_VERSION_STRING, major, minor, patch, abi, wire, platform_os(), platform_arch(), cuda ? "true" : "false", metal ? "true" : "false");
    }
    else
    {
        std::printf("TBCCL %s\n", TBCCL_VERSION_STRING);
        std::printf("Package version:   %u.%u.%u\n", major, minor, patch);
        std::printf("C ABI version:     %u\n", abi);
        std::printf("Wire protocol:     %u\n", wire);
        std::printf("Platform:          %s %s\n", platform_os(), platform_arch());
        std::printf("Memory providers:  host%s%s\n", cuda ? ", cuda" : "", metal ? ", metal_shared" : "");
    }
    return 0;
}
