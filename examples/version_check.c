/* The smallest C consumer of an installed TBCCL: prints the version information the library reports.
 * Build: see docs/getting-started/install.md (CMake find_package or pkg-config). */
#include <stdint.h>
#include <stdio.h>

#include <tbccl/tbccl.h>
#include <tbccl/version.h>

int main(void)
{
    uint32_t abi = 0, major = 0, minor = 0, patch = 0;
    if (tbcclGetAbiVersion(&abi) != TBCCL_SUCCESS || tbcclGetPackageVersion(&major, &minor, &patch) != TBCCL_SUCCESS)
    {
        fprintf(stderr, "could not query the TBCCL version\n");
        return 1;
    }
    printf("TBCCL %s (package %u.%u.%u, C ABI %u)\n", TBCCL_VERSION_STRING, major, minor, patch, abi);
    return 0;
}
