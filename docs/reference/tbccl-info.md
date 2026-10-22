# tbccl-info

`tbccl-info` is a small read-only tool installed with TBCCL (in `bin/` of every install prefix and release archive). It prints the version, the interface versions, the platform and the memory providers that were compiled in. It opens no sockets, starts no threads and reads no configuration, so it is always safe to run. **It is the first command to run when something about the runtime looks wrong.**

```text
$ tbccl-info
TBCCL 0.7.0rc1
Package version:   0.7.0
C ABI version:     1
Wire protocol:     4
Platform:          linux x86_64
Memory providers:  host, cuda
```

`tbccl-info --json` prints the same information as one JSON object, for scripts and bug reports:

```json
{"public_version": "0.7.0rc1", "package_version": "0.7.0", "c_abi": 1, "wire_protocol": 4, "os": "linux", "arch": "x86_64", "providers": {"host": true, "cuda": true, "metal_shared": false}}
```

## Fields

| Field | Meaning | What to compare |
|---|---|---|
| public version (`TBCCL 0.7.0rc1`) | the release name (`0.7.0rc1`; a release candidate carries an `rcN` suffix) | what you intended to install |
| package version | the numeric `major.minor.patch` of the CMake package and `tbcclGetPackageVersion` | the minimum an adapter requires |
| C ABI version | the stable binary interface (`TBCCL_C_ABI_VERSION`) | must match what a consumer was built against |
| wire protocol | what ranks exchange on the network | **must be equal on every rank** |
| platform | operating system and CPU architecture the library was built for | the machine you run on |
| memory providers | `host`, `cuda`, `metal_shared`: the buffer kinds this build can move | `cuda` must be listed to move CUDA buffers |

`providers` reports what is compiled in, not what the machine has: `cuda` can be true on a machine without a GPU. The same strings are available to programs through the installed `tbccl/version.h` (`TBCCL_VERSION_STRING`, `TBCCL_VERSION_MAJOR`, `TBCCL_VERSION_MINOR`, `TBCCL_VERSION_PATCH`) and `tbcclGetAbiVersion` / `tbcclGetPackageVersion`.

Adapters report the TBCCL they were built with in their own diagnostics (`python -m torch_tbccl.info`, `python -m vllm_tbccl.info`); compare those fields with the `tbccl-info` output of the release you expect. See [diagnostics](../guides/diagnostics.md) and [versioning](versioning.md).
