# tbccl-info

`tbccl-info` is a small read-only tool installed with TBCCL. It prints the version, the C ABI version, the wire protocol version, the platform and the memory providers that were compiled in. It opens no sockets, starts no threads and reads no configuration.

```
$ tbccl-info
TBCCL 0.6.0
C ABI version:     1
Wire protocol:     4
Platform:          linux x86_64
Memory providers:  host, cuda
```

`tbccl-info --json` prints the same information as one JSON object:

```
{"package_version": "0.6.0", "c_abi": 1, "wire_protocol": 4, "os": "linux", "arch": "x86_64", "providers": {"host": true, "cuda": true, "metal_shared": false}}
```

Use it to check which TBCCL a machine has before building an adapter against it, or when reporting a problem. The tool is in `bin/` of an install prefix and of the native release archive; it is not a daemon and it does not manage a running TBCCL. See [versioning](versioning.md) for what the three version numbers mean.
