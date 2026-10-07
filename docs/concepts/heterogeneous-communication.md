# Heterogeneous communication

TBCCL is built to connect ranks that differ in operating system, CPU architecture and memory type: for example a Linux x86-64 host with an NVIDIA GPU (`Cuda` buffers) and an Apple-silicon Mac (`MetalShared` buffers), joined by a direct Thunderbolt 4 link or any TCP network.

## What TBCCL makes uniform

- **One wire format.** Every integer on the wire is serialized big-endian into fixed-size buffers, so nothing depends on structure layout, padding or host endianness of the control and handshake messages. Payload bytes are transferred unchanged; no byte-order conversion is applied, so both ranks must share an element byte order (both machines of the validated pair are little-endian).
- **One buffer model.** `Host`, `MetalShared` and `Cuda` buffers all enter through `BufferView`; `MetalShared` is CPU-visible memory as far as the transport is concerned.
- **One communicator.** The same `Communicator` runs with a different memory kind on each rank.

## What stays per rank

Each rank advertises its own capabilities during bootstrap, and they differ:

- A reduction type may be available on one rank's memory kind and not on another's. For example `MetalShared` supports the original four reduction types only (Float32, Float64, Int32, Int64), while `Host` and `Cuda` support every supported type. A collective that needs an unavailable combination fails on every rank with `unsupported:` naming the rank, without poisoning the communicator.
- 16-bit floating-point sums (Float16, BFloat16) use identical arithmetic on Host and CUDA (widen to float32, add once, round once to nearest even); host and device results agree bit for bit.
- Capability queries (`Communicator::capabilities()`, `tbcclCommSupportsAllReduce`) answer per rank, memory kind, datatype and operation.

## Typical pairing

The validated heterogeneous configuration is a Linux x86-64 host with an NVIDIA GPU and a macOS arm64 host using Metal shared memory, connected over Thunderbolt 4 (see [Validation](../validation/index.md) and the [Thunderbolt link guide](../guides/thunderbolt-link.md)). Framework adapters add device specifics on top: torch-tbccl maps PyTorch CUDA and MPS tensors onto `Cuda` and `MetalShared` buffers.
