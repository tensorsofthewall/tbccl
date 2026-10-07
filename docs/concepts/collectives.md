# Collectives

`barrier`, `broadcast`, `all_gather` and `all_reduce` are asynchronous: each returns a `Work`. A communicator runs one collective at a time, in submission order.

## The ordering contract

**Every rank must issue the same collectives in the same order.** Each collective begins with every rank sending a small descriptor (sequence number, kind, root, element count, datatype, reduction op, byte count, local memory kind, forced algorithm) to rank 0 over the control plane. Rank 0 answers every rank with a verdict that carries the chosen algorithm before any payload moves.

- A rank-local capability problem (an unregistered memory kind, a reduction type the memory kind cannot do) fails the collective on **every** rank with `unsupported:` naming the rank, without poisoning the communicator.
- A disagreement on sequence, kind, root, count, datatype, op or byte count fails every rank with `protocol_mismatch:` and aborts the communicator.
- At world size 2 the collective path keeps a descriptor-free wire format; a count or size mismatch between the two ranks is **not** detected there. Worlds of size 3 and above are descriptor-checked.

Set `TBCCL_TRACE=1` for a per-rank log of sequence, kind, bytes, datatype, peer and verdict.

## Algorithms

Rank 0 chooses the algorithm for each collective as a pure function of the collective kind, world size, byte count and datatype, so all ranks always run the same plan. World size 1 is local, and world size 2 always uses a specialised path. Above 2, the planner picks among tree, recursive-doubling, ring and dissemination algorithms. The full table, thresholds and required connections are in [Collective algorithms](../reference/collective-algorithms.md).

## Datatypes and reductions

`all_reduce` is `Sum` for Float32, Float64, Int32, Int64 everywhere, Int8 and UInt8 (modulo-256 sum), and Float16 and BFloat16 only at world size 2. `broadcast` and `all_gather` are byte-generic. Reordered floating-point sums at world size above 2 are valid reductions with identical bits on every rank, but are not bitwise identical across algorithms or world sizes: see [Numerical semantics](../reference/numerical-semantics.md) and {doc}`low-precision`.

## Failure behavior

A collective's `Work` is terminal only after no TBCCL thread can still touch the caller's buffer. A transport or protocol failure aborts the communicator on every rank; see {doc}`failure-handling`.
