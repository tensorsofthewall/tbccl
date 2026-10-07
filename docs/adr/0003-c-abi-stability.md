# ADR 0003: A stable C ABI, frozen at version 1

- Status: accepted
- Date: 2026-10-03

## Context

Consumers in other languages, and consumers that must not depend on the C++ ABI of a particular compiler or standard library (exo-tbccl links only the C interface), need a stable binary interface.

## Decision

`include/tbccl/tbccl.h` and `TBCCL::tbccl_c` are the stable ABI, version 1. They wrap the C++ runtime and add nothing to it. The C++ API is explicitly not an ABI promise. Rules for v1: constants are never renumbered; struct fields are never reordered or removed; exported functions are never removed or changed; growth is compatible only (append struct fields guarded by `struct_size`, add constants, functions and struct types). Enumerations are fixed-width integer typedefs with `#define` constants, never C enums. Layouts are pinned by static assertions, the exported symbols by `tests/c_api/abi_symbols_v1.txt`, and a pure-C consumer is built against an installed package in the tests. No exception crosses the boundary.

## Consequences

- The ABI version changes only if the ABI is broken, which is not planned. It is independent of the package and wire versions ([Versioning](../reference/versioning.md)).
- ABI changes need maintainer review and are checked mechanically.
- Features that cannot be expressed compatibly (for example group start/end calls) are not added to v1 ({doc}`0008-nonblocking-submission-without-group-calls`).
