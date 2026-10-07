# Numerical contract for N>2 reductions

## Why a contract is needed

Floating-point addition is not associative: `(a + b) + c` and `a + (b + c)` can differ in the last bits. The reference all_reduce folds the ranks in a fixed order on rank 0, so its bits are
reproducible but arbitrary. A ring, a binomial tree and recursive doubling each combine the rank values in a *different* order, so they cannot, and need not, reproduce the reference bits.

## Guarantees (world_size > 2)

For `Float32` and `Float64` SUM:

1. **Valid reduction.** Every rank receives the result of a valid reduction over all participating rank values, combined according to the selected algorithm's fixed combination order.
2. **Rank agreement.** Every rank receives the **identical bits** (the algorithms either broadcast one rank's result or perform symmetric commutative operations whose two operand orders give the same bits).
3. **Determinism.** For a fixed TBCCL version, algorithm, world size, rank order and input bits the result is deterministic: running it again gives the same bits.

Not guaranteed: bitwise identity **between algorithms** (reference / tree / recursive doubling / ring), **between world sizes**, or between **rank orderings**. Applications that need run-to-run reproducibility get it
for a fixed configuration; an application that changes world size or the forced algorithm may see last-bit differences. The default algorithm for a given `(world_size, bytes)` is chosen by rank 0 and carried in the
collective verdict, so all ranks always run the same algorithm.

Error bound: each reordering changes the result by the usual floating-point accumulation error. The tests compare every algorithm against a high-precision reference (Float32 sums accumulated in Float64; Float64 sums in
`long double` where it is wider, otherwise a compensated/double-double reference) with a relative tolerance proportional to `(N - 1) * epsilon` of the type times the sum of absolute values, plus a ULP-based check for
well-conditioned inputs. The test inputs include huge and tiny magnitudes, cancellation, alternating signs and subnormals.

Special values: `inf` / `-inf` follow IEEE semantics; a sum containing `+inf` and `-inf` is NaN; **any NaN operand makes the result NaN**. NaN payloads and sign are not specified and are not compared (tests compare "is NaN").

## Exact types

`Int32`, `Int64`, `Int8`, `UInt8` SUM are exact in the documented arithmetic (two's complement wrap-around, modulo 2^8 for the 8-bit types) **whatever the algorithm and order**, because modular addition is associative
and commutative. Tests compare these bit for bit with no tolerance.

## Still rejected

`Float16` and `BFloat16` SUM remain **rejected for world_size > 2** (their two-operand rounding semantics are not defined for sequences of roundings), whatever algorithm would be selected. The planner is never consulted for them.
Forcing an algorithm cannot enable them either: validation happens before planning.

## world_size 2

Unchanged: the specialised heterogeneous N=2 engine, with its exact one-rounding semantics for every supported type, is the only path at N=2.
