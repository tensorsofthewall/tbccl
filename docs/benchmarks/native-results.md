# Native performance

These are reference measurements of the native TBCCL benchmark programs on one pair of machines. They show what the benchmarks produce and how large the effects were there; they are not a statement about other hardware, other links or other software. Run the suite on your own systems for numbers that apply to them.

## What was measured

| | Host A (Linux) | Host B (macOS) |
|---|---|---|
| machine | Intel Core i7-12800HX laptop, NVIDIA RTX 3070 Ti Laptop GPU (8 GiB), CUDA 13 | Apple M4 Mac mini, 16 GiB unified memory (Metal 4) |
| operating system | EndeavourOS, Linux 6.18 LTS, NVIDIA driver 610 series | macOS 26.5.1 |
| link | one direct Thunderbolt 4 cable, IP over Thunderbolt (`thunderbolt0` / `bridge0`), MTU 9000, TCP/IPv4 | |
| software | TBCCL 0.6.0 (C ABI 1, wire protocol 4), Release builds | the same |

"A to B" means host A (Linux) is the sender. Every row is the median over independent repetitions (separate process launches) of each repetition's median, and the range is the minimum and maximum over those repetitions. The number of repetitions is stated. Latencies are microseconds on one machine's own clock; no timestamps from the two machines are subtracted.

**Hardware health.** Host A was booted with `pci=noaer`, so PCIe Advanced Error Reporting was disabled for the whole campaign. Do not read these results as "zero PCIe errors". The statement that holds is: *AER unavailable; no detected link-state or interface-health regression under the available checks.* Link state, interface counters and thermal state were checked before and between runs.

## Latency

| Measurement | Value | Range over repetitions | Repetitions | Notes |
|---|---|---|---|---|
| raw TCP ping-pong round trip, 64 B | 128.2 us | 128.2-128.6 | 5 | the mean per repetition; a **round trip**, not a one-way latency |
| TBCCL point-to-point round trip, 64 B | 128.1 us | 128.0-128.1 | 5 | median per repetition (a different statistic from the TCP mean above) |
| TBCCL point-to-point round trip, 4 KiB | 128.4 us | 128.0-134.2 | 5 | one repetition at 134 us |

Native TBCCL equals raw TCP in round trip at this resolution, and nothing in these data shows a TBCCL protocol cost. For completion-confirmed 4 KiB transfers the individual process pairs fell into two groups (about 125 us and about 230-260 us); a single "small-message latency" figure does not exist for that case, and its cause was not found.

## Large transfers, host memory

One completion-confirmed message at a time, 16 MiB, TBCCL transfer benchmark; "iperf3" is a single TCP stream measured with iperf3 (receiver goodput).

| Direction | TBCCL completion time | Effective rate | iperf3 single stream | Repetitions |
|---|---|---|---|---|
| A to B | 8,667 us (8,664-8,669) | 15.5 Gbit/s | 15.81 Gbit/s | 3 |
| B to A | 7,318 us (7,312-7,333) | 18.3 Gbit/s | 18.81 Gbit/s | 3 |

TBCCL and iperf3 agree: both measure **TCP payload goodput through two operating-system kernels**, not the Thunderbolt link rate. The direction matters (B to A was 16% faster in both programs); the cause was not isolated. Message-size sweep, A to B (median of 3-6 repetitions):

| Message size | Time | Effective rate |
|---|---|---|
| 4 KiB | 126 us | 0.3 Gbit/s |
| 64 KiB | 269 us | 1.9 Gbit/s |
| 1 MiB | 1,118 us (779-1,480) | 7.5 Gbit/s |
| 4 MiB | 2,432 us | 13.8 Gbit/s |
| 16 MiB | 8,669 us | 15.5 Gbit/s |

Small and medium messages are latency- and variability-bound; only the largest sizes approach the streaming goodput.

## Device memory

CUDA pinned memory on host A to Metal shared memory on host B, 16 MiB, completion-confirmed, A to B, 3 repetitions:

| Path | Time | Notes |
|---|---|---|
| host memory (control) | 8,667 us | from the table above |
| CUDA pinned to Metal shared, synchronous | 9,968 us (9,957-9,984) | 1,301 us more than the control |
| CUDA pinned device-to-host staging alone, 16 MiB, local | 1,279 us (13.1 GB/s) | the extra time of the row above is this staging |
| asynchronous chunked (256 KiB chunks, depth 2) | 8,677 us (8,671-8,717) | equal to the host-memory control: staging is overlapped |

Submitting an asynchronous operation returns in microseconds and says nothing about completion; the table compares completion times. Overlap of CUDA compute with communication (4 MiB in four buckets): serial 14.1 ms, overlapped 10.0 ms, compute only 6.1 ms, communication only 8.7 ms. Metal compute overlap was not tested.

## Collectives

Two ranks, host memory, all_reduce, slowest rank per iteration, 3 process pairs per point:

| Payload | ring | reference | auto |
|---|---|---|---|
| 1 KiB | 138 us | 124 us | 125 us |
| 64 KiB | 254 us | 201 us | 205 us |
| 1 MiB | 855 us | 1,251 us | 1,237 us |
| 4 MiB | 3,541 us | 4,749 us | 4,683 us |
| 16 MiB | 16,192 us | 18,163 us | 18,145 us |

Ring was slower at small sizes and faster from about 1 MiB; `auto` selected `reference` at every size on this two-rank setup (see the [collective algorithms reference](../reference/collective-algorithms.md)).

## Limitations

* One pair of machines and one cable. Nothing here applies to other Thunderbolt generations, operating systems, GPU families or network types.
* Three to five repetitions per point: enough to see effects of tens of percent, not small ones. Where the spread is large the range is given.
* TCP goodput through two kernels, not link bandwidth. No RDMA and no direct CUDA-to-Metal DMA is involved.
* The origin of the small-message latency modes and of the direction asymmetry is not established.
* AER was unavailable (see above).
