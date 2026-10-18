# Native performance

These are reference measurements of the native TBCCL benchmark programs on one pair of machines. They show what the benchmarks produce and how large the effects were there; they are not a statement about other hardware, other links or other software. Run the suite on your own systems for numbers that apply to them.

## Performance at a glance

| What was measured | Result on this pair of machines |
|---|---|
| 16 MiB, host memory, Linux to Mac | 15.5 Gbit/s with TBCCL; 15.81 Gbit/s with a single iperf3 stream |
| 16 MiB, host memory, Mac to Linux | 18.3 Gbit/s with TBCCL; 18.81 Gbit/s with a single iperf3 stream |
| 64 B round trip | 128.1 us with TBCCL; 128.2 us with raw TCP (a round trip, not a one-way latency) |
| 16 MiB, CUDA pinned memory to Metal shared memory | 9,968 us synchronous; 8,677 us with asynchronous chunked staging, against 8,667 us for host memory |
| `all_reduce`, two ranks | ring faster than the reference algorithm from about 1 MiB (16 MiB: 16,192 us against 18,163 us), slower below |

These are completion times and TCP goodput through two operating-system network stacks. They are **not** the Thunderbolt 4 link rate, they do not involve RDMA or direct device-to-device DMA, and they come from one pair of machines. Each chart below is drawn from the table next to it; the table is the record, the chart only makes its shape easier to see. The charts need no scripts and follow the light and dark theme.

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

TBCCL and iperf3 agree: both measure **TCP payload goodput through two operating-system kernels**, not the Thunderbolt link rate. The direction matters (B to A was 16% faster in both programs); the cause was not isolated.

```{raw} html
:file: charts/native-throughput.html
```

The chart shows that TBCCL tracks a plain single TCP stream in both directions, within about 3% here. It does not show a link rate, and 18 Gbit/s is not a general expectation: it is what these two machines and this cable gave in one direction.

Message-size sweep, A to B (median of 3-6 repetitions):

| Message size | Time | Effective rate |
|---|---|---|
| 4 KiB | 126 us | 0.3 Gbit/s |
| 64 KiB | 269 us | 1.9 Gbit/s |
| 1 MiB | 1,118 us (779-1,480) | 7.5 Gbit/s |
| 4 MiB | 2,432 us | 13.8 Gbit/s |
| 16 MiB | 8,669 us | 15.5 Gbit/s |

Small and medium messages are latency- and variability-bound; only the largest sizes approach the streaming goodput.

```{raw} html
:file: charts/native-latency-size.html
```

The chart draws only the completion-confirmed transfers: each time runs from the start of one transfer until its acknowledgement came back, so it includes the return trip. The ping-pong round trips in the Latency table (raw TCP and TBCCL, 64 B and 4 KiB) are a different measurement with a different statistic, and are not drawn on the same axes; reading them against these points would suggest a comparison the data do not support. The chart shows the scale: about 130 us for small messages, rising towards the streaming rate at 16 MiB.

## Device memory

CUDA pinned memory on host A to Metal shared memory on host B, 16 MiB, completion-confirmed, A to B, 3 repetitions:

| Path | Time | Notes |
|---|---|---|
| host memory (control) | 8,667 us | from the table above |
| CUDA pinned to Metal shared, synchronous | 9,968 us (9,957-9,984) | 1,301 us more than the control |
| CUDA pinned device-to-host staging alone, 16 MiB, local | 1,279 us (13.1 GB/s) | the extra time of the row above is this staging |
| asynchronous chunked (256 KiB chunks, depth 2) | 8,677 us (8,671-8,717) | equal to the host-memory control: staging is overlapped |

Submitting an asynchronous operation returns in microseconds and says nothing about completion; the table compares completion times. Overlap of CUDA compute with communication (4 MiB in four buckets): serial 14.1 ms, overlapped 10.0 ms, compute only 6.1 ms, communication only 8.7 ms. Metal compute overlap was not tested.

```{raw} html
:file: charts/native-staging.html
```

The asynchronous chunked path completed in the same time as the host-memory control, so in this setup the staging cost was hidden behind the transfer. That is an observed overlap in completion time. Both device paths still stage through host memory; nothing here is direct device-to-device DMA.

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

```{raw} html
:file: charts/native-collectives.html
```

The curves cross between 64 KiB and 1 MiB: ring pays more per step at small sizes and moves less data at large ones. With two ranks the ring and the reference algorithm have little to distinguish them, so this crossover is not a rule for other rank counts, and the algorithm that is fastest here is not fastest everywhere. The `auto` line follows `reference` because that is what `auto` chose.

## Limitations

* One pair of machines and one cable. Nothing here applies to other Thunderbolt generations, operating systems, GPU families or network types.
* Three to five repetitions per point: enough to see effects of tens of percent, not small ones. Where the spread is large the range is given.
* TCP goodput through two kernels, not link bandwidth. No RDMA and no direct CUDA-to-Metal DMA is involved.
* The origin of the small-message latency modes and of the direction asymmetry is not established.
* AER was unavailable (see above).

## Reproducing the measurements

Nothing on this page needs to be taken on trust, and none of it applies to your hardware until you measure it.

* Run the same benchmarks on your own machines with the [quickstart](quickstart.md) and [running on your hardware](your-hardware.md).
* Turn a run into a report with [report generation](reports.md), and compare two of your own runs with [comparing runs](comparing.md).
* What each number means and when two runs may be compared is in the [methodology](methodology.md).

## How the charts are made

The chart inputs are in `docs/benchmarks/chart-data.json`, copied by hand from the tables on this page. They are published aggregates, not raw records, and carry the exact text of each cell. `python3 tools/benchmark_charts.py` regenerates the figures in `docs/benchmarks/charts/` deterministically and `--check` fails when they are out of date; a test fails when a chart input no longer appears in the table it cites.
