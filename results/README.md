# Reference measurements

Small measurement files kept as reference evidence from a Linux host and a Mac joined by a direct Thunderbolt 4 link. They are not produced by the test suite and are not part of any pass or fail check.

| Path | Contents |
|---|---|
| `iperf-linux-to-mac-mtu9000.json`, `iperf-mac-to-linux-mtu9000.json` | iperf3 3.21 TCP runs (one stream, 30 s) in each direction with an MTU of 9000 |
| `linux-to-mac.csv`, `mac-to-linux.csv` | TCP ping-pong and stream results by message size (`benchmarks/tcp_pingpong.cpp`) |
| `latency-v1/` | Round-trip latency by message size with and without busy polling; `summary.csv` and the plot are the inputs and output of `scripts/plot_latency.py` |

The host name recorded by iperf3 in the Mac-to-Linux file was replaced by a generic name. Numbers depend on the machines, cables and settings used; treat them as examples of a measurement, not as performance claims. Use the benchmark methodology in the documentation for new measurements.
