# Diagnostics

Run these checks, in this order, before reporting a problem or digging into a failure. Each one is read-only.

## 1. `tbccl-info`: which TBCCL is this machine using?

```sh
tbccl-info
tbccl-info --json
```

Check the public version, the package version, the **C ABI** (must be 1), the **wire protocol** (must be equal on every machine in the job) and the memory providers (`cuda` for CUDA buffers). Details: [tbccl-info](../reference/tbccl-info.md). Run it on every machine; if two machines print a different wire protocol they cannot connect.

If you use an adapter, its diagnostics report the TBCCL it was built with. Compare that with `tbccl-info`:

| Adapter | Command |
|---|---|
| torch-tbccl | `python -m torch_tbccl.info` |
| vllm-tbccl | `python -m vllm_tbccl.info` |
| exo-tbccl | `python -c "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())"` |

## 2. Is the peer reachable?

TBCCL uses plain TCP, so ordinary tools work:

```sh
ping -c3 <peer address>                          # about 0.3 to 0.5 ms on a direct Thunderbolt 4 link
ss -ltn | grep <port>                            # Linux: is the rank's listener up? (macOS: netstat -an | grep LISTEN)
```

Each rank listens on a control port and a data port; for the two-rank `peers` shorthand the data port is the control port plus 1000, and both must be free. The address a rank advertises must be reachable from the peers and must not be a wildcard. For a Thunderbolt link see the [link guide](thunderbolt-link.md).

## 3. What is each rank doing?

`TBCCL_TRACE=1` makes every rank log each collective (sequence, kind, bytes, datatype, peer, verdict). Compare the logs of all ranks: the first disagreement is usually the bug.

## 4. Reporting a problem

Include the `tbccl-info --json` output from every machine, the platform (OS, CPU, GPU, driver), the exact error text (it starts with a tag such as `protocol_mismatch:` or `timeout:`), what each rank was doing, and whether the link was Thunderbolt or another network. Do not include credentials or private network details. Security problems follow `SECURITY.md`.
