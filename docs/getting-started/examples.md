# Examples and a five-minute demo

Every example here is a command you can copy. They are ordered by what they need: nothing but a laptop first, two hosts last. Each row says what has actually been run, so you know whether an example is checked on every change, was run once on real hardware, or has never been run.

## Five minutes, no special hardware

Needs a Linux x86-64 or Apple silicon machine and Python 3.13. No GPU, no second computer, no model and no network configuration: both ranks are processes on `127.0.0.1`.

```sh
curl -LO https://raw.githubusercontent.com/tensorsofthewall/tbccl/main/scripts/install_adapter.py
python3 install_adapter.py create-env .venv --python-version 3.13
# Linux: the CUDA build of PyTorch is required to import the adapter (a GPU is not). macOS: drop --torch-backend.
uv pip install --python .venv/bin/python "torch==2.13.0" numpy --torch-backend cu130
python3 install_adapter.py install --adapter torch --python .venv/bin/python --device cpu
curl -LO https://raw.githubusercontent.com/tensorsofthewall/tbccl/main/tools/clean_install_smoke.py
.venv/bin/python clean_install_smoke.py
```

Expected: two lines, `rank 0: backend=tbccl result=[3.0, 30.0] OK` and the same for rank 1, and exit status 0. That is a real two-process `all_reduce` through PyTorch's `torch.distributed` and the TBCCL runtime over loopback TCP. To see the numbers behind it, run the [benchmark harness](#measuring-it-properly) the same way.

## Which example for which question

| You want to | Needs | Run | What has been run |
|---|---|---|---|
| Check an installed `torch-tbccl` works | one machine, no GPU | `tools/clean_install_smoke.py` (above) | on every change to the install tooling and weekly, on a Linux and a macOS runner |
| Check an installed `vllm-tbccl` is wired to vLLM | one machine, vLLM installed | `tools/clean_install_vllm_check.py` (run with that environment's Python) | weekly on a macOS runner (vLLM `0.31.0+cpu`); not on Linux CI (vLLM's CUDA install is too large for a hosted runner) |
| Check an Exo checkout has the adapter and the runtime | a clone of exo-fork | `uv run --no-sync python tools/setup_tbccl.py doctor` | weekly on a Linux and a macOS runner |
| Run a first collective in C or C++ | a release archive | `examples/c_quickstart.c`, `examples/two_rank_threads.cpp` ([first program](first-program.md)) | in the test suite |
| Train with DDP over TBCCL | two processes | the `torch-tbccl` [DDP guide](https://torch-tbccl.tensorsofthewall.com/en/latest/guides/ddp.html) and `examples/ddp_train.py` | loopback in its test suite; Linux CUDA to macOS MPS over Thunderbolt 4 on a development tree |
| Serve a model split across two hosts with vLLM | two hosts, a link, a local copy of the model | `examples/serve_pp2.sh` in `vllm-tbccl` ([guide](https://vllm-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html)) | Qwen3-0.6B, Linux CUDA with Mac Metal over Thunderbolt 4 (see its [validated configurations](https://vllm-tbccl.tensorsofthewall.com/en/latest/project/validated-configurations.html)) |
| Run an Exo cluster across two hosts | two hosts, a link, a local copy of the model | the `exo-tbccl` [guide](https://exo-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html) with exo-fork `tbccl-integration-v2` | once with Qwen3-0.6B-8bit on the current integration tag, and a larger model on earlier ones (see its [validated configurations](https://exo-tbccl.tensorsofthewall.com/en/latest/project/validated-configurations.html)) |

The two-host rows are not part of any automated check. They need the hardware, the link set up as in the [Thunderbolt link guide](../guides/thunderbolt-link.md), and model files you already have: none of these tools downloads weights.

## Measuring it properly

`tools/benchmark_allreduce.py` is a small, reproducible `all_reduce` benchmark for `torch-tbccl`. It exists so that anyone who measures TBCCL measures it the same way and says what they measured.

```sh
.venv/bin/python tools/benchmark_allreduce.py                  # two loopback ranks; table + tbccl-benchmark.json
.venv/bin/python tools/benchmark_allreduce.py --backend gloo   # the same loop through gloo, for a like-for-like comparison
```

What it does: an untimed warmup, then several timed loops with no correctness checking inside them, then one untimed `all_reduce` whose result is verified (the [benchmark methodology](../development/benchmark-methodology.md)). The result file records the software versions, platform, backend, settings and whether the ranks shared a machine, plus a note that the transport is TCP over IPv4 and not RDMA.

How to read it:

- **Loopback numbers are not network numbers.** Both ranks share one machine's memory and scheduler. They are useful to compare two builds or settings on the same machine, never as TBCCL's speed on a link.
- **There are no published performance figures** for TBCCL in this documentation. A claim of speed needs a result file from your own link, with the link's health recorded before and after, a repeat on another day, and the comparison run under the same conditions.
- **Two hosts:** start one rank per host yourself, for example `--rank 0 --world-size 2 --master-addr <host 0 IP> --endpoint <host 0 IP>:0 --out host0.json`, and the same with `--rank 1` on the other host. The script never contacts another machine and never changes a network setting. Check the [link health](../guides/thunderbolt-link.md) before and after, as for any physical run.

The collectives themselves and their algorithms are described in the [collective algorithms](../reference/collective-algorithms.md) reference.
