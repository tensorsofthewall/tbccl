# Report generation

A report is generated from **your own run**. No historical data and no network connection is needed.

```sh
python3 -m benchmarks.suite report <RUN_ID>                    # report.md, summary.csv, report.json, report.html (+ plots/ if matplotlib is installed)
python3 -m benchmarks.suite report <RUN_ID> --format html      # only the HTML page
python3 -m benchmarks.suite report <RUN_ID> --baseline <OTHER> # adds baseline-versus-candidate figures
python3 -m benchmarks.suite report <RUN_ID> --output my-report # choose the directory
```

A run id, a run directory or a `results.jsonl` path all work. Files are written next to the run unless `--output` says otherwise.

| File | Contents |
|---|---|
| `report.html` | One self-contained page: hosts, framework providers, a results table, points without a valid measurement, embedded plots. Inline style only, no scripts, and a content-security policy that forbids every external request, so it opens offline and nothing in it contacts a server. |
| `report.md` | The same content as Markdown, for a repository or a ticket. |
| `report.json` | The same content as one JSON document (`tbccl-bench-report/1`) for your own tooling. Valid results are under `results`; failed, timed-out and unverified points are under `points_without_valid_measurement`. |
| `summary.csv` | One row per experiment, for a spreadsheet. |
| `plots/` | Static PNG figures (message size against latency and bandwidth, baseline against candidate, repetition variability) when `matplotlib` is installed. Each carries a footer naming hardware, operation, memory backend, transport, units and sources. |

## Reading a report

* The **statistic** column says what a value is (the median over repetitions of each repetition's median, or a mean for tools that report only a mean); `min..max` and `cv` show the spread between repetitions. With fewer than three repetitions the spread is not meaningful and the report says so.
* The **backend** column separates, for example, PyTorch over torch-tbccl from PyTorch over Gloo for the same operation.
* The **framework providers** table (framework benchmarks only) names each provider, its version, the framework versions it found, the hash of the compiled extension in the adapter wheel (the TBCCL library bundled with it, which may differ from the checkout the runner came from) and the hashes of the scripts it ran. If the compatibility copy bundled with the suite was used, it says that versions were not collected.
* **Hosts** lists each machine's hardware and build identity and whether cleanup of remote processes was confirmed. Unconfirmed cleanup is stated, not hidden.
* **Points without a valid measurement** are never averaged into a result. A point that failed verification, timed out or could not run is listed with its reason.

## Sharing a report

What is safe to share, and what is not:

| In the run directory | Contains | Share it? |
|---|---|---|
| `report.md`, `report.html`, `report.json`, `summary.csv` | measurements, hardware and software descriptions, the commands that ran | after you have read it |
| `results.jsonl`, `manifest.json` | the same records the reports are made from | after you have read it |
| `raw/`, `health/` | **unredacted** program output, and link-health snapshots with interface addresses, MAC addresses and host names | no; keep them local |

In the records and reports, configured host addresses, your home directory and the suite checkout are replaced by placeholders, and every other absolute path (interpreters, provider directories, model directories) is cut to its last two components (one for a short path, so that a user name is never kept). A report can still carry host labels, hardware names and anything you typed into a label, a command or a configuration value. Read what you publish. Nothing is uploaded by the suite.
