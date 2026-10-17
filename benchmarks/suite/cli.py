"""Command-line interface: python3 -m benchmarks.suite {list,inventory,doctor,providers,profiles,run,import,compare,report} ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import SUITE_VERSION, health, inventory, profiles, providers, provenance, schema
from .adapters import REGISTRY
from .config import ConfigError, load_config
from .importers import ImportFailure, import_path
from .preflight import Check, HealthMonitor, check_config, check_environment, check_providers, failed, format_checks
from .reporting import compare as rcompare
from .reporting import plots, summary
from .runner import RunContext, Runner, format_matrix

DEFAULT_OUTPUT = inventory.REPO_ROOT / "results" / "bench-suite-local"
EPILOG = "Benchmark tooling only: not part of the libtbccl runtime API. Run from the repository root. See docs/development/benchmark-suite.md."


def _out_root(args):
    return Path(args.output_dir) if getattr(args, "output_dir", None) else DEFAULT_OUTPUT


def _load_run(ref, root):
    """A run directory, a results file, or a run id under the output root. Returns (records, manifest, source paths)."""
    path = Path(ref)
    if not path.exists() and (root / ref).exists():
        path = root / ref
    if path.is_dir():
        results = path / "results.jsonl"
        if not results.exists():
            raise ImportFailure(f"{path}: no results.jsonl")
        manifest = json.loads((path / "manifest.json").read_text()) if (path / "manifest.json").exists() else None
        return schema.read_records(results), manifest, [str(results)]
    if path.exists():
        return schema.read_records(path), None, [str(path)]
    raise ImportFailure(f"run '{ref}' not found (looked for a path and under {root})")


def cmd_list(args):
    rows = []
    for a in REGISTRY.values():
        c = a.caps
        rows.append({"name": c.name, "category": c.category, "integrated": c.integrated, "executable": c.executable, "operations": list(c.operations), "algorithms": list(c.algorithms),
                     "memory_backends": list(c.memory_backends), "ranks": f"{c.min_ranks}-{c.max_ranks or 'N'}", "warmup_option": c.supports_warmup, "iterations_option": c.supports_iterations,
                     "verification": c.verification, "description": c.description, "notes": c.notes})
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    for r in rows:
        print(f"{r['name']}  [{r['category']}]  {'integrated' if r['integrated'] else 'inventoried only (not integrated)'}\n    {r['description']}\n    executable: {r['executable']} | ranks: {r['ranks']} | operations: {', '.join(r['operations'])}")
        if r["integrated"]:
            print(f"    algorithms: {', '.join(r['algorithms']) or '-'} | memory backends: {', '.join(r['memory_backends'])}\n    warmup option: {r['warmup_option']} | iterations option: {r['iterations_option']} | "
                  f"native verification: {', '.join(k for k, v in r['verification'].items() if v == 'native') or 'none'}")
            if r["notes"]:
                print(f"    note: {r['notes']}")
    return 0


def cmd_inventory(args):
    inv = inventory.inventory(args.build_dir or ("build-release", "build-bench", "build"))
    print(json.dumps(inv, indent=2) if args.json else inventory.format_inventory(inv))
    return 0


def cmd_doctor(args):
    checks = check_environment()
    inv = inventory.inventory()
    for name, e in inv["executables"].items():
        checks.append(Check(name, "ok" if e["found"] else "warn", e["path"] or "not found"))
    config = None
    if args.config:
        try:
            config = load_config(args.config, args.overlay)
        except ConfigError as error:
            print(error)
            return 2
        ctx = RunContext(config)
        points = [p for p in Runner(config, ctx, allow_physical=args.allow_physical).points()]
        checks += check_config(config, points, ctx, args.allow_physical, dry_run=True)
        if args.probe:
            checks += [c for c in check_providers(config, points, probe=True) if c.status != "skip"]
            runner = Runner(config, ctx)
            runner.prepare(points)
            for hid, info in ctx.host_provenance.items():
                if info.get("error"):
                    checks.append(Check(f"probe@{hid}", "fail", info["error"]))
                    continue
                for name, e in info["executables"].items():
                    checks.append(Check(f"build@{hid}:{name}", "ok" if e["status"] == "stamped" else "warn",
                                                  f"{e['status']} ({e['reason']}); sha256 {str(e['executable_sha256'])[:12]}, {e['build_type']}, {e['compiler']}, TBCCL {e['tbccl_package_version']} / ABI {e['c_abi']} / wire {e['wire_protocol']}"))
            for problem in runner.provenance_problems:
                checks.append(Check("host consistency", "fail", problem))
            for problem in getattr(runner, "route_problems", []):
                checks.append(Check("data path", "fail", problem))
            for note in getattr(runner, "management_notes", []):
                checks.append(Check("management path", "ok", note))
            if config.physical and not getattr(runner, "route_problems", None):
                checks.append(Check("data path", "ok", "the route from each host to its peer's benchmark address leaves through that host's configured data interface"))
        if config.physical and (args.snapshot or args.probe):
            monitor = _monitor(config)
            snap = monitor.snapshot()
            failures, warnings = monitor.baseline_problems(snap)
            for host in monitor.hosts:
                checks.append(Check(f"health@{host.id}", "fail" if any(f.startswith(host.id + ":") for f in failures) else "ok", "read-only snapshot taken; nothing was changed"))
            checks += [Check("health signal", "fail", f) for f in failures] + [Check("health signal", "warn", w) for w in warnings]
    print(format_checks(checks))
    bad = failed(checks)
    print(f"\n{len(bad)} failing check(s)." if bad else "\nNo failing checks. Warnings only limit what can be run; physical links are never touched without --allow-physical.")
    return 1 if bad else 0


def cmd_providers(args):
    """Which benchmark providers are usable, per host. Read-only: a provider's describe loads no model, framework runtime or device."""
    config = None
    if args.config:
        try:
            config = load_config(args.config, args.overlay)
        except ConfigError as error:
            print(error)
            return 2
    ctx = RunContext(config) if config else None
    rows = providers.discover(config, ctx, probe=not args.no_probe, remote_describe=providers.remote_describe) if config else [
        {"provider_id": "native", "host": None, "status": "available", "reasons": ["built in"], "description": None},
        *({"provider_id": pid, "host": None, "status": "not_configured", "reasons": ["no configuration given: pass a CONFIG whose hosts set executables.%s and executables.%s" % (spec["python"], spec["dir"])], "description": None}
          for pid, spec in providers.PROVIDERS.items())]
    print(json.dumps(rows, indent=2) if args.json else providers.format_rows(rows))
    return 0


def cmd_profiles(args):
    rows = profiles.list_profiles()
    print(json.dumps(rows, indent=2) if args.json else profiles.format_profiles(rows) + "\n\nCopy a profile, fill its <PLACEHOLDER> values (or put them in an overlay), then: python3 -m benchmarks.suite run FILE --dry-run")
    return 0


def _filter(points, args):
    if args.benchmark:
        points = [p for p in points if p.benchmark == args.benchmark]
    if args.operation:
        points = [p for p in points if p.operation == args.operation]
    if getattr(args, "workload", None):
        points = [p for p in points if p.workload == args.workload]
    if getattr(args, "orientation", None):
        points = [p for p in points if p.orientation == args.orientation]
    return points


def _monitor(config):
    used = {r["host"] for r in config.ranks}
    hosts = [h for h in config.hosts.values() if h.id in used and (h.health_snapshot or h.health_command or h.health_script or config.topology.get("interconnect", "").startswith("thunderbolt"))]
    gpu_hosts = {r["host"] for r in config.ranks if str(r["memory_backend"]).startswith("cuda")}
    busy = sorted({a.caps.executable for a in REGISTRY.values() if a.caps.integrated} | {"tbccl_async_transfer_bench", "tbccl_async_overlap_bench", "tbccl_cuda_overlap_bench"})
    return HealthMonitor(hosts, gpu_hosts=gpu_hosts, strict=bool((config.defaults.get("health") or {}).get("strict", True)), busy=busy)


def cmd_run(args):
    try:
        config = load_config(args.config, args.overlay)
    except ConfigError as error:
        print(error, file=sys.stderr)
        return 2
    ctx = RunContext(config)
    monitor = None
    if config.physical and not args.dry_run:
        monitor = _monitor(config)
    runner = Runner(config, ctx, output_dir=args.output_dir, label=args.label, allow_physical=args.allow_physical, monitor=monitor)
    points = _filter(runner.points(), args)
    if not points:
        print("no workload points match the filters", file=sys.stderr)
        return 2
    if args.dry_run:
        matrix, commands = format_matrix(points, config, ctx)
        checks = check_config(config, points, ctx, args.allow_physical, dry_run=True)
        total = sum(p.repetitions for p in points if not p.skip_reason)
        print(f"configuration: {config.name} ({config.source})  overlays: {len(config.overlays)}\ntopology: {config.topology['kind']} / {config.topology['interconnect']} / transport {config.topology['transport']}"
              f"{' / PHYSICAL (needs --allow-physical)' if config.physical else ''}\n")
        print(matrix)
        print(f"\ncommands (first repetition of each runnable point; ports are chosen at run time):\n{commands}")
        print(f"\npreflight (dry run):\n{format_checks(checks)}")
        runnable = [p for p in points if not p.skip_reason]
        volume = sum((p.estimated_bytes or 0) * p.repetitions for p in runnable)
        lim = config.limits
        print(f"\nestimated link volume: {volume / 2**30:.2f} GiB over {len(runnable)} point(s) (per-repetition ceiling {max([p.estimated_bytes or 0 for p in runnable] or [0]) / 2**20:.0f} MiB); "
              f"limits: per point {('%.1f GiB' % (lim['max_point_bytes'] / 2**30)) if lim.get('max_point_bytes') else 'none'}, total {('%.1f GiB' % (lim['max_total_bytes'] / 2**30)) if lim.get('max_total_bytes') else 'none'}, "
              f"duration {('%g s' % lim['max_campaign_s']) if lim.get('max_campaign_s') else 'none'}"
              + ("  ** the estimate exceeds the total limit; the run will stop when it is reached **" if lim.get("max_total_bytes") and volume > lim["max_total_bytes"] else ""))
        print(f"\n{sum(1 for p in points if not p.skip_reason)} runnable point(s), {total} native invocation group(s), {sum(1 for p in points if p.skip_reason)} point(s) not runnable. Nothing was launched.")
        return 0
    try:
        manifest, records = runner.run(points)
    except PermissionError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 2
    print(f"\nrun {manifest['run_id']}: {manifest['status']}  {manifest.get('counts')}\nresults: {runner.output_root / manifest['run_id']}")
    if manifest.get("cleanup_unconfirmed"):
        print(f"WARNING: cleanup of {len(manifest['cleanup_unconfirmed'])} supervised rank(s) could NOT be confirmed (see the manifest's cleanup_unconfirmed and each record's artifacts.ranks)", file=sys.stderr)
    return 0 if manifest["status"] in ("success",) else 1


def cmd_import(args):
    root = _out_root(args)
    status = 0
    for path in args.path:
        try:
            result = import_path(path, args.set, args.skip_bad_rows)
        except ImportFailure as error:
            print(f"import failed: {error}", file=sys.stderr)
            status = 1
            continue
        run_id = result.records[0]["run_id"] + (f"-{args.label}" if args.label else "")
        run_dir = root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        for r in result.records:
            r["run_id"] = run_id
        schema.write_jsonl(run_dir / "results.jsonl", result.records)
        (run_dir / "manifest.json").write_text(json.dumps({"schema_version": schema.MANIFEST_VERSION, "run_id": run_id, "suite_version": SUITE_VERSION, "status": "imported",
                                                           "source_format": result.source_format, "source_file": Path(path).name, "records": len(result.records),
                                                           "rows_skipped": result.skipped, "imported_utc": schema.utc_now(),
                                                           "note": "historical measurements; provenance 'imported'; fields the source lacks are 'unknown', not inferred"}, indent=2))
        print(f"{Path(path).name}: {len(result.records)} record(s) as {result.source_format} -> {run_dir}" + (f" ({len(result.skipped)} row(s) skipped)" if result.skipped else ""))
    return status


def cmd_compare(args):
    root = _out_root(args)
    try:
        a, _, src_a = _load_run(args.run_a, root)
        b, _, src_b = _load_run(args.run_b, root)
    except (ImportFailure, schema.SchemaError) as error:
        print(error, file=sys.stderr)
        return 2
    result = rcompare.compare(a, b, contextual=args.contextual)
    text = json.dumps(result, indent=2, default=str) if args.json else rcompare.to_markdown(result, args.run_a, args.run_b)
    if args.output:
        Path(args.output).write_text(text)
    print(text)
    return 0 if result["comparisons"] else 1


def cmd_report(args):
    root = _out_root(args)
    try:
        records, manifest, sources = _load_run(args.run, root)
        base = _load_run(args.baseline, root) if args.baseline else None
    except (ImportFailure, schema.SchemaError) as error:
        print(error, file=sys.stderr)
        return 2
    formats = {f.strip() for f in args.format.split(",") if f.strip()}
    unknown = formats - {"md", "csv", "html", "json"}
    if unknown or not formats:
        print(f"unknown report format(s) {sorted(unknown) or args.format!r}; choose from md, csv, html, json", file=sys.stderr)
        return 2
    rows = summary.summarize(records)
    out = Path(args.output) if args.output else (Path(sources[0]).parent if Path(sources[0]).name == "results.jsonl" else Path(sources[0]).with_suffix(""))
    out.mkdir(parents=True, exist_ok=True)
    title = f"Benchmark report: {args.run}"
    written = []
    if "md" in formats:
        (out / "report.md").write_text(summary.to_markdown(rows, records, title, manifest))
        written.append("report.md")
    if "csv" in formats:
        (out / "summary.csv").write_text(summary.to_csv(rows))
        written.append("summary.csv")
    if "json" in formats:
        (out / "report.json").write_text(summary.to_json(rows, records, title, manifest))
        written.append("report.json")
    made = []
    if not args.no_plots:
        if not plots.available():
            print("matplotlib is not installed: plots skipped (pip install matplotlib)")
        else:
            comparison = rcompare.compare(base[0], records) if base else None
            made = plots.make_all(records, out / "plots", sources + (base[2] if base else []), baseline=base[0] if base else None, comparison=comparison, names=(args.baseline or "baseline", args.run))
    if "html" in formats:
        images = [(Path(p).stem.replace("_", " "), Path(p).read_bytes()) for p in made if str(p).endswith(".png") and Path(p).is_file()]
        (out / "report.html").write_text(summary.to_html(rows, records, title, manifest, images))
        written.append("report.html")
    print("wrote " + ", ".join(str(out / name) for name in written))
    if not args.no_plots and plots.available():
        print("plots: " + (", ".join(made) if made else "none (no suitable data)"))
    return 0


def cmd_provenance(args):
    try:
        path, stamp = provenance.stamp_build(args.build_dir, args.repo or inventory.REPO_ROOT, args.executable)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    print(f"wrote {path}\n  commit {stamp['repository_commit']}{' (tree was dirty)' if stamp['repository_dirty'] else ''}\n  executables: {', '.join(f'{n} {h[:12]}' for n, h in stamp['executables'].items())}")
    if stamp["repository_dirty"]:
        print("warning: the working tree had uncommitted changes, so this stamp does not identify a commit exactly", file=sys.stderr)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="python3 -m benchmarks.suite", description=f"TBCCL benchmark suite {SUITE_VERSION}: drive the existing native benchmarks, normalise and compare their results.",
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    s = sub.add_parser("list", help="registered benchmarks and their capabilities", description="List registered benchmarks, the operations, algorithms and memory backends each supports, and whether it is integrated.")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_list)

    s = sub.add_parser("inventory", help="detected executables, versions, hardware and tools", description="Show detected executables with build metadata, the local hardware, optional modules and iperf3. Read-only.")
    s.add_argument("--json", action="store_true")
    s.add_argument("--build-dir", action="append", help="build directory to search (repeatable; TBCCL_BUILD_DIR is searched first)")
    s.set_defaults(func=cmd_inventory)

    s = sub.add_parser("doctor", help="is this host (and optional configuration) ready?", description="Check Python and optional modules, executables and, with CONFIG, the configuration's hosts, executables and physical opt-in. Read-only; no sudo.")
    s.add_argument("config", nargs="?")
    s.add_argument("--overlay", action="append", help="local overlay file (repeatable)")
    s.add_argument("--allow-physical", action="store_true")
    s.add_argument("--snapshot", action="store_true", help="for a physical configuration, also take one read-only link-health snapshot per host")
    s.add_argument("--probe", action="store_true", help="probe every host the configuration uses (hardware, executables, build provenance) and, if physical, evaluate its health signals; read-only")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("providers", help="which framework benchmark providers are available", description="List the built-in native benchmarks and every external provider (torch-tbccl, vllm-tbccl, exo-tbccl) configured on each host of CONFIG, with the provider's own report of its environment. A provider that is missing or unsupported is reported with its reason; it never stops the native benchmarks. Read-only.")
    s.add_argument("config", nargs="?")
    s.add_argument("--overlay", action="append", help="local overlay file (repeatable)")
    s.add_argument("--no-probe", action="store_true", help="only report what is configured; run no provider")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_providers)

    s = sub.add_parser("profiles", help="ready-made configurations: quick, native, device, pytorch, vllm, exo, custom", description="List the shipped configuration profiles, whether each needs a physical link and which framework providers it uses. Profiles hold placeholders only; copy one and fill it in.")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_profiles)

    s = sub.add_parser("provenance", help="stamp a build with the commit it was built from", description="Record the repository commit and the executables' SHA-256 next to a build, so results can attribute the binary to a commit. Run it on the host that built the binaries, right after building.")
    s.add_argument("action", choices=["stamp"])
    s.add_argument("--build-dir", required=True)
    s.add_argument("--repo", help="the source checkout the build came from (default: this repository)")
    s.add_argument("--executable", action="append", help="executable to record (repeatable; default: every known benchmark executable present)")
    s.set_defaults(func=cmd_provenance)

    s = sub.add_parser("run", help="validate a configuration and run its workloads", description="Validate CONFIG, expand its workload matrix and run it. --dry-run prints the matrix and the exact commands without launching anything.")
    s.add_argument("config")
    s.add_argument("--dry-run", action="store_true", help="print the workload matrix and commands; launch nothing")
    s.add_argument("--output-dir", help="where run directories are created (default: results/bench-suite-local, which is git-ignored)")
    s.add_argument("--label", help="free-text label stored in the run id and manifest")
    s.add_argument("--overlay", action="append", help="local overlay file merged over CONFIG (repeatable); CONFIG.local.yaml is used automatically if present")
    s.add_argument("--allow-physical", action="store_true", help="permit a configuration that uses physical hosts (opt-in)")
    s.add_argument("--benchmark", help="run only this benchmark")
    s.add_argument("--operation", help="run only this operation")
    s.add_argument("--workload", help="run only the workload with this name")
    s.add_argument("--orientation", choices=["forward", "reverse"], help="run only this rank orientation")
    s.set_defaults(func=cmd_run)

    s = sub.add_parser("import", help="normalise historical CSV/JSON/JSONL results", description="Import tb_pingpong / latency_sweep / collective / tensor-transfer CSV, small_message_latency JSONL, iperf3 and allreduce-harness JSON.")
    s.add_argument("path", nargs="+")
    s.add_argument("--output-dir")
    s.add_argument("--label")
    s.add_argument("--set", action="append", metavar="PATH=VALUE", help="add context the file lacks, e.g. environment.topology.kind=direct-link (JSON values allowed)")
    s.add_argument("--skip-bad-rows", action="store_true", help="skip malformed rows instead of rejecting the file (skipped rows are listed in the manifest)")
    s.set_defaults(func=cmd_import)

    s = sub.add_parser("compare", help="compare two runs", description="Compare equivalent experiments of two runs (run id, directory or results file). Non-equivalent pairs are listed with reasons, not compared.")
    s.add_argument("run_a", metavar="RUN_A", help="baseline")
    s.add_argument("run_b", metavar="RUN_B", help="candidate")
    s.add_argument("--output-dir")
    s.add_argument("--contextual", action="store_true", help="also show labelled comparisons between non-equivalent experiments")
    s.add_argument("--json", action="store_true")
    s.add_argument("--output", help="also write the comparison here")
    s.set_defaults(func=cmd_compare)

    s = sub.add_parser("report", help="Markdown/CSV summary and plots for a run", description="Write report.md, summary.csv, report.json, a self-contained offline report.html and (if matplotlib is installed) static plots for RUN_ID.")
    s.add_argument("run", metavar="RUN_ID")
    s.add_argument("--output-dir")
    s.add_argument("--output", help="directory for the report files (default: the run directory)")
    s.add_argument("--baseline", help="a second run to draw baseline-vs-candidate from")
    s.add_argument("--no-plots", action="store_true")
    s.add_argument("--format", default="md,csv,html,json", help="comma-separated report files to write: md, csv, html (one self-contained offline page), json (default: all four)")
    s.set_defaults(func=cmd_report)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)
