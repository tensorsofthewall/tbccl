"""Ready-made configuration profiles: quick, native, device, pytorch, vllm, exo and a custom template.

A profile is an ordinary configuration file (``profiles/<name>.yaml``) that names benchmarks; it holds no workload implementation. Framework workloads are run by the provider that the
benchmark belongs to. Profiles contain only ``<PLACEHOLDER>`` values: copy one, fill it (or put your values in an overlay) and run it with ``run --dry-run`` first.
"""
from __future__ import annotations

from pathlib import Path

from . import providers
from .config import load_config

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"
ORDER = ("quick", "native", "device", "pytorch", "vllm", "exo", "custom")


def list_profiles() -> list:
    rows = []
    for name in ORDER:
        path = PROFILE_DIR / f"{name}.yaml"
        if not path.is_file():
            continue
        config = load_config(path, auto_overlay=False)
        needs = sorted({pid for w in config.workloads if (pid := providers.provider_of(w["benchmark"]))})
        rows.append({"name": name, "path": str(path), "description": config.description or "", "physical": bool(config.physical), "providers": needs,
                     "hosts": len(config.hosts), "placeholders": len(_placeholders(config))})
    return rows


def _placeholders(config):
    from .config import unresolved_placeholders
    return unresolved_placeholders(config)


def format_profiles(rows) -> str:
    lines = []
    for r in rows:
        needs = ", ".join(r["providers"]) or "none (native)"
        lines.append(f"{r['name']:<8} {'physical' if r['physical'] else 'loopback':<9} providers: {needs}\n         {r['description']}\n         benchmarks/suite/profiles/{r['name']}.yaml")
    return "\n".join(lines)
