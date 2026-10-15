#!/usr/bin/env python3
"""Install a TBCCL adapter wheel from its GitHub Release into an explicitly named Python environment. Standard library only.

    install_adapter.py install --adapter torch|vllm --python ENV_PYTHON [--device cpu|cuda|mps|metal] [--version V] [--dry-run] [--reinstall]
    install_adapter.py list [--adapter A]
    install_adapter.py create-env PATH [--python-version 3.13]       (creates an empty environment with uv; installs nothing else)

What it does, in order: reads the target environment (Python, platform, installed torch/vllm), picks the one wheel in release/wheels.json that matches it, checks the
installed framework against the recorded pin, downloads the wheel to a temporary file, requires the SHA-256 to agree with BOTH the inventory and the release's own
SHA256SUMS, inspects the wheel's tags, installs it with --no-deps, and runs the adapter's verification module. It never installs, upgrades or removes PyTorch or vLLM,
never guesses a filename, and never installs into an environment you did not name. Exo is installed with its own `uv sync --extra tbccl`, not by this script.
The manual installation commands in the documentation remain supported; --dry-run prints them.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_INVENTORY = "https://raw.githubusercontent.com/tensorsofthewall/tbccl/main/release/wheels.json"
VERIFY_MODULE = {"torch": "torch_tbccl.info", "vllm": "vllm_tbccl.info"}
DIST_NAME = {"torch": "torch-tbccl", "vllm": "vllm-tbccl", "exo": "exo-tbccl"}
EXIT = {"unsupported": 2, "network": 3, "integrity": 4, "install": 5}

PROBE = r"""
import json, os, platform, sys, sysconfig
from importlib import metadata
def ver(n):
    try:
        return metadata.version(n)
    except metadata.PackageNotFoundError:
        return None
pure = sysconfig.get_paths()["purelib"]
print(json.dumps({"impl": sys.implementation.name, "py": list(sys.version_info[:3]), "platform": sys.platform, "machine": platform.machine(),
  "mac": platform.mac_ver()[0], "libc": list(platform.libc_ver()), "purelib": pure, "writable": os.access(pure, os.W_OK),
  "venv": sys.prefix != sys.base_prefix, "torch": ver("torch"), "vllm": ver("vllm"), "installed": {n: ver(n) for n in ("torch-tbccl", "vllm-tbccl", "exo-tbccl")}}))
"""


class Fail(Exception):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


def say(msg=""):
    print(msg, flush=True)


def is_local(url):
    return urllib.parse.urlparse(url).hostname in ("127.0.0.1", "localhost")


def fetch(url, what):
    scheme = urllib.parse.urlparse(url).scheme
    if scheme != "https" and not (scheme == "http" and is_local(url)):
        raise Fail("unsupported", f"refusing to fetch {what} over {scheme or 'a non-URL'}: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise Fail("network", f"{what} not found or refused ({e.code}): {url}. Check the release page, or use the manual installation commands in the documentation.") from e
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise Fail("network", f"cannot download {what} ({e}): {url}. Nothing was installed; retry when the network is reachable, or install manually from a wheel you downloaded.") from e


def load_inventory(source):
    if source is None:
        local = Path(__file__).resolve().parent.parent / "release" / "wheels.json"
        source = str(local) if local.is_file() else DEFAULT_INVENTORY
    data = fetch(source, "the wheel inventory") if re.match(r"https?://", source) else Path(source).read_bytes()
    inv = json.loads(data)
    if inv.get("schema_version") != 1:
        raise Fail("unsupported", f"unsupported inventory schema {inv.get('schema_version')!r}; update this script")
    return inv


def probe(python):
    if not os.path.isfile(python) and not shutil.which(python):
        raise Fail("unsupported", f"--python {python!r} does not exist. Name the Python of the environment to install into, for example .venv/bin/python.")
    try:
        out = subprocess.run([python, "-I", "-c", PROBE], capture_output=True, text=True, timeout=120, check=True).stdout
        return json.loads(out)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        raise Fail("unsupported", f"cannot run {python!r} to inspect it: {e}") from e


def release_tuple(v):
    m = re.match(r"(\d+(?:\.\d+)*)", v or "")
    return tuple(int(x) for x in m.group(1).split(".")) if m else ()


def target_platform(env):
    if env["platform"].startswith("linux") and env["machine"] == "x86_64":
        return "linux_x86_64"
    if env["platform"] == "darwin" and env["machine"] == "arm64":
        return "macos_arm64"
    return None


def supported_list(entry):
    return ", ".join(sorted({f"{w['python']} on {w['platform']}" for w in entry["wheels"]}))


def select_wheel(adapter, entry, env):
    if env["impl"] != "cpython":
        raise Fail("unsupported", f"{env['impl']} is not supported; {DIST_NAME[adapter]} {entry['version']} has wheels for {supported_list(entry)} (CPython only).")
    plat = target_platform(env)
    if plat is None:
        raise Fail("unsupported", f"{env['platform']}/{env['machine']} is not supported; {DIST_NAME[adapter]} {entry['version']} has wheels for {supported_list(entry)}. Linux aarch64, Intel macOS and Windows have no wheel.")
    py = f"cp{env['py'][0]}{env['py'][1]}"
    match = [w for w in entry["wheels"] if w["python"] == py and w["platform"] == plat]
    if not match:
        have = sorted({w["python"] for w in entry["wheels"] if w["platform"] == plat})
        raise Fail("unsupported", f"no {DIST_NAME[adapter]} {entry['version']} wheel for CPython {env['py'][0]}.{env['py'][1]} on {plat}; wheels exist for {', '.join(have) or 'none'} on this platform. Create an environment with a supported Python, for example `{Path(sys.argv[0]).name} create-env PATH --python-version {have[-1][2]}.{have[-1][3:]}`." if have else f"no wheel for {plat}")
    w = match[0]
    if plat == "macos_arm64" and env["mac"] and release_tuple(env["mac"])[:2] < release_tuple(w["min_macos"]):
        raise Fail("unsupported", f"{w['filename']} needs macOS {w['min_macos']} or newer; this is macOS {env['mac']}.")
    if plat == "linux_x86_64" and env["libc"][0] == "glibc":
        m = re.search(r"manylinux_(\d+)_(\d+)", w["filename"])
        if m and release_tuple(env["libc"][1])[:2] < (int(m.group(1)), int(m.group(2))):
            raise Fail("unsupported", f"{w['filename']} needs glibc {m.group(1)}.{m.group(2)} or newer; this system has glibc {env['libc'][1]}.")
    return w


def check_framework(adapter, entry, env):
    fw = entry["framework"]
    if fw is None:
        return None
    installed = env[fw["name"]]
    install_hint = {"torch": "Install PyTorch first (see the torch-tbccl installation guide), then run this again.",
                    "vllm": "Install vLLM first (see the vllm-tbccl installation guide), then run this again."}[fw["name"]]
    if installed is None:
        raise Fail("unsupported", f"{fw['name']} is not installed in this environment; {DIST_NAME[adapter]} {entry['version']} needs {fw['name']} {fw.get('series', fw.get('exact'))}. This script does not install frameworks. {install_hint}")
    rel = release_tuple(installed)
    if "series" in fw:
        want, ok = fw["series"], rel[:2] == release_tuple(fw["series"])
    else:
        want, ok = "==" + fw["exact"], rel == release_tuple(fw["exact"])
    if not ok:
        raise Fail("unsupported", f"{fw['name']} {installed} is installed but {DIST_NAME[adapter]} {entry['version']} is built for {fw['name']} {want}. The framework was not changed. Use a matching {fw['name']}, or an older/newer adapter that supports {installed} if one exists.")
    return installed


def torch_cuda(python):
    """The CUDA version of the PyTorch build in the target environment ('' for a CPU/MPS build), by asking torch itself: the version string alone is not reliable (the PyPI Linux wheel is a CUDA build with no local label)."""
    r = subprocess.run([python, "-I", "-c", "import torch; print(torch.version.cuda or '')"], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise Fail("unsupported", f"the installed PyTorch cannot be imported ({r.stderr.strip().splitlines()[-1] if r.stderr.strip() else 'unknown error'}); fix the PyTorch installation first. It was not changed.")
    return r.stdout.strip()


def check_device(adapter, device, env, cuda=None):
    if device is None:
        return
    plat = target_platform(env)
    torch = env["torch"] or ""
    if device == "cuda":
        if plat != "linux_x86_64":
            raise Fail("unsupported", "CUDA is supported on Linux x86-64 only. On macOS use --device mps (or cpu); the CPU wheel is the same file.")
        if cuda == "":
            raise Fail("unsupported", f"--device cuda but the installed PyTorch ({torch}) is not a CUDA build. Install the CUDA build (`uv pip install --python <env> --torch-backend cu130 torch`) or use --device cpu. PyTorch was not changed.")
        if not os.path.exists("/proc/driver/nvidia") and not shutil.which("nvidia-smi") and not os.environ.get("TBCCL_SKIP_GPU_CHECK"):
            raise Fail("unsupported", "--device cuda but no NVIDIA driver was found on this machine (/proc/driver/nvidia, nvidia-smi). Install the driver, or use --device cpu. Set TBCCL_SKIP_GPU_CHECK=1 to skip this check (for example inside a container).")
    elif device in ("mps", "metal"):
        if plat != "macos_arm64":
            raise Fail("unsupported", f"--device {device} needs Apple silicon macOS; this is {plat or env['platform']}.")
    elif device == "cpu" and cuda:
        say(f"note: --device cpu, but PyTorch {torch} is a CUDA build ({cuda}); the adapter will run on CPU tensors only if you create them.")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def release_sums(entry):
    out = {}
    for line in fetch(entry["sums_url"], "the release's SHA256SUMS").decode().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            out[name.strip().lstrip("*")] = digest
    return out


def inspect_wheel(path, wheel, entry):
    try:
        z = zipfile.ZipFile(path)
        info = [n for n in z.namelist() if n.endswith(".dist-info/WHEEL")]
        meta = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
        tags = [l.split(":", 1)[1].strip() for l in z.read(info[0]).decode().splitlines() if l.startswith("Tag:")]
        name = re.search(r"^Version: (.+)$", z.read(meta[0]).decode(), re.M).group(1).strip()
    except (zipfile.BadZipFile, IndexError, AttributeError, KeyError) as e:
        raise Fail("integrity", f"{wheel['filename']} is not a valid wheel ({e}). It was not installed.") from e
    if name != entry["version"] or not any(t.startswith(wheel["python"] + "-") for t in tags):
        raise Fail("integrity", f"{wheel['filename']} declares version {name} and tags {tags}, which disagree with the inventory. It was not installed.")


def installer_command(choice, python, reinstall=False):
    uv = shutil.which("uv")
    if choice == "uv" or (choice == "auto" and uv):
        if not uv:
            raise Fail("unsupported", "--installer uv requested but uv is not on PATH.")
        return [uv, "pip", "install", "--python", python, "--no-deps"] + (["--reinstall"] if reinstall else [])
    return [python, "-m", "pip", "install", "--no-deps", "--disable-pip-version-check"] + (["--force-reinstall"] if reinstall else [])


def manual_commands(adapter, wheel, python):
    return [f"curl -LO {wheel['url']}", f"echo '{wheel['sha256']}  {wheel['filename']}' | sha256sum -c -        # macOS: shasum -a 256 -c -",
            f"uv pip install --python {python} --no-deps {wheel['filename']}", f"{python} -m {VERIFY_MODULE.get(adapter, '<adapter>.info')}"]


def cmd_install(args):
    if args.adapter == "exo":
        raise Fail("unsupported", "Exo is installed by its own locked environment, not by this script:\n  git clone --depth 1 --branch tbccl-integration-v2 https://github.com/tensorsofthewall/exo-fork.git\n  cd exo-fork && uv sync --no-dev --locked --extra mlx-cuda13 --extra tbccl   # Linux CUDA 13; macOS: --extra mlx --extra tbccl\nSee the exo-tbccl installation guide.")
    inv = load_inventory(args.inventory)
    entry = inv["adapters"][args.adapter]
    if args.version and args.version != entry["version"]:
        raise Fail("unsupported", f"{DIST_NAME[args.adapter]} {args.version} is not in the inventory; the supported version is {entry['version']}. Older versions are not resolved automatically; use the release page and the manual commands.")
    env = probe(args.python)
    if not env["writable"] and not args.dry_run:
        raise Fail("unsupported", f"the environment's site-packages ({env['purelib']}) is not writable by you; nothing was installed. Use your own virtual environment (`{Path(sys.argv[0]).name} create-env PATH`), not a system Python.")
    if not env["venv"]:
        say(f"warning: {args.python} is not a virtual environment; it will modify that Python's packages.")
    wheel = select_wheel(args.adapter, entry, env)
    fw = check_framework(args.adapter, entry, env)
    check_device(args.adapter, args.device, env, torch_cuda(args.python) if args.device in ("cuda", "cpu") and env["torch"] and target_platform(env) == "linux_x86_64" else None)
    current = env["installed"][DIST_NAME[args.adapter]]
    say(f"adapter     {DIST_NAME[args.adapter]} {entry['version']} ({entry['tag']})")
    say(f"wheel       {wheel['filename']}")
    say(f"sha256      {wheel['sha256']}")
    if fw:
        say(f"framework   {entry['framework']['name']} {fw} (pin {entry['framework'].get('series', '==' + entry['framework'].get('exact', ''))})")
    say(f"environment {args.python}: CPython {env['py'][0]}.{env['py'][1]}.{env['py'][2]} on {target_platform(env)}")
    say(f"installed   {current or 'not installed'}")
    if args.dry_run:
        say("\ndry run: nothing was downloaded or installed. Equivalent manual commands:")
        for c in manual_commands(args.adapter, wheel, args.python):
            say("  " + c)
        return 0
    if current == entry["version"] and not args.reinstall:
        say(f"\n{DIST_NAME[args.adapter]} {current} is already installed; skipping installation (use --reinstall to force). Verifying.")
    else:
        sums = release_sums(entry)
        if sums.get(wheel["filename"]) != wheel["sha256"]:
            raise Fail("integrity", f"the inventory hash for {wheel['filename']} does not match the release's SHA256SUMS ({sums.get(wheel['filename'], 'file not listed')}). Nothing was installed; do not work around this.")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, wheel["filename"])
            with open(path, "wb") as f:
                f.write(fetch(wheel["url"], f"the wheel {wheel['filename']}"))
            got = sha256_of(path)
            if got != wheel["sha256"]:
                raise Fail("integrity", f"checksum mismatch for {wheel['filename']}: expected {wheel['sha256']}, downloaded {got}. The file was discarded and nothing was installed.")
            inspect_wheel(path, wheel, entry)
            say("\nchecksum and wheel tags verified; installing")
            r = subprocess.run(installer_command(args.installer, args.python, args.reinstall) + [path], text=True)
            if r.returncode != 0:
                raise Fail("install", f"the installer exited with {r.returncode}; see its output above. The environment may be unchanged; check `{args.python} -m pip list`.")
    r = subprocess.run([args.python, "-m", VERIFY_MODULE[args.adapter]], capture_output=True, text=True)
    if r.returncode != 0:
        raise Fail("install", f"installed, but the verification `{args.python} -m {VERIFY_MODULE[args.adapter]}` failed:\n{r.stdout}{r.stderr}")
    say(r.stdout.strip())
    after = probe(args.python)["installed"][DIST_NAME[args.adapter]]
    say(f"\nOK: {DIST_NAME[args.adapter]} {after} installed in {args.python}")
    return 0


def cmd_list(args):
    inv = load_inventory(args.inventory)
    for name, a in inv["adapters"].items():
        if args.adapter and name != args.adapter:
            continue
        fw = a["framework"]
        say(f"{DIST_NAME[name]} {a['version']} ({a['tag']}); framework: {fw['name'] + ' ' + fw.get('series', fw.get('exact', '')) if fw else 'none (exo pins its own)'}")
        for w in a["wheels"]:
            say(f"  {w['python']:6} {w['platform']:13} {w['filename']}  {w['sha256'][:12]}")
    return 0


def cmd_create_env(args):
    uv = shutil.which("uv")
    if not uv:
        raise Fail("unsupported", "create-env needs uv (https://docs.astral.sh/uv/). Create the environment with `python3 -m venv PATH` instead.")
    if Path(args.path).exists() and any(Path(args.path).iterdir()):
        raise Fail("unsupported", f"{args.path} already exists and is not empty; choose a new path. Nothing was changed.")
    r = subprocess.run([uv, "venv", "--python", args.python_version, args.path])
    if r.returncode != 0:
        raise Fail("install", "uv could not create the environment (see above). Nothing else was done.")
    py = os.path.join(args.path, "bin", "python")
    say(f"\ncreated an empty environment: {py}\nNext: install the framework yourself, pinned as the adapter's guide says (PyTorch 2.13.x for torch-tbccl; vLLM 0.31.0 for vllm-tbccl), then `install_adapter.py install --adapter torch --python {py}`.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", help="path or URL of wheels.json (default: the file next to this script's repository, else the main branch of tbccl)")
    sub = ap.add_subparsers(dest="command", required=True)
    i = sub.add_parser("install")
    i.add_argument("--adapter", required=True, choices=("torch", "vllm", "exo"))
    i.add_argument("--python", required=True, help="the Python of the environment to install into")
    i.add_argument("--device", choices=("cpu", "cuda", "mps", "metal"))
    i.add_argument("--version")
    i.add_argument("--dry-run", action="store_true")
    i.add_argument("--reinstall", action="store_true")
    i.add_argument("--installer", choices=("auto", "uv", "pip"), default="auto")
    i.set_defaults(fn=cmd_install)
    ls = sub.add_parser("list")
    ls.add_argument("--adapter", choices=("torch", "vllm", "exo"))
    ls.set_defaults(fn=cmd_list)
    c = sub.add_parser("create-env")
    c.add_argument("path")
    c.add_argument("--python-version", default="3.13")
    c.set_defaults(fn=cmd_create_env)
    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except Fail as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT[e.kind]


if __name__ == "__main__":
    sys.exit(main())
