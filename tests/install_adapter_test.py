#!/usr/bin/env python3
"""scripts/install_adapter.py against a hermetic release: a throwaway venv with a fake torch, a fake adapter wheel and a loopback HTTP server. No network, no real frameworks."""
import base64
import contextlib
import hashlib
import http.server
import io
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import install_adapter as ia  # noqa: E402

PY = f"cp{sys.version_info[0]}{sys.version_info[1]}"
PLAT = {("Linux", "x86_64"): ("linux_x86_64", "manylinux_2_17_x86_64"), ("Darwin", "arm64"): ("macos_arm64", "macosx_14_0_arm64")}.get((platform.system(), platform.machine()))
VERSION = "0.2.0"


def make_wheel(directory, version=VERSION, corrupt=False):
    plat = PLAT[1]
    name = f"torch_tbccl-{version}-{PY}-{PY}-{plat}.whl"
    di = f"torch_tbccl-{version}.dist-info"
    files = {
        "torch_tbccl/__init__.py": b"",
        "torch_tbccl/info.py": b"import json\nprint(json.dumps({'package': '%s', 'fake': True}))\n" % version.encode(),
        f"{di}/METADATA": f"Metadata-Version: 2.1\nName: torch-tbccl\nVersion: {version}\n".encode(),
        f"{di}/WHEEL": f"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: false\nTag: {PY}-{PY}-{plat}\n".encode(),
    }
    record = "".join(f"{n},sha256={base64.urlsafe_b64encode(hashlib.sha256(d).digest()).rstrip(b'=').decode()},{len(d)}\n" for n, d in files.items())
    path = Path(directory) / name
    with zipfile.ZipFile(path, "w") as z:
        for n, d in files.items():
            z.writestr(n, d)
        z.writestr(f"{di}/RECORD", record + f"{di}/RECORD,,\n")
    good = hashlib.sha256(path.read_bytes()).hexdigest()
    if corrupt:
        path.write_bytes(path.read_bytes() + b"tamper")
    return name, good


class Quiet:
    def __enter__(self):
        self.o, self.e = io.StringIO(), io.StringIO()
        self.cm = contextlib.ExitStack()
        self.cm.enter_context(contextlib.redirect_stdout(self.o))
        self.cm.enter_context(contextlib.redirect_stderr(self.e))
        return self

    def __exit__(self, *a):
        self.cm.close()


@unittest.skipIf(PLAT is None, "installer is supported on Linux x86-64 and macOS arm64 only")
class InstallAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.serve = cls.tmp / "serve"
        cls.serve.mkdir()
        handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(cls.serve), **k)  # noqa: E731
        handler_cls = type("H", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None})
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), lambda *a, **k: handler_cls(*a, directory=str(cls.serve), **k))
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        for f in self.serve.iterdir():
            f.unlink()
        self.env = self.tmp / f"env{id(self)}"
        subprocess.run([sys.executable, "-m", "venv", str(self.env)], check=True, capture_output=True)
        self.py = str(self.env / "bin" / "python")
        self.fake_dist("torch", "2.13.0+cu130" if sys.platform.startswith("linux") else "2.13.0")

    def fake_dist(self, name, version):
        purelib = subprocess.run([self.py, "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"], capture_output=True, text=True, check=True).stdout.strip()
        d = Path(purelib) / f"{name}-{version.split('+')[0]}.dist-info"
        d.mkdir(exist_ok=True)
        (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")

    def release(self, corrupt=False, sums_hash=None, listed=True, python=PY, version=VERSION):
        name, good = make_wheel(self.serve, version, corrupt)
        if not listed:
            (self.serve / name).unlink()
        (self.serve / "SHA256SUMS").write_text(f"{sums_hash or good}  {name}\n")
        inv = {"schema_version": 1, "adapters": {"torch": {"repo": "x/y", "version": version, "tag": f"v{version}", "commit": "0" * 40, "sums_url": f"{self.base}/SHA256SUMS",
               "framework": {"name": "torch", "series": "2.13"}, "wheels": [{"filename": name, "url": f"{self.base}/{name}", "sha256": good, "python": python, "platform": PLAT[0]}]},
               "vllm": {"repo": "x/z", "version": version, "tag": f"v{version}", "commit": "0" * 40, "sums_url": f"{self.base}/SHA256SUMS", "framework": {"name": "vllm", "exact": "0.31.0"}, "wheels": []}}}
        path = self.tmp / "inv.json"
        path.write_text(json.dumps(inv))
        return str(path), name

    def run_cli(self, *args):
        with Quiet() as q:
            code = ia.main(list(args))
        return code, q.o.getvalue(), q.e.getvalue()

    def installed(self):
        r = subprocess.run([self.py, "-c", "from importlib import metadata as m;print(m.version('torch-tbccl'))"], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    def test_install_verify_and_repeat(self):
        inv, name = self.release()
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--installer", "pip")
        self.assertEqual(code, 0, err + out)
        self.assertIn("checksum and wheel tags verified", out)
        self.assertIn("OK: torch-tbccl 0.2.0", out)
        self.assertEqual(self.installed(), "0.2.0")
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--installer", "pip")
        self.assertEqual(code, 0, err)
        self.assertIn("already installed", out)
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--installer", "pip", "--reinstall")
        self.assertEqual(code, 0, err + out)
        self.assertNotIn("already installed", out)

    def test_dry_run_changes_nothing_and_prints_the_manual_commands(self):
        inv, name = self.release()
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--dry-run")
        self.assertEqual(code, 0, err)
        self.assertIn("nothing was downloaded or installed", out)
        self.assertIn("sha256sum -c", out)
        self.assertIn(name, out)
        self.assertIsNone(self.installed())

    def test_unsupported_python_lists_what_exists(self):
        inv, _ = self.release(python="cp399")
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py)
        self.assertEqual(code, 2)
        self.assertIn("no torch-tbccl 0.2.0 wheel for CPython", err)
        self.assertIsNone(self.installed())

    def test_unsupported_architecture_and_implementation(self):
        entry = {"version": "0.2.0", "wheels": [{"python": "cp313", "platform": "linux_x86_64", "filename": "f"}]}
        env = {"impl": "cpython", "platform": "linux", "machine": "aarch64", "py": [3, 13, 0], "mac": "", "libc": ["glibc", "2.39"]}
        with self.assertRaisesRegex(ia.Fail, "not supported.*Linux aarch64"):
            ia.select_wheel("torch", entry, env)
        with self.assertRaisesRegex(ia.Fail, "pypy is not supported"):
            ia.select_wheel("torch", entry, {**env, "impl": "pypy", "machine": "x86_64"})
        with self.assertRaisesRegex(ia.Fail, "needs glibc"):
            ia.select_wheel("torch", {"version": "0.2.0", "wheels": [{"python": "cp313", "platform": "linux_x86_64", "filename": "a-manylinux_2_28_x86_64.whl"}]}, {**env, "machine": "x86_64", "libc": ["glibc", "2.17"]})

    def test_wrong_framework_version_is_reported_and_not_changed(self):
        inv, _ = self.release()
        self.fake_dist("torch", "2.12.1")
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py)
        self.assertEqual(code, 2)
        self.assertIn("torch 2.12.1 is installed but torch-tbccl 0.2.0 is built for torch 2.13", err)
        self.assertIn("was not changed", err)

    def test_missing_framework_is_not_installed_for_you(self):
        inv, _ = self.release()
        shutil.rmtree(next(Path(self.env).rglob("torch-2.13.0.dist-info")))
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py)
        self.assertEqual(code, 2)
        self.assertIn("does not install frameworks", err)

    def test_cuda_cpu_pairing_and_missing_gpu_runtime(self):
        env = {"platform": "linux", "machine": "x86_64", "torch": "2.13.0+cpu"}
        with self.assertRaisesRegex(ia.Fail, "not a CUDA build"):
            ia.check_device("torch", "cuda", env, cuda="")
        with self.assertRaisesRegex(ia.Fail, "CUDA is supported on Linux x86-64 only"):
            ia.check_device("torch", "cuda", {"platform": "darwin", "machine": "arm64", "torch": "2.13.0"})
        with self.assertRaisesRegex(ia.Fail, "needs Apple silicon"):
            ia.check_device("torch", "mps", env)
        with mock.patch("os.path.exists", return_value=False), mock.patch("shutil.which", return_value=None), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TBCCL_SKIP_GPU_CHECK", None)
            with self.assertRaisesRegex(ia.Fail, "no NVIDIA driver"):
                ia.check_device("torch", "cuda", {**env, "torch": "2.13.0+cu130"}, cuda="13.0")

    def test_missing_wheel_asset_is_a_network_error(self):
        inv, _ = self.release(listed=False)
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--installer", "pip")
        self.assertEqual(code, 3)
        self.assertIn("not found or refused (404)", err)
        self.assertIsNone(self.installed())

    def test_checksum_mismatch_installs_nothing(self):
        inv, _ = self.release(corrupt=True)
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--installer", "pip")
        self.assertEqual(code, 4)
        self.assertIn("checksum mismatch", err)
        self.assertIsNone(self.installed())

    def test_inventory_and_release_sums_must_agree(self):
        inv, _ = self.release(sums_hash="0" * 64)
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--installer", "pip")
        self.assertEqual(code, 4)
        self.assertIn("does not match the release's SHA256SUMS", err)
        self.assertIsNone(self.installed())

    def test_network_failure_is_clear(self):
        inv, _ = self.release()
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
        s.close()
        data = json.loads(Path(inv).read_text())
        data["adapters"]["torch"]["sums_url"] = f"http://127.0.0.1:{dead}/SHA256SUMS"
        Path(inv).write_text(json.dumps(data))
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--installer", "pip")
        self.assertEqual(code, 3)
        self.assertIn("cannot download", err)
        self.assertIn("Nothing was installed", err)

    def test_non_https_remote_is_refused(self):
        with self.assertRaisesRegex(ia.Fail, "refusing to fetch"):
            ia.fetch("http://example.com/x", "x")

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root can write anywhere")
    def test_unwritable_environment(self):
        inv, _ = self.release()
        purelib = Path(next(Path(self.env).rglob("torch-2.13.0.dist-info"))).parent
        purelib.chmod(0o555)
        try:
            code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--installer", "pip")
        finally:
            purelib.chmod(0o755)
        self.assertEqual(code, 2)
        self.assertIn("not writable", err)

    def test_exo_is_delegated_to_its_own_sync(self):
        inv, _ = self.release()
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "exo", "--python", self.py)
        self.assertEqual(code, 2)
        self.assertIn("--extra tbccl", err)
        self.assertIn("tbccl-integration-v2", err)

    def test_unknown_version_and_bad_python_path(self):
        inv, _ = self.release()
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", self.py, "--version", "0.1.0")
        self.assertEqual(code, 2)
        self.assertIn("not in the inventory", err)
        code, out, err = self.run_cli("--inventory", inv, "install", "--adapter", "torch", "--python", "/nonexistent/python")
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err)

    def test_create_env_refuses_a_populated_directory(self):
        with mock.patch("shutil.which", return_value="/usr/bin/uv"):
            code, out, err = self.run_cli("create-env", str(self.env))
        self.assertEqual(code, 2)
        self.assertIn("not empty", err)


class Inventory(unittest.TestCase):
    def test_the_repository_inventory_matches_ecosystem_json(self):
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_wheel_inventory.py")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_list_reads_the_real_inventory(self):
        with Quiet() as q:
            code = ia.main(["--inventory", str(ROOT / "release" / "wheels.json"), "list"])
        self.assertEqual(code, 0)
        self.assertIn("torch-tbccl 0.2.0", q.o.getvalue())
        self.assertIn("cp312", q.o.getvalue())


if __name__ == "__main__":
    unittest.main()
