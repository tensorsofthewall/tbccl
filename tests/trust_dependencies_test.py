"""The trust-store dependency policy (tools/check_trust_dependencies.py): the real repository is clean, each rule rejects an affected state, and historical records
are never read. Static; no network. The same file is used by all four TBCCL repositories (it is `test_trust_dependencies.py` in the adapters).
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_trust_dependencies as ctd  # noqa: E402

GOOD_RTD = f'build:\n  jobs:\n    post_create_environment:\n      - python -m pip install "pip @ https://x/{ctd.PIP_WHEEL_MARK}"\n'


class Repo:
    def __init__(self, files):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        for rel, text in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")

    def problems(self):
        return ctd.problems(self.root)

    def close(self):
        self.dir.cleanup()


class TrustDependencies(unittest.TestCase):
    def check(self, files):
        r = Repo(files)
        self.addCleanup(r.close)
        return r.problems()

    def test_the_real_repository_is_clean(self):
        self.assertEqual(ctd.problems(ROOT), [])

    def test_a_vulnerable_certifi_pin_is_rejected_and_the_fixed_one_is_not(self):
        self.assertEqual(self.check({"docs/requirements.txt": "certifi==2026.10.19\n"}), [])
        self.assertEqual(self.check({"docs/requirements.txt": "certifi==2026.10.20\n"}), [])
        self.assertTrue(self.check({"docs/requirements.txt": "certifi==2026.7.22\n"}))
        self.assertTrue(self.check({"docs/requirements.txt": "certifi==2026.10.18\n"}))
        self.assertTrue(self.check({"constraints-docs.txt": "certifi>=2026.1.4\n"}))
        self.assertTrue(self.check({"pyproject.toml": 'dependencies = ["certifi==2026.1.4"]\n'}))
        self.assertEqual(self.check({"docs/requirements.txt": "# certifi==2026.1.4 was the old pin\ncertifi==2026.10.19\n"}), [])

    def test_a_vulnerable_certifi_in_a_lockfile_is_rejected(self):
        old = 'version = 1\n\n[[package]]\nname = "certifi"\nversion = "2026.1.4"\nsource = { registry = "https://pypi.org/simple" }\n'
        self.assertTrue(self.check({"uv.lock": old}))
        self.assertEqual(self.check({"uv.lock": old.replace("2026.1.4", "2026.10.19")}), [])

    def test_historical_records_are_never_read(self):
        stale = "certifi==2026.1.4\nUV_VERSION: 0.9.18\n"
        self.assertEqual(
            self.check(
                {
                    "docs/validation/0.7.0.md": "certifi 2026.7.22 and uv 0.9.18 were current\n",
                    "docs/benchmarks/requirements-bench.txt": stale,
                    "benchmarks/results/requirements.txt": stale,
                    "release/requirements.txt": stale,
                    "CHANGELOG.md": "certifi==2026.1.4\n",
                }
            ),
            [],
        )

    def test_an_old_uv_pin_is_rejected(self):
        self.assertTrue(self.check({".github/workflows/a.yml": 'env:\n  UV_VERSION: "0.9.18"\n'}))
        self.assertTrue(self.check({".github/workflows/a.yml": "run: python -m pip install uv==0.13.3  # bootstrap_pip.sh\n"}))
        self.assertTrue(self.check({"scripts/lib.sh": 'curl -LsSf "https://astral.sh/uv/0.9.18/install.sh" | sh\n'}))
        self.assertEqual(self.check({".github/workflows/a.yml": 'env:\n  UV_VERSION: "0.13.5"\n'}), [])
        self.assertEqual(self.check({".github/workflows/a.yml": 'env:\n  UV_VERSION: "0.13.4"\n'}), [])

    def test_pip_must_be_bootstrapped_where_it_is_used(self):
        self.assertTrue(self.check({".github/workflows/a.yml": "run: python -m pip install pyyaml==6.0.3\n"}))
        self.assertTrue(self.check({"Makefile": "x:\n\t$(PY) -m pip install -r r.txt\n"}))
        self.assertTrue(self.check({"tools/ci/build.sh": "pip install -q build\n"}))
        self.assertEqual(self.check({".github/workflows/a.yml": "run: |\n  bash scripts/ci/bootstrap_pip.sh\n  python -m pip install pyyaml==6.0.3\n"}), [])
        self.assertEqual(self.check({"tools/ci/build.sh": "bash tools/ci/bootstrap_pip.sh\npip install -q build\n"}), [])
        self.assertEqual(self.check({".github/workflows/a.yml": "run: uv pip install --python env/bin/python numpy\n"}), [])

    def test_the_read_the_docs_build_installs_the_pinned_pip(self):
        self.assertTrue(self.check({".readthedocs.yaml": "version: 2\nbuild:\n  os: ubuntu-24.04\n"}))
        self.assertEqual(self.check({".readthedocs.yaml": GOOD_RTD}), [])

    def test_dnf_in_a_build_script_needs_the_emsign_blocklist_first(self):
        self.assertTrue(self.check({"scripts/ci/a.sh": "dnf install -y gcc\n"}))
        self.assertEqual(self.check({"scripts/ci/a.sh": "bash scripts/ci/distrust_emsign.sh\ndnf install -y gcc\n"}), [])

    def test_an_affected_image_digest_is_rejected(self):
        digest = next(iter(ctd.AFFECTED_IMAGE_DIGESTS))
        self.assertTrue(self.check({".github/workflows/a.yml": f"env:\n  IMAGE: quay.io/pypa/manylinux_2_28_x86_64@sha256:{digest}\n"}))
        self.assertEqual(self.check({".github/workflows/a.yml": "env:\n  IMAGE: quay.io/pypa/manylinux_2_28_x86_64@sha256:" + "0" * 64 + "\n"}), [])

    def test_tls_verification_bypasses_are_rejected(self):
        for line in (
            "pip install --trusted-host pypi.org x",
            "curl -k https://example.org/x",
            "curl -fsSLk https://example.org/x",
            "curl --insecure https://example.org/x",
            "wget --no-check-certificate https://example.org/x",
            "export PYTHONHTTPSVERIFY=0",
            "export GIT_SSL_NO_VERIFY=1",
            "export NODE_TLS_REJECT_UNAUTHORIZED=0",
            "uv pip install --allow-insecure-host example.org x",
            "r = requests.get(url, verify=False)",
        ):
            with self.subTest(line):
                self.assertTrue(self.check({"tools/ci/x.sh": "bash tools/ci/bootstrap_pip.sh\n" + line + "\n"}), line)
        self.assertEqual(self.check({"tools/ci/x.sh": "curl -fsS --retry 3 -o /dev/null https://example.org/x\n"}), [])

    def test_documentation_that_mentions_a_bypass_is_not_a_finding(self):
        self.assertEqual(self.check({"docs/guides/security.md": "Never use --trusted-host or curl -k to make a bootstrap succeed.\n"}), [])


if __name__ == "__main__":
    unittest.main()
