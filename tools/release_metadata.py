#!/usr/bin/env python3
"""Release metadata for CI-built artifacts. This file is kept identical in the four TBCCL repositories.

    release_metadata.py sums DIR                      write DIR/SHA256SUMS for every file in DIR (sorted, sha256sum format) and verify it
    release_metadata.py sbom ARTIFACT --out FILE ...  write an SPDX 2.3 JSON document that describes one artifact
    release_metadata.py check-sbom FILE ARTIFACT      sanity-check an SBOM against the artifact it describes
    release_metadata.py compare A B                   compare two wheels or archives member by member

The SBOM names the artifact (SHA-256), its version and licence, the commit it was built from, what is statically linked into it (a TBCCL archive given with
--core-archive, and the static CUDA runtime when --bundles-cuda-runtime says it is in the artifact) and what it depends on at run time (the wheel's Requires-Dist).
Everything is derived from the artifact and from flags; nothing is taken from the machine, so the output is the same wherever it is generated.
"""
import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile

PRIVATE = re.compile(r"/(?:home|Users|mnt|tmp|var/folders)/[\w.\-]+|runner/work|\\Users\\")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_sums(args):
    out = os.path.join(args.dir, "SHA256SUMS")
    names = sorted(n for n in os.listdir(args.dir) if n != "SHA256SUMS" and os.path.isfile(os.path.join(args.dir, n)))
    if not names:
        sys.exit("no files to checksum")
    with open(out, "w") as f:
        for n in names:
            f.write(f"{sha256_file(os.path.join(args.dir, n))}  {n}\n")
    for line in open(out):  # verify the file we just wrote against the files on disk
        digest, name = line.rstrip("\n").split("  ", 1)
        if sha256_file(os.path.join(args.dir, name)) != digest:
            sys.exit(f"checksum verification failed for {name}")
    print(f"SHA256SUMS: {len(names)} files verified")
    print(open(out).read(), end="")


def read_metadata(path):
    """(kind, name, version, license, requires, extra) of a wheel, sdist or TBCCL native archive."""
    if path.endswith(".whl"):
        z = zipfile.ZipFile(path)
        meta = z.read(next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))).decode()
        kind = "wheel"
    elif path.endswith(".tar.gz") and os.path.basename(path).startswith("tbccl-"):
        t = tarfile.open(path)
        info = json.load(t.extractfile(next(m for m in t.getmembers() if m.name.endswith("share/tbccl/BUILDINFO.json"))))
        return "archive", "tbccl", info["public_version"], "Apache-2.0", [], info
    elif path.endswith(".tar.gz"):
        t = tarfile.open(path)
        meta = t.extractfile(next(m for m in t.getmembers() if m.name.endswith("/PKG-INFO"))).read().decode()
        kind = "sdist"
    else:
        sys.exit(f"unsupported artifact: {path}")
    get = lambda k: (re.search(rf"^{k}: (.+)$", meta, re.M) or [None, None])[1]
    requires = [r for r in re.findall(r"^Requires-Dist: (.+)$", meta, re.M) if "extra ==" not in r]
    return kind, get("Name"), get("Version"), get("License-Expression") or get("License"), requires, {}


def spdx_id(text):
    return "SPDXRef-" + re.sub(r"[^A-Za-z0-9.\-]", "-", text)


def cmd_sbom(args):
    kind, name, version, lic, requires, info = read_metadata(args.artifact)
    digest = sha256_file(args.artifact)
    base = os.path.basename(args.artifact)
    created = args.epoch
    pkgs, rels = [], []
    main = {
        "SPDXID": "SPDXRef-Package-main", "name": name, "versionInfo": version, "supplier": "NOASSERTION", "downloadLocation": "NOASSERTION", "filesAnalyzed": False,
        "checksums": [{"algorithm": "SHA256", "checksumValue": digest}], "licenseConcluded": lic or "NOASSERTION", "licenseDeclared": lic or "NOASSERTION",
        "copyrightText": "NOASSERTION", "homepage": args.homepage or "NOASSERTION", "primaryPackagePurpose": "ARCHIVE" if kind != "wheel" else "LIBRARY",
        "sourceInfo": f"{kind} {base} built from {args.repo}@{args.commit}", "comment": f"artifact file name: {base}",
    }
    if kind in ("wheel", "sdist"):
        main["externalRefs"] = [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl", "referenceLocator": f"pkg:pypi/{name.lower().replace('_', '-')}@{version}"}]
    pkgs.append(main)
    rels.append({"spdxElementId": "SPDXRef-DOCUMENT", "relationshipType": "DESCRIBES", "relatedSpdxElement": "SPDXRef-Package-main"})
    licenses = []
    core = None
    if args.core_archive:
        ckind, cname, cver, clic, _, cinfo = read_metadata(args.core_archive)
        core = {"SPDXID": "SPDXRef-Package-tbccl-core", "name": "tbccl", "versionInfo": cver, "supplier": "NOASSERTION", "filesAnalyzed": False, "downloadLocation": "NOASSERTION",
                "licenseConcluded": "Apache-2.0", "licenseDeclared": "Apache-2.0", "copyrightText": "NOASSERTION",
                "checksums": [{"algorithm": "SHA256", "checksumValue": sha256_file(args.core_archive)}],
                "sourceInfo": f"native archive {os.path.basename(args.core_archive)} built from commit {cinfo.get('commit', 'unknown')}"}
        pkgs.append(core)
        rels.append({"spdxElementId": "SPDXRef-Package-main", "relationshipType": "STATIC_LINK", "relatedSpdxElement": core["SPDXID"]})
        if args.bundles_cuda_runtime:
            if not cinfo.get("cuda"):
                sys.exit("--bundles-cuda-runtime needs a CUDA core archive")
            cuda = {"SPDXID": "SPDXRef-Package-cuda-runtime-static", "name": "NVIDIA CUDA Runtime (static, libcudart_static)", "versionInfo": cinfo.get("cuda_toolkit", "NOASSERTION"),
                    "supplier": "Organization: NVIDIA Corporation", "filesAnalyzed": False, "downloadLocation": "https://developer.nvidia.com/cuda-toolkit",
                    "licenseConcluded": "LicenseRef-NVIDIA-CUDA-Toolkit-EULA", "licenseDeclared": "LicenseRef-NVIDIA-CUDA-Toolkit-EULA", "copyrightText": "NOASSERTION"}
            pkgs.append(cuda)
            rels.append({"spdxElementId": "SPDXRef-Package-main", "relationshipType": "STATIC_LINK", "relatedSpdxElement": cuda["SPDXID"]})
            licenses.append({"licenseId": "LicenseRef-NVIDIA-CUDA-Toolkit-EULA", "name": "NVIDIA CUDA Toolkit End User License Agreement",
                             "extractedText": "The CUDA runtime is linked statically under the NVIDIA CUDA Toolkit End User License Agreement: https://docs.nvidia.com/cuda/eula/",
                             "seeAlsos": ["https://docs.nvidia.com/cuda/eula/"]})
    for req in requires:
        m = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)", req)
        dep = {"SPDXID": spdx_id("Package-dep-" + m.group(1)), "name": m.group(1), "versionInfo": "NOASSERTION", "supplier": "NOASSERTION", "filesAnalyzed": False,
               "downloadLocation": "NOASSERTION", "licenseConcluded": "NOASSERTION", "licenseDeclared": "NOASSERTION", "copyrightText": "NOASSERTION",
               "comment": f"run-time requirement: {req}", "externalRefs": [{"referenceCategory": "PACKAGE-MANAGER", "referenceType": "purl", "referenceLocator": f"pkg:pypi/{m.group(1).lower().replace('_', '-')}"}]}
        pkgs.append(dep)
        rels.append({"spdxElementId": "SPDXRef-Package-main", "relationshipType": "DEPENDS_ON", "relatedSpdxElement": dep["SPDXID"]})
    doc = {"spdxVersion": "SPDX-2.3", "dataLicense": "CC0-1.0", "SPDXID": "SPDXRef-DOCUMENT", "name": f"{base}-sbom",
           "documentNamespace": f"https://github.com/tensorsofthewall/{args.repo}/spdx/{base}-{digest}",
           "creationInfo": {"created": created, "creators": ["Tool: tbccl-release-metadata"]}, "packages": pkgs, "relationships": rels}
    if licenses:
        doc["hasExtractedLicensingInfos"] = licenses
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {args.out}: {len(pkgs)} packages, {len(rels)} relationships")


def cmd_check_sbom(args):
    doc = json.load(open(args.sbom))
    problems = []
    digest = sha256_file(args.artifact)
    kind, name, version, lic, requires, _ = read_metadata(args.artifact)
    main = next((p for p in doc.get("packages", []) if p.get("SPDXID") == "SPDXRef-Package-main"), None)
    if doc.get("spdxVersion") != "SPDX-2.3":
        problems.append("not an SPDX 2.3 document")
    if not main:
        problems.append("no main package")
    else:
        if main.get("name") != name or main.get("versionInfo") != version:
            problems.append(f"package {main.get('name')} {main.get('versionInfo')} != artifact {name} {version}")
        if {"algorithm": "SHA256", "checksumValue": digest} not in main.get("checksums", []):
            problems.append("the artifact SHA-256 is not recorded")
        if main.get("licenseDeclared") in (None, "NOASSERTION"):
            problems.append("no declared licence")
    if not any(r.get("relationshipType") == "DESCRIBES" for r in doc.get("relationships", [])):
        problems.append("no DESCRIBES relationship")
    deps = {p["name"] for p in doc.get("packages", []) if p.get("SPDXID", "").startswith("SPDXRef-Package-dep-")}
    for r in requires:
        if re.match(r"[A-Za-z0-9_.\-]+", r).group(0) not in deps:
            problems.append(f"requirement {r} is not in the SBOM")
    text = open(args.sbom).read()
    if PRIVATE.search(text):
        problems.append(f"private or build paths in the SBOM: {sorted(set(PRIVATE.findall(text)))}")
    try:  # a full SPDX validation when the validator is installed (CI installs it)
        from spdx_tools.spdx.parser.parse_anything import parse_file
        from spdx_tools.spdx.validation.document_validator import validate_full_spdx_document
        errors = validate_full_spdx_document(parse_file(args.sbom))
        problems += [f"SPDX validation: {e.validation_message}" for e in errors]
        print("SPDX validator: ran")
    except ImportError:
        print("SPDX validator: not installed (structural checks only)")
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        sys.exit(1)
    print(f"SBOM ok: {name} {version}, {len(doc['packages'])} packages")


# nvcc puts a per-process temporary-file identifier into symbol names (tmpxft_<pid>_<n>); it is the only run-to-run difference in a CUDA build
NVCC_ID = re.compile(rb"tmpxft_[0-9a-f]{8}_[0-9a-f]{8}")


def members(path, normalize_nvcc=False):
    fix = (lambda b: NVCC_ID.sub(b"tmpxft_00000000_00000000", b)) if normalize_nvcc else (lambda b: b)
    if path.endswith(".whl"):
        z = zipfile.ZipFile(path)
        return {n: hashlib.sha256(fix(z.read(n))).hexdigest() for n in z.namelist() if not n.endswith("/")}
    t = tarfile.open(path)
    return {m.name: hashlib.sha256(fix(t.extractfile(m).read())).hexdigest() for m in t.getmembers() if m.isfile()}


def cmd_compare(args):
    a, b = members(args.a, args.normalize_nvcc), members(args.b, args.normalize_nvcc)
    same_names = sorted(a) == sorted(b)
    diff = sorted(n for n in set(a) & set(b) if a[n] != b[n])
    print(f"members: {len(a)} vs {len(b)}; same member list: {same_names}; differing contents: {diff or 'none'}")
    print(f"whole file identical: {sha256_file(args.a) == sha256_file(args.b)}" + (" (members compared after normalizing the nvcc temporary-file identifier)" if args.normalize_nvcc else ""))
    if not same_names:
        print("only in A:", sorted(set(a) - set(b)), "only in B:", sorted(set(b) - set(a)))
        sys.exit(1)
    allowed = {n for n in diff if n.endswith(("/RECORD",))}  # RECORD lists hashes of members; it differs only when a member does
    if set(diff) - allowed:
        sys.exit(1 if args.strict else 0)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sums"); s.add_argument("dir"); s.set_defaults(f=cmd_sums)
    s = sub.add_parser("sbom"); s.add_argument("artifact"); s.add_argument("--out", required=True); s.add_argument("--repo", required=True); s.add_argument("--commit", required=True)
    s.add_argument("--epoch", required=True, help="creation time, ISO 8601 UTC (use the commit time)"); s.add_argument("--homepage"); s.add_argument("--core-archive", help="the TBCCL native archive that is statically linked into the artifact")
    s.add_argument("--bundles-cuda-runtime", action="store_true", help="the artifact contains the static CUDA runtime (libcudart_static)"); s.set_defaults(f=cmd_sbom)
    s = sub.add_parser("check-sbom"); s.add_argument("sbom"); s.add_argument("artifact"); s.set_defaults(f=cmd_check_sbom)
    s = sub.add_parser("compare"); s.add_argument("a"); s.add_argument("b"); s.add_argument("--strict", action="store_true")
    s.add_argument("--normalize-nvcc", action="store_true", help="ignore nvcc's per-process tmpxft_ identifier when comparing members"); s.set_defaults(f=cmd_compare)
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
