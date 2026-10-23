#!/usr/bin/env bash
# distrust_emsign.sh   Run INSIDE the manylinux (AlmaLinux 8) build container, as root, before anything is downloaded.
# The distribution's ca-certificates package still lists the emSign root that was revoked in October 2026 (emSign ECC Root CA - G3; the private keys of
# eight eMudhra intermediates were compromised). Python's certifi 2026.10.19 removes exactly that root and keeps the unaffected emSign roots, and so does this
# script: the root is put on the p11-kit blocklist and the system bundle is regenerated, so dnf, curl and everything linked to the system OpenSSL stop
# trusting it. This only removes trust; certificate verification stays on. The script fails if the root is still in the regenerated bundle.
set -euo pipefail
bundle=/etc/pki/tls/certs/ca-bundle.crt
extracted=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
dir=/etc/pki/ca-trust/source/blocklist
[ -d "$dir" ] || dir=/etc/pki/ca-trust/source/blacklist
[ -d "$dir" ] || { echo "no p11-kit blocklist directory under /etc/pki/ca-trust/source" >&2; exit 1; }
REVOKED='CN=emSign ECC Root CA - G3,'

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
revoked_in() {
    local n=0 f
    rm -f "$tmp"/c*.pem
    csplit -s -z -f "$tmp/c" -b '%03d.pem' "$1" '/-----BEGIN CERTIFICATE-----/' '{*}'
    for f in "$tmp"/c*.pem; do
        if openssl x509 -in "$f" -noout -subject -nameopt RFC2253 2>/dev/null | grep -qF "$REVOKED"; then n=$((n + 1)); fi
    done
    echo "$n"
}

before=$(revoked_in "$extracted")
n=0
for f in "$tmp"/c*.pem; do
    if openssl x509 -in "$f" -noout -subject -nameopt RFC2253 2>/dev/null | grep -qF "$REVOKED"; then
        cp "$f" "$dir/emsign-ecc-g3-$n.pem"; n=$((n + 1))
    fi
done
update-ca-trust extract
after=$(revoked_in "$bundle")
echo "emSign ECC Root CA - G3 in the system TLS bundle: $before before, $after after"
[ "$after" = 0 ]
