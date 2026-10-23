#!/usr/bin/env bash
# distrust_emsign.sh   Run INSIDE the manylinux (AlmaLinux 8) build container, as root, before anything is downloaded.
# The distribution's ca-certificates package still lists the emSign roots that were revoked in October 2026 (their intermediates' private keys were compromised).
# The roots are put on the p11-kit blocklist and the system bundle is regenerated, so dnf, curl and everything linked to the system OpenSSL stop trusting them.
# This only removes trust; certificate verification stays on. The script fails if an emSign certificate is still in the regenerated bundle.
set -euo pipefail
bundle=/etc/pki/tls/certs/ca-bundle.crt
extracted=/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem
dir=/etc/pki/ca-trust/source/blocklist
[ -d "$dir" ] || dir=/etc/pki/ca-trust/source/blacklist
[ -d "$dir" ] || { echo "no p11-kit blocklist directory under /etc/pki/ca-trust/source" >&2; exit 1; }

tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
emsign_count() {
    local n=0 f
    rm -f "$tmp"/c*.pem
    csplit -s -z -f "$tmp/c" -b '%03d.pem' "$1" '/-----BEGIN CERTIFICATE-----/' '{*}'
    for f in "$tmp"/c*.pem; do
        if openssl x509 -in "$f" -noout -subject -nameopt RFC2253 2>/dev/null | grep -qi 'emSign'; then n=$((n + 1)); fi
    done
    echo "$n"
}

before=$(emsign_count "$extracted")
n=0
for f in "$tmp"/c*.pem; do
    if openssl x509 -in "$f" -noout -subject -nameopt RFC2253 2>/dev/null | grep -qi 'emSign'; then
        cp "$f" "$dir/emsign-$n.pem"; n=$((n + 1))
    fi
done
update-ca-trust extract
after=$(emsign_count "$bundle")
echo "emSign roots in the system TLS bundle: $before before, $after after"
[ "$after" = 0 ]
