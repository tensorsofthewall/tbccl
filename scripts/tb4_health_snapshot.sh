#!/bin/sh
# Read-only. Missing devices/optional commands are reported in the JSON snapshot.
exec python3 "$(dirname "$0")/tb4_health_snapshot.py" "$@"
