#!/bin/sh
# Start the given kvprof (a built runtime's bin/kvprof) with no llama-server and check that its GUI answers.
#   packaging/smoke.sh path/to/bin/kvprof
set -eu
kvprof=${1:?usage: smoke.sh path/to/bin/kvprof}
data=$(mktemp -d)
trap 'kill "$pid" 2>/dev/null || true; rm -rf "$data"' EXIT
KVPROF_DATA_DIR=$data "$kvprof" start --no-open --quiet --listen 127.0.0.1:18581 --port 18584 --upstream http://127.0.0.1:18580 >"$data/out" 2>&1 &
pid=$!
for _ in 1 2 3 4 5 6 7 8 9 10; do
    if curl -fsS http://127.0.0.1:18584/api/status 2>/dev/null | grep -q '"package": "kvprof"'; then
        curl -fsS http://127.0.0.1:18584/ | grep -q '<title>kvprof</title>'
        echo "ok: $("$kvprof" --version) starts and serves its page"
        exit 0
    fi
    sleep 1
done
cat "$data/out" >&2
echo "kvprof start did not answer" >&2
exit 1
