#!/bin/sh
# Build the self-contained runtime: a pinned standalone Python with kvprof installed and a `bin/kvprof` launcher.
# The same tree is the install script's tarball and the payload of the macOS app.
#
#   packaging/build-runtime.sh <macos|linux> <arm64|x86_64>
#
# Needs the wheel in dist/ (`uv build` or `python -m build`). Writes build/runtime/kvprof-<version>-<os>-<arch>/
# and dist/kvprof-<version>-<os>-<arch>.tar.gz.
set -eu

os=${1:?usage: build-runtime.sh <macos|linux> <arm64|x86_64>}
arch=${2:?usage: build-runtime.sh <macos|linux> <arm64|x86_64>}
root=$(cd "$(dirname "$0")/.." && pwd -P)
. "$root/packaging/python.lock"

case "$arch" in arm64) cpu=aarch64 ;; x86_64) cpu=x86_64 ;; *) echo "unknown arch: $arch" >&2; exit 2 ;; esac
case "$os" in macos) triple=$cpu-apple-darwin ;; linux) triple=$cpu-unknown-linux-gnu ;; *) echo "unknown os: $os" >&2; exit 2 ;; esac
eval "want=\$SHA256_$(echo "$triple" | tr - _)"

wheel=$(ls "$root"/dist/kvprof-*-py3-none-any.whl 2>/dev/null | tail -1)
[ -n "$wheel" ] || { echo "no wheel in dist/: run \`uv build\` (or \`python -m build\`) first" >&2; exit 1; }
version=$(basename "$wheel" | sed 's/^kvprof-\(.*\)-py3-none-any\.whl$/\1/')
name=kvprof-$version-$os-$arch
out=$root/build/runtime/$name
cache=$root/build/cache
py=python${PYTHON_VERSION%.*}

sha256() { if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1"; else sha256sum "$1"; fi | cut -d' ' -f1; }

archive=cpython-$PYTHON_VERSION+$PBS_TAG-$triple-install_only_stripped.tar.gz
mkdir -p "$cache"
if [ ! -f "$cache/$archive" ] || [ "$(sha256 "$cache/$archive")" != "$want" ]; then
    echo "downloading $archive"
    curl -fL --retry 3 -o "$cache/$archive.part" \
        "https://github.com/astral-sh/python-build-standalone/releases/download/$PBS_TAG/$(echo "$archive" | sed 's/+/%2B/')"
    mv "$cache/$archive.part" "$cache/$archive"
fi
got=$(sha256 "$cache/$archive")
[ "$got" = "$want" ] || { echo "sha256 mismatch for $archive: got $got, want $want" >&2; exit 1; }

rm -rf "$out"
mkdir -p "$out/bin"
tar -xzf "$cache/$archive" -C "$out"        # -> $out/python

# What kvprof (stdlib-only, no GUI toolkit, no packaging at run time) never imports.
lib=$out/python/lib/$py
rm -rf "$lib/test" "$lib/idlelib" "$lib/tkinter" "$lib/turtledemo" "$lib/ensurepip" "$lib/pydoc_data" "$lib"/config-* \
       "$lib/site-packages/pip" "$lib"/site-packages/pip-* "$lib"/lib-dynload/_tkinter* "$lib"/lib-dynload/_test* \
       "$lib"/lib-dynload/_ctypes_test* "$lib"/lib-dynload/xxlimited* \
       "$out"/python/lib/tcl* "$out"/python/lib/tk* "$out"/python/lib/itcl* "$out"/python/lib/thread* \
       "$out"/python/lib/libtcl* "$out"/python/lib/libtk* "$out/python/lib/pkgconfig" \
       "$out/python/include" "$out/python/share"
find "$out/python/bin" -mindepth 1 ! -name 'python3*' -exec rm -rf {} +
rm -f "$out"/python/bin/python3*-config
find "$out/python" -name __pycache__ -type d -prune -exec rm -rf {} +

unzip -q "$wheel" -d "$lib/site-packages"

cat > "$out/bin/kvprof" <<'LAUNCHER'
#!/bin/sh
# kvprof with its own Python. Follows symlinks, so this file can be linked into a directory on the PATH.
self=$0
while [ -L "$self" ]; do
    target=$(readlink "$self")
    case "$target" in /*) self=$target ;; *) self=$(dirname "$self")/$target ;; esac
done
here=$(cd "$(dirname "$self")" && pwd -P)
# -I: ignore PYTHON* variables and the user's site-packages. -B: never write into the (possibly signed) tree.
exec "$here/../python/bin/python3" -I -B -m kvprof "$@"
LAUNCHER
chmod 755 "$out/bin/kvprof"
cp "$root/LICENSE" "$out/LICENSE"

# Bytecode up front (the launcher never writes any). Only possible when this machine can run the target's Python.
if "$out/python/bin/python3" -I -c pass 2>/dev/null; then
    "$out/python/bin/python3" -I -m compileall -q --invalidation-mode unchecked-hash "$lib" >/dev/null
    [ "$("$out/bin/kvprof" --version)" = "kvprof $version" ] || { echo "the built runtime does not run" >&2; exit 1; }
else
    echo "note: cannot run $triple here; shipping without precompiled bytecode (slower start)"
fi

mkdir -p "$root/dist"
tar -czf "$root/dist/$name.tar.gz" -C "$root/build/runtime" "$name"
echo "$out"
echo "$root/dist/$name.tar.gz"
