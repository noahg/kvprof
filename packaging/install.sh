#!/bin/sh
# Install kvprof with its own Python: nothing else needs to be installed first.
#
#   curl -LsSf https://github.com/noahg/kvprof/releases/latest/download/install.sh | sh
#
# Environment:
#   KVPROF_VERSION      release to install, e.g. 0.2.0 (default: the latest)
#   KVPROF_INSTALL_DIR  where the runtime goes (default: ~/.local/share/kvprof/runtime)
#   KVPROF_BIN_DIR      where the `kvprof` link goes (default: ~/.local/bin)
#   KVPROF_BASE_URL     directory holding SHA256SUMS and the tarballs (default: the GitHub release)
#
# To uninstall: remove the link and the install directory. Traces are kept elsewhere and are not touched.
set -eu

REPO=noahg/kvprof

say() { printf '%s\n' "$*"; }
die() { printf 'kvprof install: %s\n' "$*" >&2; exit 1; }
sha256() { if command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1"; else sha256sum "$1"; fi | cut -d' ' -f1; }

main() {
    command -v curl >/dev/null 2>&1 || die "curl is required"
    command -v tar >/dev/null 2>&1 || die "tar is required"

    case "$(uname -s)" in
        Darwin) os=macos ;;
        Linux) os=linux ;;
        *) die "no prebuilt runtime for $(uname -s); with Python 3.11+ use: uv tool install kvprof (or pipx install kvprof)" ;;
    esac
    case "$(uname -m)" in
        arm64 | aarch64) arch=arm64 ;;
        x86_64 | amd64) arch=x86_64 ;;
        *) die "no prebuilt runtime for $(uname -m); with Python 3.11+ use: uv tool install kvprof (or pipx install kvprof)" ;;
    esac
    # a shell running under Rosetta reports x86_64 on an Apple Silicon Mac
    if [ "$os" = macos ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
        arch=arm64
    fi
    if [ "$os" = linux ] && (ldd --version 2>&1 || true) | grep -qi musl; then
        die "the prebuilt runtime needs glibc; with Python 3.11+ use: uv tool install kvprof (or pipx install kvprof)"
    fi

    if [ -n "${KVPROF_BASE_URL:-}" ]; then
        base=$KVPROF_BASE_URL
    elif [ -n "${KVPROF_VERSION:-}" ]; then
        base=https://github.com/$REPO/releases/download/v$KVPROF_VERSION
    else
        base=https://github.com/$REPO/releases/latest/download
    fi
    install_dir=${KVPROF_INSTALL_DIR:-$HOME/.local/share/kvprof/runtime}
    bin_dir=${KVPROF_BIN_DIR:-$HOME/.local/bin}

    tmp=$(mktemp -d)
    trap 'rm -rf "$tmp"' EXIT

    # SHA256SUMS names the tarball (so its version) and gives its digest
    curl -fsSL "$base/SHA256SUMS" -o "$tmp/SHA256SUMS" || die "cannot fetch $base/SHA256SUMS"
    line=$(grep -E "  kvprof-[^ ]*-$os-$arch\.tar\.gz\$" "$tmp/SHA256SUMS" | head -1) || true
    [ -n "$line" ] || die "this release has no runtime for $os $arch"
    want=${line%% *}
    file=${line##* }
    name=${file%.tar.gz}

    say "downloading $file"
    curl -fL --retry 3 --progress-bar "$base/$file" -o "$tmp/$file" || die "cannot fetch $base/$file"
    got=$(sha256 "$tmp/$file")
    [ "$got" = "$want" ] || die "checksum mismatch for $file (got $got, expected $want)"

    mkdir -p "$install_dir" "$bin_dir"
    rm -rf "$install_dir/$name.new"
    mkdir "$install_dir/$name.new"
    tar -xzf "$tmp/$file" -C "$install_dir/$name.new"
    "$install_dir/$name.new/$name/bin/kvprof" --version >/dev/null || die "the downloaded runtime does not run on this machine"
    rm -rf "${install_dir:?}/$name"
    mv "$install_dir/$name.new/$name" "$install_dir/$name"
    rmdir "$install_dir/$name.new"
    ln -sfn "$install_dir/$name/bin/kvprof" "$bin_dir/kvprof"

    # earlier versions of this runtime
    for old in "$install_dir"/kvprof-*-"$os"-"$arch"; do
        [ "$old" = "$install_dir/$name" ] || [ ! -d "$old" ] || rm -rf "$old"
    done

    say "installed $("$bin_dir/kvprof" --version) to $install_dir/$name"
    case ":$PATH:" in
        *":$bin_dir:"*) say "run: kvprof start" ;;
        *) say "$bin_dir is not on your PATH. Add it (export PATH=\"$bin_dir:\$PATH\"), or run: $bin_dir/kvprof start" ;;
    esac
}

main "$@"
