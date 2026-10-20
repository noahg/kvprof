#!/bin/sh
# Assemble kvprof.app: the menu-bar executable around the self-contained runtime (packaging/build-runtime.sh).
#
#   macos/build-app.sh [arm64|x86_64]        (default: this machine's architecture)
#
# Needs the wheel in dist/ (`uv build`). Writes build/app/<arch>/kvprof.app, ad-hoc signed: it runs on the machine
# that built it. macos/sign-and-notarize.sh turns it into something to hand out.
set -eu

arch=${1:-$(uname -m)}
root=$(cd "$(dirname "$0")/.." && pwd -P)
bundle_id=${BUNDLE_ID:-io.github.noahg.kvprof}

wheel=$(ls "$root"/dist/kvprof-*-py3-none-any.whl 2>/dev/null | tail -1)
[ -n "$wheel" ] || { echo "no wheel in dist/: run \`uv build\` first" >&2; exit 1; }
version=$(basename "$wheel" | sed 's/^kvprof-\(.*\)-py3-none-any\.whl$/\1/')

"$root/packaging/build-runtime.sh" macos "$arch" >/dev/null
runtime=$root/build/runtime/kvprof-$version-macos-$arch

swift build -c release --arch "$arch" --package-path "$root/macos"
exe=$(swift build -c release --arch "$arch" --package-path "$root/macos" --show-bin-path)/KvprofApp

app=$root/build/app/$arch/kvprof.app
rm -rf "$app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp "$exe" "$app/Contents/MacOS/kvprof-app"
cp -R "$runtime" "$app/Contents/Resources/runtime"
cp "$root/macos/AppIcon.icns" "$app/Contents/Resources/AppIcon.icns"
cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key><string>$bundle_id</string>
    <key>CFBundleName</key><string>kvprof</string>
    <key>CFBundleDisplayName</key><string>kvprof</string>
    <key>CFBundleExecutable</key><string>kvprof-app</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
    <key>CFBundleShortVersionString</key><string>$version</string>
    <key>CFBundleVersion</key><string>$version</string>
    <key>LSMinimumSystemVersion</key><string>13.0</string>
    <key>LSUIElement</key><true/>
    <key>LSApplicationCategoryType</key><string>public.app-category.developer-tools</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/></dict>
    <key>NSHumanReadableCopyright</key><string>MIT License</string>
</dict>
</plist>
EOF
plutil -lint "$app/Contents/Info.plist" >/dev/null
codesign --force -s - "$app"
echo "$app"
