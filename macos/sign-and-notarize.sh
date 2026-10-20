#!/bin/sh
# Sign build/app/<arch>/kvprof.app, wrap it in a DMG, notarize and staple it.
#
#   macos/sign-and-notarize.sh [arm64|x86_64]        (default: this machine's architecture)
#
#   SIGN_IDENTITY   "Developer ID Application: Name (TEAMID)". Unset or "-": ad-hoc, for trying the DMG locally;
#                   on another Mac Gatekeeper blocks it until the user allows it in System Settings.
#   Notarization (skipped when ad-hoc), one of:
#     NOTARY_PROFILE                              a keychain profile from `xcrun notarytool store-credentials`
#     NOTARY_KEY, NOTARY_KEY_ID, NOTARY_ISSUER    an App Store Connect API key: .p8 path, key id, issuer id
#
# Writes dist/kvprof-<version>-macos-<arch>.dmg.
set -eu

arch=${1:-$(uname -m)}
root=$(cd "$(dirname "$0")/.." && pwd -P)
app=$root/build/app/$arch/kvprof.app
identity=${SIGN_IDENTITY:--}
[ -d "$app" ] || { echo "no $app: run macos/build-app.sh $arch first" >&2; exit 1; }
version=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Contents/Info.plist")
dmg=$root/dist/kvprof-$version-macos-$arch.dmg

if [ "$identity" = - ]; then
    sign() { codesign --force -s - "$@"; }
else
    # hardened runtime and a secure timestamp: both required for notarization
    sign() { codesign --force --timestamp --options runtime -s "$identity" "$@"; }
fi

# Inside out: every Mach-O of the embedded Python first, then the bundle (which seals them as resources).
find "$app/Contents/Resources/runtime" -type f \( -name '*.so' -o -name '*.dylib' -o -perm -u+x \) | while read -r f; do
    if file -b "$f" | grep -q 'Mach-O'; then
        sign "$f"
    fi
done
sign "$app"
codesign --verify --deep --strict "$app"

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
cp -R "$app" "$stage/"
ln -s /Applications "$stage/Applications"
mkdir -p "$root/dist"
rm -f "$dmg"
hdiutil create -quiet -volname kvprof -srcfolder "$stage" -format UDZO "$dmg"

if [ "$identity" = - ]; then
    echo "ad-hoc signed, not notarized: $dmg"
    exit 0
fi
codesign --force --timestamp -s "$identity" "$dmg"

if [ -n "${NOTARY_PROFILE:-}" ]; then
    xcrun notarytool submit "$dmg" --wait --keychain-profile "$NOTARY_PROFILE"
elif [ -n "${NOTARY_KEY:-}" ]; then
    xcrun notarytool submit "$dmg" --wait --key "$NOTARY_KEY" --key-id "$NOTARY_KEY_ID" --issuer "$NOTARY_ISSUER"
else
    echo "signed but not notarized (set NOTARY_PROFILE, or NOTARY_KEY, NOTARY_KEY_ID and NOTARY_ISSUER): $dmg"
    exit 0
fi
xcrun stapler staple "$dmg"
spctl --assess --type open --context context:primary-signature --verbose "$dmg"
echo "$dmg"
