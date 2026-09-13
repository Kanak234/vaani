#!/usr/bin/env bash
set -euo pipefail

echo "=== Vaani Debian Build Script ==="

HERE="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$HERE/build_debian"
RELEASE_DIR="$HERE/release/debian"
VERSION="1.0.1"
PKG_NAME="vaani_${VERSION}_amd64"

# 1. Check tools
if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "Error: dpkg-deb not found. Cannot build Debian package."
    exit 1
fi

# 2. Staging dir
echo "Creating staging directory..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/$PKG_NAME"

# 4. DEBIAN dir
mkdir -p "$BUILD_DIR/$PKG_NAME/DEBIAN"
cp "$HERE/packaging/debian/control" "$BUILD_DIR/$PKG_NAME/DEBIAN/"
cp "$HERE/packaging/debian/postinst" "$BUILD_DIR/$PKG_NAME/DEBIAN/"
cp "$HERE/packaging/debian/prerm" "$BUILD_DIR/$PKG_NAME/DEBIAN/"

# 3. Source files
echo "Copying source files..."
mkdir -p "$BUILD_DIR/$PKG_NAME/opt/vaani"
cp -r "$HERE/src" "$BUILD_DIR/$PKG_NAME/opt/vaani/"
cp "$HERE/pyproject.toml" "$BUILD_DIR/$PKG_NAME/opt/vaani/"
cp "$HERE/LICENSE" "$BUILD_DIR/$PKG_NAME/opt/vaani/" 2>/dev/null || true
find "$BUILD_DIR/$PKG_NAME/opt/vaani" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Desktop & Icon
mkdir -p "$BUILD_DIR/$PKG_NAME/usr/share/applications"
mkdir -p "$BUILD_DIR/$PKG_NAME/usr/share/icons/hicolor/scalable/apps"
cp "$HERE/packaging/debian/vaani.desktop" "$BUILD_DIR/$PKG_NAME/usr/share/applications/"
cp "$HERE/packaging/vaani.svg" "$BUILD_DIR/$PKG_NAME/usr/share/icons/hicolor/scalable/apps/" 2>/dev/null || true

# 5. Permissions
echo "Setting permissions..."
chmod 755 "$BUILD_DIR/$PKG_NAME/DEBIAN"
chmod 755 "$BUILD_DIR/$PKG_NAME/DEBIAN/postinst"
chmod 755 "$BUILD_DIR/$PKG_NAME/DEBIAN/prerm"

# 6. Build
echo "Building package..."
mkdir -p "$RELEASE_DIR"
dpkg-deb --build "$BUILD_DIR/$PKG_NAME" "$RELEASE_DIR"

# 7 & 8. Checksum
echo "Generating checksums..."
DEB_FILE="$RELEASE_DIR/${PKG_NAME}.deb"
if [[ -f "$DEB_FILE" ]]; then
    sha256sum "$DEB_FILE" > "${DEB_FILE}.sha256"
fi

echo "Build complete!"
