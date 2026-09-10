#!/usr/bin/env bash
# Install Vaani for the current user. No root required.
#
# Everything goes under ~/.local, which is on the XDG path for both the launcher
# and the desktop menu. Nothing is copied: the desktop entries and the launcher
# point back at this checkout, so `git pull` or an edit takes effect immediately
# and there is no second stale copy to confuse things later.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$HOME/.local/bin"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/scalable/apps"

echo "Installing Vaani from: $HERE"

if [[ ! -x "$HERE/.venv/bin/python" ]]; then
  echo "ERROR: virtualenv missing at $HERE/.venv" >&2
  echo "       create it first:  uv venv --python 3.14 .venv" >&2
  exit 2
fi

mkdir -p "$BIN" "$APPS" "$ICONS"

ln -sf "$HERE/vaani" "$BIN/vaani"
echo "  launcher  -> $BIN/vaani"

install -m 644 "$HERE/packaging/vaani.svg" "$ICONS/vaani.svg"
echo "  icon      -> $ICONS/vaani.svg"

# Exec points at the ~/.local/bin launcher rather than the checkout path.
# The checkout may live somewhere containing spaces or parentheses -- this one
# does ("New Folder (4)") -- and those are reserved characters in an Exec key.
# desktop-file-validate rejects them, and some launchers mis-parse the command.
TERMINAL_CMD=""
for t in konsole gnome-terminal xterm; do
  if command -v "$t" >/dev/null 2>&1; then TERMINAL_CMD="$t -e"; break; fi
done

cat > "$APPS/vaani.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Vaani
GenericName=Voice Translator
Comment=Speak Hindi or Hinglish, be heard in English in your own voice
Exec=$BIN/vaani gui
Icon=vaani
Terminal=false
Categories=Utility;
Keywords=translate;translation;voice;hindi;hinglish;meeting;speech;
StartupNotify=true
Actions=Meeting;Diagnostics;

[Desktop Action Meeting]
Name=Start Meeting Mode (hidden)
Exec=$BIN/vaani meeting

[Desktop Action Diagnostics]
Name=Run Diagnostics
Exec=${TERMINAL_CMD:-xterm -e} $BIN/vaani doctor
DESKTOP

if command -v desktop-file-validate >/dev/null 2>&1; then
  desktop-file-validate "$APPS/vaani.desktop" || {
    echo "WARNING: the desktop entry did not validate (see above)" >&2; }
fi
echo "  menu entry-> $APPS/vaani.desktop"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS" 2>/dev/null || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
  gtk-update-icon-cache -f -t "$HOME/.local/share/icons/hicolor" 2>/dev/null || true
fi

case ":$PATH:" in
  *":$BIN:"*) ;;
  *) echo
     echo "NOTE: $BIN is not on your PATH. Add this to ~/.bashrc:"
     echo "      export PATH=\"\$HOME/.local/bin:\$PATH\"" ;;
esac

echo
echo "Installed. Try:"
echo "  vaani doctor       check everything works"
echo "  vaani gui          the desktop app"
echo "  vaani meeting      background mode for calls"
echo
echo "Optional — bind meeting controls to keys (KDE):"
echo "  System Settings -> Shortcuts -> Add Command"
echo "    vaani toggle    show/hide the overlay"
echo "    vaani panic     emergency stop"
