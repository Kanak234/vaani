"""Entry point for ``python -m vaani.ui``."""

import os


def main() -> int:
    if os.name == "nt":
        from .windows_console import main as _main
    else:
        from .app import main as _main
    return _main()


raise SystemExit(main())
