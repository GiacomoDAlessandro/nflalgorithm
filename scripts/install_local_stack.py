"""Materialize the gitignored private-module filenames as local-stack shims.

A fresh clone has no ``api/server.py``: it is gitignored because the production
copy is proprietary. ``make doctor`` reports that as a FAIL and
``make fullstack`` cannot import an app. This writer creates a thin shim at each
gitignored path that delegates to the tracked implementation in ``local_stack``.

The shim, not the implementation, is the module the API test suite patches
(``api.server.read_dataframe`` and friends), so ``install`` publishes its names
into the shim's own namespace.

Existing files are never overwritten without ``--force``: on a machine that has
the real private modules, clobbering them would replace production logic with a
development stand-in and nothing downstream would report the substitution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

API_SERVER_SHIM = '''"""Local-development API module written by scripts/install_local_stack.py.

Not the deployment-supplied `api/server.py`. This file is gitignored; replace it
with the production module before running production workers.
"""

PUBLIC_VALUE_VISIBILITY_CONTRACT = "publication-safe-v1"

from local_stack.api_app import install  # noqa: E402

install(globals())
'''

SHIMS: dict[str, str] = {"api/server.py": API_SERVER_SHIM}


def install_shims(root: Path = PROJECT_ROOT, *, force: bool = False) -> tuple[list[str], list[str]]:
    """Write each shim, returning (written, skipped) relative paths."""
    written: list[str] = []
    skipped: list[str] = []
    for relative, source in SHIMS.items():
        target = root / relative
        if target.exists() and not force:
            skipped.append(relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        written.append(relative)
    return written, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files, including deployment-supplied private modules.",
    )
    args = parser.parse_args(argv)

    written, skipped = install_shims(force=args.force)

    for relative in written:
        print(f"wrote {relative}")
    for relative in skipped:
        print(f"kept  {relative} (already present; pass --force to replace)")

    if written:
        print("\nLocal stack installed. Next: make doctor && make fullstack")
    return 0


if __name__ == "__main__":
    sys.exit(main())
