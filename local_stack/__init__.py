"""Tracked local-development stand-ins for deployment-supplied private modules.

A fresh clone has no ``api/server.py`` (and no other gitignored execution
module), so ``make doctor`` fails and ``make fullstack`` cannot import an app.
The modules here implement those interfaces from tracked code only, and
``scripts/install_local_stack.py`` materializes the gitignored filenames as thin
shims that delegate here.

These are development stand-ins, not the production implementations. A
deployment that supplies the real private modules must not run the installer:
the generated shims would overwrite them.
"""
