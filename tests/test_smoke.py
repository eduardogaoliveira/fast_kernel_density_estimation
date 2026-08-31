"""Packaging smoke tests for the compiled ``fast_kde`` extension.

These tests fail when the Rust extension is not actually built and installed.
Importing ``fast_kde`` alone is not sufficient: with the source checkout on
``sys.path`` the ``fast_kde/`` directory resolves as an implicit namespace
package, so ``import fast_kde`` succeeds while exporting nothing.  The public
functions are therefore asserted explicitly.
"""

import fast_kde


def test_extension_exports_public_functions() -> None:
    """The built extension must expose both public KDE entry points."""
    missing = [
        name
        for name in ("kde_deriche", "kde_mode_deriche")
        if not callable(getattr(fast_kde, name, None))
    ]
    assert not missing, (
        f"fast_kde does not export {missing}; "
        f"__file__={fast_kde.__file__!r} — the Rust extension is probably not "
        "built (a namespace package was imported instead). "
        "Run: uv sync --reinstall-package fast-kernel-density-estimation"
    )
