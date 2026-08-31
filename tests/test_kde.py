"""Numerical invariants and input validation for the Rust ``fast_kde`` extension.

The assertions here fix behaviour that must hold for well-formed input: output
shapes, finiteness, a monotonically increasing grid, PDF normalisation, the mode
lying inside the data range, and rejection of degenerate arguments.  A separate
test pins the extension against the Numba reference implementation of the same
algorithm.

``scipy.stats.gaussian_kde`` is intentionally *not* used as an oracle: it applies
a different bandwidth convention on a padded grid, so disagreement with SciPy is
expected and is a benchmark observation rather than a test failure.

``numba_reference`` is imported unconditionally rather than through
``importorskip``: a silently skipped comparison would hide a drift between the
two implementations. Run the suite with the ``benchmark`` extra::

    uv run --extra benchmark pytest -q
"""

import numba_reference
import numpy as np
import pytest

import fast_kde

BINS = 512
SIGMA = 0.2


@pytest.fixture
def data() -> np.ndarray:
    """A deterministic, well-conditioned sample (see workdoc section 4.3)."""
    return np.random.default_rng(0).normal(0.0, 1.0, 1_000)


def test_kde_shape_and_normalization(data: np.ndarray) -> None:
    """The grid and PDF are well-formed and the PDF integrates to one."""
    x, pdf = fast_kde.kde_deriche(data, BINS, SIGMA)

    assert x.shape == (BINS,)
    assert pdf.shape == (BINS,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(pdf))
    assert np.all(np.diff(x) > 0), "grid coordinates must increase strictly"
    assert np.trapezoid(pdf, x) == pytest.approx(1.0, abs=1e-3)


def test_mode_is_inside_data_range(data: np.ndarray) -> None:
    """The estimated mode never falls outside the observed data range."""
    mode = fast_kde.kde_mode_deriche(data, BINS, SIGMA)

    assert np.isfinite(mode)
    assert data.min() <= mode <= data.max()


def test_rejects_too_few_samples() -> None:
    """Fewer than four samples is rejected rather than silently handled."""
    with pytest.raises(ValueError, match="at least 4 samples"):
        fast_kde.kde_deriche(np.array([0.0, 1.0, 2.0]), BINS, SIGMA)


def test_rejects_zero_bins(data: np.ndarray) -> None:
    """A zero bin count is rejected rather than producing an empty result."""
    with pytest.raises(ValueError, match="bins must be greater than 0"):
        fast_kde.kde_deriche(data, 0, SIGMA)


def test_rust_matches_numba_reference(data: np.ndarray) -> None:
    """The extension agrees with the independent Numba implementation.

    Both implement the same linear-binning plus Deriche-filter algorithm, so
    they must agree to near machine precision; a wider gap means one of them
    has drifted.
    """
    x_rust, pdf_rust = fast_kde.kde_deriche(data, BINS, SIGMA)
    x_ref, pdf_ref = numba_reference.kde_deriche(data, BINS, SIGMA)

    np.testing.assert_allclose(x_rust, x_ref, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(pdf_rust, pdf_ref, rtol=1e-9, atol=1e-12)

    mode_rust = fast_kde.kde_mode_deriche(data, BINS, SIGMA)
    mode_ref = numba_reference.kde_mode_deriche(data, BINS, SIGMA)
    assert mode_rust == pytest.approx(mode_ref, rel=1e-9, abs=1e-12)
