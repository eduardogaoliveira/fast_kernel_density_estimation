"""Numerical invariants and input validation for the Rust ``fast_kde`` extension.

The assertions here fix behaviour that must hold for well-formed input: output
shapes, finiteness, a monotonically increasing grid, PDF normalisation, the mode
lying inside the data range, and rejection of degenerate arguments.  A separate
test pins the extension against the Numba reference implementation of the same
algorithm.

Edge cases follow the specification table in the workdoc (section 5.3, S-1 to
S-7). Every rule is asserted against both the extension and the reference, so
the two cannot drift apart on degenerate input the way they did before.

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


# --- Edge cases (workdoc 5.3) -------------------------------------------------

IMPLEMENTATIONS = pytest.mark.parametrize(
    "kde",
    [pytest.param(fast_kde, id="rust"), pytest.param(numba_reference, id="numba")],
)


@IMPLEMENTATIONS
@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"]
)
def test_rejects_non_finite_data(kde, bad_value: float) -> None:
    """S-3: non-finite input is rejected instead of being silently dropped.

    The extension used to skip NaN (f64::min/max ignore it and the range
    comparison is false for it), while the reference propagated it -- so the
    same input produced a plausible density in one and NaN in the other.
    """
    data = np.append(np.random.default_rng(0).normal(0.0, 1.0, 9), bad_value)

    with pytest.raises(ValueError, match="must be finite; found 1 non-finite"):
        kde.kde_deriche(data, BINS, SIGMA)
    with pytest.raises(ValueError, match="must be finite; found 1 non-finite"):
        kde.kde_mode_deriche(data, BINS, SIGMA)


@IMPLEMENTATIONS
def test_reports_the_number_of_non_finite_values(kde) -> None:
    """S-3: the error says how many values were bad, not just that some were."""
    data = np.append(np.random.default_rng(0).normal(0.0, 1.0, 8), [np.nan, np.inf])

    with pytest.raises(ValueError, match="found 2 non-finite"):
        kde.kde_deriche(data, BINS, SIGMA)


@IMPLEMENTATIONS
@pytest.mark.parametrize("sigma", [-0.2, -1e-15], ids=["large", "tiny"])
def test_rejects_negative_sigma(kde, data: np.ndarray, sigma: float) -> None:
    """S-4: a negative bandwidth is rejected.

    It flips the sign in exp(-lambda / sigma), producing a filter that is not a
    Gaussian approximation -- previously returned as if it were a valid result.
    """
    with pytest.raises(ValueError, match="Sigma must be non-negative"):
        kde.kde_deriche(data, BINS, sigma)


@IMPLEMENTATIONS
@pytest.mark.parametrize("sigma", [0.0, 1e-15], ids=["zero", "below-threshold"])
def test_zero_sigma_returns_unsmoothed_histogram(
    kde, data: np.ndarray, sigma: float
) -> None:
    """S-5: sigma == 0 means "no smoothing", and still yields a valid PDF."""
    x, pdf = kde.kde_deriche(data, BINS, sigma)

    assert np.all(np.isfinite(pdf))
    assert np.sum(pdf) * (x[1] - x[0]) == pytest.approx(1.0, abs=1e-9)


@IMPLEMENTATIONS
@pytest.mark.parametrize("bins", [7, 256, BINS], ids=["odd", "even", "default"])
def test_constant_data_is_normalized(kde, bins: int) -> None:
    """S-6: constant input yields a degenerate but normalised PDF.

    The height used to be a fixed 1/1e-9 regardless of the grid, which left the
    integral short by a factor of `bins` (1.95e-03 at bins=512).
    """
    x, pdf = kde.kde_deriche(np.full(10, 1.0), bins, SIGMA)

    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(pdf))
    assert np.count_nonzero(pdf) == 1
    assert np.sum(pdf) * (x[1] - x[0]) == pytest.approx(1.0, abs=1e-3)
    assert kde.kde_mode_deriche(np.full(10, 1.0), bins, SIGMA) == pytest.approx(1.0)


@IMPLEMENTATIONS
@pytest.mark.parametrize("bins", [1, 2, 3], ids=["one", "two", "three"])
def test_degenerate_bins_are_normalized(kde, data: np.ndarray, bins: int) -> None:
    """S-7: very small bin counts still produce a normalised PDF."""
    _, pdf = kde.kde_deriche(data, bins, SIGMA)

    assert pdf.shape == (bins,)
    assert np.all(np.isfinite(pdf))
    dx = (data.max() - data.min()) / bins
    assert np.sum(pdf) * dx == pytest.approx(1.0, abs=1e-9)


@IMPLEMENTATIONS
def test_minimum_sample_count_is_accepted(kde) -> None:
    """S-7: exactly four samples is the documented lower bound, not an error."""
    _, pdf = kde.kde_deriche(np.array([0.0, 1.0, 2.0, 3.0]), 8, 0.5)

    assert pdf.shape == (8,)
    assert np.all(np.isfinite(pdf))


def test_validation_order_is_samples_then_bins_then_finite_then_sigma() -> None:
    """5.3: an input breaking several rules reports the first one in order."""
    with pytest.raises(ValueError, match="at least 4 samples"):
        fast_kde.kde_deriche(np.array([np.nan, 1.0, 2.0]), 0, -1.0)
    with pytest.raises(ValueError, match="bins must be greater than 0"):
        fast_kde.kde_deriche(np.array([np.nan, 1.0, 2.0, 3.0]), 0, -1.0)
    with pytest.raises(ValueError, match="must be finite"):
        fast_kde.kde_deriche(np.array([np.nan, 1.0, 2.0, 3.0]), 8, -1.0)
    with pytest.raises(ValueError, match="Sigma must be non-negative"):
        fast_kde.kde_deriche(np.array([0.0, 1.0, 2.0, 3.0]), 8, -1.0)


# --- Filter accuracy (workdoc phase 5, steps 37-38) ---------------------------


@IMPLEMENTATIONS
@pytest.mark.parametrize("sigma", [0.05, 0.2, 1.0], ids=["narrow", "default", "wide"])
def test_filter_matches_exact_gaussian_convolution(kde, sigma: float) -> None:
    """The Deriche approximation tracks a true Gaussian at any bandwidth.

    The original coefficients kept only the real part of the complex pair from
    the paper and applied the four poles as a cascade, so the filter width
    saturated near 11 bins: the error against an exact Gaussian grew with sigma
    (43% at sigma=1.0) and the estimated mode stopped moving. Pin the accuracy
    so that cannot come back unnoticed.
    """
    ndimage = pytest.importorskip("scipy.ndimage")

    data = np.random.default_rng(42).normal(0.0, 1.0, 2_000)
    bins = 256
    _, pdf = kde.kde_deriche(data, bins, sigma)

    xmin, xmax = float(data.min()), float(data.max())
    binned = numba_reference.linear_binning(data, xmin, xmax, bins)
    exact = ndimage.gaussian_filter1d(
        binned, sigma * bins / (xmax - xmin), mode="constant", truncate=12.0
    )
    exact = exact / (exact.sum() * (xmax - xmin) / bins)

    assert np.max(np.abs(pdf - exact)) / exact.max() < 1e-3


@IMPLEMENTATIONS
def test_mode_converges_to_the_true_mode_as_sigma_grows(kde) -> None:
    """More smoothing must move the estimate towards the underlying peak.

    With the saturating filter the mode sat near 0.26 no matter how large sigma
    became; the true mode of this sample is 0.
    """
    data = np.random.default_rng(42).normal(0.0, 1.0, 2_000)

    modes = [abs(kde.kde_mode_deriche(data, 256, s)) for s in (0.1, 0.3, 0.8)]

    assert modes[0] > modes[1] > modes[2]
    assert modes[-1] < 0.05


def test_agrees_with_scipy_on_a_well_defined_mode() -> None:
    """The extension and scipy.stats.gaussian_kde find the same peak.

    They are not expected to agree pointwise -- one evaluates the kernel sum
    exactly, the other smooths a binned grid -- but at a bandwidth where the
    mode is well determined they should pick the same location.
    """
    gaussian_kde = pytest.importorskip("scipy.stats").gaussian_kde

    data = np.random.default_rng(42).normal(0.0, 1.0, 2_000)
    x, pdf = fast_kde.kde_deriche(data, 256, 0.3)

    reference = gaussian_kde(data)
    reference.set_bandwidth(bw_method=0.3 / np.std(data, ddof=1))
    scipy_pdf = reference.evaluate(x)

    grid_step = x[1] - x[0]
    assert abs(x[int(np.argmax(pdf))] - x[int(np.argmax(scipy_pdf))]) <= grid_step
    assert np.max(np.abs(pdf - scipy_pdf)) / pdf.max() < 0.05
