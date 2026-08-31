"""Numba reference implementation of the Deriche-filter KDE.

This module mirrors the Rust extension in ``fast_kde/src/lib.rs`` step for
step so that the compiled extension can be checked against an independent
implementation of the *same* algorithm.  It exists purely for verification
and benchmarking; it is not part of the installed ``fast_kde`` package and
must never be imported by library code.

Public API (matching ``fast_kde``):
    kde_deriche(data, bins, sigma) -> (x_coords, pdf_values)
    kde_mode_deriche(data, bins, sigma) -> float

Reference: "Fast & Accurate Gaussian Kernel Density Estimation" — linear
binning (Section 3) followed by a K=4 Deriche recursive filter (Section 2).

Requires the ``benchmark`` extra:
    uv sync --extra benchmark
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Deriche K=4 coefficients from Heer 2021, equation (2). They are complex
# conjugate pairs; taking only the real parts does not approximate a Gaussian.
# Identical to ALPHA_RE/ALPHA_IM and LAMBDA_RE/LAMBDA_IM in fast_kde/src/lib.rs.
ALPHA = np.array(
    [0.84 + 1.8675j, 0.84 - 1.8675j, -0.34015 - 0.1299j, -0.34015 + 0.1299j]
)
LAMBDA = np.array([1.783 + 0.6318j, 1.783 - 0.6318j, 1.723 + 1.997j, 1.723 - 1.997j])


def deriche_coefficients(sigma):
    """Expand sum_k alpha_k / (1 - exp(-lambda_k / sigma) z^-1) into a rational form.

    Returns ``(b_plus, b_minus, a)``: the causal numerator, the anticausal
    numerator and the shared denominator coefficients for z^-1..z^-4. The
    conjugate pairing cancels the imaginary parts, so the result is real.
    """
    poles = np.exp(-LAMBDA / sigma)

    denominator = np.array([1.0 + 0j])
    for p in poles:
        denominator = np.convolve(denominator, [1.0, -p])

    numerator = np.zeros(4, dtype=complex)
    for k in range(4):
        partial = np.array([1.0 + 0j])
        for j in range(4):
            if j != k:
                partial = np.convolve(partial, [1.0, -poles[j]])
        numerator += ALPHA[k] * partial

    # Fold the 1 / sqrt(2 pi sigma^2) factor of equation (2) into the numerator.
    b_plus = numerator.real * (1.0 / (np.sqrt(2.0 * np.pi) * sigma))
    a = denominator.real[1:]

    # Anticausal numerator (Getreuer, "A Survey of Gaussian Convolution
    # Algorithms", IPOL 2013).
    b_minus = np.zeros(5)
    b_minus[1:4] = b_plus[1:4] - a[:3] * b_plus[0]
    b_minus[4] = -a[3] * b_plus[0]
    return b_plus, b_minus, a


@njit(cache=True, nogil=True)
def _deriche_passes(signal, b_plus, b_minus, a):
    """Run the causal and anticausal 4th-order recursions and sum them."""
    m = len(signal)
    causal = np.zeros(m, dtype=np.float64)
    anticausal = np.zeros(m, dtype=np.float64)

    for i in range(m):
        acc = 0.0
        for k in range(4):
            if i >= k:
                acc += b_plus[k] * signal[i - k]
        for k in range(4):
            if i > k:
                acc -= a[k] * causal[i - k - 1]
        causal[i] = acc

    for i in range(m - 1, -1, -1):
        acc = 0.0
        for k in range(1, 5):
            if i + k < m:
                acc += b_minus[k] * signal[i + k]
        for k in range(4):
            if i + k + 1 < m:
                acc -= a[k] * anticausal[i + k + 1]
        anticausal[i] = acc

    return causal + anticausal


@njit(cache=True, nogil=True)
def linear_binning(data, xmin, xmax, bins):
    """Distribute each sample over its two neighbouring bins by linear weight."""
    if bins == 0:
        return np.zeros(0, dtype=np.float64)

    bin_width = (xmax - xmin) / bins
    if abs(bin_width) < 1e-9:
        return np.zeros(bins, dtype=np.float64)
    if len(data) == 0:
        return np.zeros(bins, dtype=np.float64)

    hist = np.zeros(bins, dtype=np.float64)
    for i in range(len(data)):
        x_val = data[i]
        if xmin <= x_val < xmax:
            pos = (x_val - xmin) / bin_width
            k = int(pos)
            fraction_right = pos - k
            fraction_left = 1.0 - fraction_right
            if k < bins:
                hist[k] += fraction_left
                if k + 1 < bins:
                    hist[k + 1] += fraction_right
        elif abs(x_val - xmax) < 1e-9 and bins > 0:
            # A sample exactly at xmax belongs to the last bin.
            hist[bins - 1] += 1.0
    return hist


def deriche_recursive_filter(signal, sigma):
    """Approximate Gaussian smoothing with Deriche's 4th-order recursive filter.

    A causal pass runs left to right and an anticausal pass right to left; their
    sum approximates convolution with a Gaussian of standard deviation ``sigma``
    (in samples), to better than 0.05% of the peak for sigma from 1 to 50.
    """
    if abs(sigma) < 1e-9 or len(signal) == 0:
        return signal.astype(np.float64).copy()

    b_plus, b_minus, a = deriche_coefficients(sigma)
    return _deriche_passes(
        np.ascontiguousarray(signal, dtype=np.float64), b_plus, b_minus, a
    )


def kde_deriche(data, bins, sigma):
    """Kernel density estimate via linear binning plus a Deriche filter.

    Returns ``(x_coords, pdf_values)`` where ``x_coords`` are bin centres over
    ``[min(data), max(data)]`` and ``pdf_values`` integrate to one.

    Raises:
        ValueError: fewer than four samples, ``bins == 0``, an inverted data
            range, a degenerate bin width, or an inconsistent normalisation.
    """
    data = np.ascontiguousarray(data, dtype=np.float64)

    if len(data) < 4:
        raise ValueError("Need at least 4 samples for KDE.")
    if bins == 0:
        raise ValueError("Number of bins must be greater than 0.")

    xmin = float(np.min(data))
    xmax = float(np.max(data))

    # Degenerate range: every sample sits at (essentially) the same place.
    if abs(xmax - xmin) < 1e-9:
        x_coords = np.array(
            [xmin + (i + 0.5) * (xmax - xmin + 1e-9) / bins for i in range(bins)],
            dtype=np.float64,
        )
        pdf_vals = np.zeros(bins, dtype=np.float64)
        if bins > 0 and len(data) > 0:
            pdf_vals[bins // 2] = 1.0 / 1e-9
        return x_coords, pdf_vals

    if xmax < xmin:
        raise ValueError(
            f"xmax ({xmax}) must be greater than or equal to xmin ({xmin})"
        )

    hist_counts = linear_binning(data, xmin, xmax, bins)

    dx = (xmax - xmin) / bins
    if np.all(np.abs(hist_counts) < 1e-9):
        x_coords = np.array(
            [xmin + (i + 0.5) * dx for i in range(bins)], dtype=np.float64
        )
        return x_coords, np.zeros(bins, dtype=np.float64)

    # Convert sigma from data units into bin units before filtering.
    sf = bins / (xmax - xmin)
    filtered = deriche_recursive_filter(hist_counts, sigma * sf)

    if abs(dx) < 1e-9:
        raise ValueError("dx (bin width) is too small or zero.")

    normalization_factor = float(np.sum(filtered)) * dx
    if abs(normalization_factor) < 1e-9:
        if np.any(np.abs(filtered) >= 1e-9):
            raise ValueError(
                "Normalization factor is near zero despite non-zero filtered "
                "histogram sum."
            )
    else:
        filtered = filtered / normalization_factor

    x_coords = np.array([xmin + (i + 0.5) * dx for i in range(bins)], dtype=np.float64)
    return x_coords, filtered


def kde_mode_deriche(data, bins, sigma):
    """Return the x coordinate where the Deriche KDE attains its maximum."""
    x_coords, pdf_values = kde_deriche(data, bins, sigma)

    if len(pdf_values) == 0:
        raise ValueError("PDF is empty, cannot find mode.")

    finite = np.isfinite(pdf_values)
    if not finite.any():
        raise ValueError(
            "Could not find a valid mode in the PDF (e.g., all values are NaN/Inf)."
        )

    masked = np.where(finite, pdf_values, -np.inf)
    return float(x_coords[int(np.argmax(masked))])
