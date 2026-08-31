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

# Deriche K=4 coefficients, identical to ALPHA_COEFFS / LAMBDA_COEFFS in
# fast_kde/src/lib.rs.
ALPHA_COEFFS = np.array([0.84, -0.34015, 0.84, -0.34015], dtype=np.float64)
LAMBDA_COEFFS = np.array([1.783, 1.723, 1.783, 1.723], dtype=np.float64)


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


@njit(cache=True, nogil=True)
def deriche_recursive_filter(signal, sigma):
    """Approximate Gaussian smoothing with four forward and four backward passes."""
    if abs(sigma) < 1e-9 or len(signal) == 0:
        return signal.copy()

    m = len(signal)
    result = signal.astype(np.float64).copy()

    poles = np.empty(4, dtype=np.float64)
    for k in range(4):
        poles[k] = ALPHA_COEFFS[k] * np.exp(-LAMBDA_COEFFS[k] / sigma)

    # Forward passes: y[i] = x[i] + a * y[i-1]
    for k_pass in range(4):
        pole = poles[k_pass]
        prev_output = 0.0
        for i in range(m):
            prev_output = result[i] + pole * prev_output
            result[i] = prev_output

    # Backward passes: y[i] = x[i] + a * y[i+1], accumulated onto the result.
    # Each pass reads the state left by the previous one.
    temp = np.empty(m, dtype=np.float64)
    for k_pass in range(4):
        pole = poles[k_pass]
        prev_output = 0.0
        for i in range(m):
            temp[i] = result[i]
        for i in range(m - 1, -1, -1):
            prev_output = temp[i] + pole * prev_output
            result[i] += prev_output
    return result


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
