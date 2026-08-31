"""Compare the Rust ``fast_kde`` extension against reference KDE implementations.

Runs the same datasets through the Rust extension, the Numba reference in
:mod:`numba_reference`, and ``scipy.stats.gaussian_kde``, reporting wall-clock
time, the PDF integral and the estimated mode for each.

SciPy is included for orientation only, not as an oracle: it uses its own
bandwidth convention and evaluates on a padded grid, so its mode and integral
are not expected to match the binned Deriche estimates.

Usage:
    uv run --extra benchmark python benchmarks/benchmark_kde.py --quick \\
        --output-dir outputs/smoke
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Datasets are fixed by seed so that repeated runs are comparable.
DATASETS: dict[str, tuple[str, int]] = {
    "single_gaussian": ("normal(0, 1)", 42),
    "multimodal": ("mixture of three normals", 123),
    "uniform": ("uniform(-5, 5)", 456),
}
QUICK_SIZE = 2_000
FULL_SIZE = 50_000
QUICK_BINS = 256
FULL_BINS = 1_024
SIGMA = 0.2
REPEATS = 3


def build_dataset(name: str, size: int) -> np.ndarray:
    """Return a deterministic sample of ``size`` points for ``name``."""
    _, seed = DATASETS[name]
    rng = np.random.default_rng(seed)
    if name == "single_gaussian":
        return rng.normal(0.0, 1.0, size)
    if name == "multimodal":
        parts = [
            rng.normal(-2.0, 0.3, size // 3),
            rng.normal(0.5, 0.7, size - 2 * (size // 3)),
            rng.normal(3.0, 0.2, size // 3),
        ]
        data = np.concatenate(parts)
        rng.shuffle(data)
        return data
    if name == "uniform":
        return rng.uniform(-5.0, 5.0, size)
    raise ValueError(f"Unknown dataset: {name}")


def _time_it(func, repeats: int):
    """Run ``func`` ``repeats`` times after one warm-up; return (best_seconds, result)."""
    result = func()  # warm-up, also triggers Numba JIT compilation
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        best = min(best, time.perf_counter() - start)
    return best, result


def _method_rust(data, bins, sigma):
    import fast_kde

    x, pdf = fast_kde.kde_deriche(data, bins, sigma)
    return x, pdf, fast_kde.kde_mode_deriche(data, bins, sigma)


def _method_numba(data, bins, sigma):
    import numba_reference

    x, pdf = numba_reference.kde_deriche(data, bins, sigma)
    return x, pdf, numba_reference.kde_mode_deriche(data, bins, sigma)


def _method_scipy(data, bins, sigma):
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(data)
    kde.set_bandwidth(bw_method=sigma / np.std(data, ddof=1))
    x = np.linspace(np.min(data) - 3 * sigma, np.max(data) + 3 * sigma, bins)
    pdf = kde.evaluate(x)
    return x, pdf, float(x[int(np.argmax(pdf))])


METHODS = {"rust": _method_rust, "numba": _method_numba, "scipy": _method_scipy}


def run(datasets, size: int, bins: int, sigma: float, repeats: int) -> list[dict]:
    """Benchmark every method on every dataset; raise if any method fails."""
    rows: list[dict] = []
    for name in datasets:
        data = build_dataset(name, size)
        for method, func in METHODS.items():
            seconds, (x, pdf, mode) = _time_it(
                lambda f=func, d=data: f(d, bins, sigma), repeats
            )
            rows.append(
                {
                    "dataset": name,
                    "method": method,
                    "n": len(data),
                    "bins": bins,
                    "seconds": seconds,
                    "integral": float(np.trapezoid(pdf, x)),
                    "mode": float(mode),
                }
            )
    return rows


def format_report(rows, size, bins, sigma, repeats) -> str:
    """Render the benchmark rows as a fixed-width text report."""
    lines = [
        "fast_kde benchmark",
        f"python  : {platform.python_version()} on {platform.platform()}",
        f"settings: n={size} bins={bins} sigma={sigma} best-of={repeats}",
        "",
        f"{'dataset':<18}{'method':<8}{'time [ms]':>12}{'integral':>14}{'mode':>14}",
        "-" * 66,
    ]
    for row in rows:
        lines.append(
            f"{row['dataset']:<18}{row['method']:<8}"
            f"{row['seconds'] * 1e3:>12.3f}{row['integral']:>14.6f}{row['mode']:>14.6f}"
        )
    lines += [
        "",
        "SciPy uses a padded evaluation grid and its own bandwidth convention;",
        "its integral and mode are reference points, not expected matches.",
    ]
    return "\n".join(lines)


def write_plot(rows, path: Path) -> None:
    """Save a grouped bar chart of per-method timings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted({row["dataset"] for row in rows})
    methods = list(METHODS)
    width = 0.8 / len(methods)
    positions = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for index, method in enumerate(methods):
        times = [
            next(
                r["seconds"] * 1e3
                for r in rows
                if r["dataset"] == dataset and r["method"] == method
            )
            for dataset in datasets
        ]
        ax.bar(positions + index * width, times, width, label=method)
    ax.set_xticks(positions + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("time [ms] (best of run)")
    ax.set_yscale("log")
    ax.set_title("KDE runtime by method")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"use n={QUICK_SIZE}, bins={QUICK_BINS} instead of n={FULL_SIZE}, bins={FULL_BINS}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory the report (and plot) are written to; created if missing",
    )
    parser.add_argument(
        "--plot", action="store_true", help="also write timings.png to --output-dir"
    )
    parser.add_argument(
        "--sigma", type=float, default=SIGMA, help=f"kernel bandwidth (default {SIGMA})"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=REPEATS,
        help=f"timed runs per method after warm-up (default {REPEATS})",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    missing = []
    try:
        import fast_kde  # noqa: F401
    except ImportError:
        missing.append(
            "fast_kde (run: uv sync --reinstall-package fast-kernel-density-estimation)"
        )
    for module, hint in (
        ("numba", "--extra benchmark"),
        ("scipy", "--extra benchmark"),
    ):
        try:
            __import__(module)
        except ImportError:
            missing.append(f"{module} (run: uv sync {hint})")
    if missing:
        for item in missing:
            print(f"error: missing dependency: {item}", file=sys.stderr)
        return 1

    size = QUICK_SIZE if args.quick else FULL_SIZE
    bins = QUICK_BINS if args.quick else FULL_BINS

    rows = run(DATASETS, size, bins, args.sigma, args.repeats)

    non_finite = [
        row
        for row in rows
        if not (np.isfinite(row["integral"]) and np.isfinite(row["mode"]))
    ]
    if non_finite:
        for row in non_finite:
            print(
                f"error: non-finite result for {row['method']} on {row['dataset']}",
                file=sys.stderr,
            )
        return 1

    report = format_report(rows, size, bins, args.sigma, args.repeats)
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "benchmark_report.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nwrote {report_path}")

    if args.plot:
        plot_path = args.output_dir / "timings.png"
        write_plot(rows, plot_path)
        print(f"wrote {plot_path}")

    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
