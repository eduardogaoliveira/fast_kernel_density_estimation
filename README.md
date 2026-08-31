# Fast & Accurate Gaussian Kernel Density Estimation (Rust + Python)

A one-dimensional Gaussian kernel density estimator implemented in Rust and exposed
to Python through PyO3. It combines **linear binning** with a **K=4 Deriche recursive
filter** to approximate Gaussian smoothing in time independent of the kernel width,
following Jeffrey Heer's *"Fast & Accurate Gaussian Kernel Density Estimation"*.

## Public API

The compiled `fast_kde` module exports two functions:

| Function | Returns |
| :--- | :--- |
| `kde_deriche(data, bins, sigma)` | `(x_coords, pdf_values)` — bin centres over `[min(data), max(data)]` and the PDF, normalised to integrate to 1 |
| `kde_mode_deriche(data, bins, sigma)` | `float` — the `x` coordinate where the PDF is maximal |

Both raise `ValueError` for fewer than four samples, `bins == 0`, a degenerate bin
width, or an inconsistent normalisation.

## Requirements

* A stable Rust toolchain — install with [rustup](https://rustup.rs), then `source $HOME/.cargo/env`
* [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python environment management

Python packaging uses the **Maturin** build backend (`[tool.maturin]` in
`pyproject.toml`), so `uv sync` compiles the Rust extension and installs it as
`fast_kde` — no separate `maturin develop` step is needed.

## Setup

```bash
uv sync
```

To rebuild the extension after changing Rust sources:

```bash
uv sync --reinstall-package fast-kernel-density-estimation
```

`uv sync` installs exactly the groups you name, so add back any extras you were
using in the same command, e.g.
`uv sync --extra benchmark --reinstall-package fast-kernel-density-estimation`.

Verify the extension is really built (importing `fast_kde` alone is not enough —
the source directory would resolve as an empty namespace package):

```bash
uv run python -c "import fast_kde; print(fast_kde.kde_deriche, fast_kde.kde_mode_deriche)"
```

### Optional dependency groups

| Group | Install | Contents |
| :--- | :--- | :--- |
| `benchmark` | `uv sync --extra benchmark` | SciPy, Numba, Matplotlib — needed for the test suite and benchmarks |
| `notebook` | `uv sync --extra notebook` | JupyterLab, ipykernel — needed for `kde_comparison.ipynb` |

Combine them when you need both: `uv sync --extra benchmark --extra notebook`.

## Usage

```python
import numpy as np
import fast_kde

data = np.random.default_rng(0).normal(0.0, 1.0, 1_000)

x, pdf = fast_kde.kde_deriche(data, 512, 0.2)
mode = fast_kde.kde_mode_deriche(data, 512, 0.2)

print(x.shape, pdf.shape)          # (512,) (512,)
print(np.trapezoid(pdf, x))        # ~1.0
print(mode)                        # location of the density peak
```

## Tests

```bash
uv run --extra benchmark pytest -q
```

The suite checks that the extension is exported at all, that output shapes,
finiteness, grid monotonicity and PDF normalisation hold, that malformed input is
rejected, and that the extension agrees with the Numba reference implementation in
`benchmarks/numba_reference.py` to `rtol=1e-9, atol=1e-12`.

## Benchmark

```bash
uv run --extra benchmark python benchmarks/benchmark_kde.py --quick --output-dir outputs/smoke
```

Compares the Rust extension, the Numba reference and `scipy.stats.gaussian_kde` on
three fixed-seed datasets, reporting wall-clock time, the PDF integral and the
estimated mode. Drop `--quick` for the full size (n=50,000, bins=1,024) and add
`--plot` to also write a timings chart. `--output-dir` is required — nothing is
written outside it.

## Repository layout

```text
fast_kde/            Rust crate (PyO3 extension module)
benchmarks/          verification-only code, not part of the installed package
  numba_reference.py Numba implementation of the same algorithm
  benchmark_kde.py   benchmark CLI
tests/               pytest suite
kde_comparison.ipynb exploratory notebook (needs the notebook extra)
```

## Generated artifacts

Benchmark output (`outputs/`), build artifacts (`dist/`, `*.so`), and Python/Numba/
Ruff/pytest caches are git-ignored and are never committed — regenerate them by
rerunning the commands above.

## Notes on comparison

`scipy.stats.gaussian_kde` appears in the benchmark for orientation only, not as a
correctness oracle. It applies its own bandwidth convention and evaluates on a grid
padded beyond the data range, so its integral and mode legitimately differ from the
binned Deriche estimates; the benchmark output has shown mode differences of well
over one unit on uniform data. Agreement is asserted only between the Rust
extension and the Numba reference, which implement the same algorithm.

## Quality gates

```bash
uv run ruff format --check benchmarks tests
uv run ruff check benchmarks tests
cargo fmt --manifest-path fast_kde/Cargo.toml -- --check
cargo clippy --manifest-path fast_kde/Cargo.toml --all-targets -- -D warnings
uv build   # produces a platform wheel containing the compiled extension
```

## References

* **Jeffrey Heer.** "Fast & Accurate Gaussian Kernel Density Estimation." IEEE VIS Short Papers, 2021.
  * [Paper](http://idl.cs.washington.edu/papers/fast-kde)
