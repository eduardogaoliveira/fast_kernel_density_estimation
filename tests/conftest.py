"""Make the ``benchmarks`` directory importable from the test suite.

``benchmarks/`` holds verification-only code (the Numba reference) that is
deliberately excluded from the installed package, so it is not on ``sys.path``
by default.
"""

import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))
