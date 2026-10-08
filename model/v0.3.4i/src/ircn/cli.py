from __future__ import annotations

import argparse
import os
import sys

# Pin CPU math-library threads before NumPy/SciPy are imported.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_var] = "1"

from .experiment import run_e0, run_e1


def main() -> None:
    parser = argparse.ArgumentParser(prog="ircn")
    parser.add_argument("command", choices=("e0", "e1"))
    args = parser.parse_args()
    if args.command == "e0":
        ok = run_e0()
    else:
        ok = run_e1()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
