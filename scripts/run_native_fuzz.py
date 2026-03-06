#!/usr/bin/env python3
"""Run deterministic and optional libFuzzer checks for the native ABI."""

from __future__ import annotations

import argparse
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NATIVE_CRATE = REPO_ROOT / "native" / "mojo_kernel_abi"
FUZZ_CRATE = NATIVE_CRATE / "fuzz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run native ABI fuzz checks.")
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="smoke runs deterministic regression fuzz tests; full also runs cargo-fuzz.",
    )
    parser.add_argument(
        "--fuzz-seconds",
        type=int,
        default=30,
        help="max seconds for cargo-fuzz when --mode=full",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str], *, cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)  # nosec
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def nightly_toolchain_available() -> bool:
    rustup = shutil.which("rustup")
    if rustup is None:
        return False

    proc = subprocess.run(  # nosec
        [rustup, "toolchain", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return False

    return any(line.startswith("nightly") for line in proc.stdout.splitlines())


def main() -> int:
    args = parse_args()

    # Always run deterministic fuzz regression tests in stable toolchains.
    run_cmd(
        [
            "cargo",
            "test",
            "--manifest-path",
            str(NATIVE_CRATE / "Cargo.toml"),
            "fuzz_",
        ],
        cwd=REPO_ROOT,
    )

    if args.mode == "smoke":
        print("native fuzz smoke checks passed")
        return 0

    if shutil.which("cargo-fuzz") is None:
        print(
            "ERROR cargo-fuzz not found. Install with: cargo install cargo-fuzz",
            file=sys.stderr,
        )
        return 2
    if not nightly_toolchain_available():
        print(
            "ERROR Rust nightly toolchain is required for cargo-fuzz. "
            "Install with: rustup toolchain install nightly",
            file=sys.stderr,
        )
        return 2

    try:
        run_cmd(
            [
                "cargo",
                "+nightly",
                "fuzz",
                "run",
                "abi_entrypoint",
                "--",
                f"-max_total_time={args.fuzz_seconds}",
                "-timeout=10",
            ],
            cwd=FUZZ_CRATE,
        )
    except RuntimeError as exc:
        print(f"ERROR cargo-fuzz run failed: {exc}", file=sys.stderr)
        return 3

    print("native fuzz full checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
