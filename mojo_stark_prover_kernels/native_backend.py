"""Helpers to build and load the native Rust kernel backend."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path

from .backends import MojoSharedLibraryBackend


class NativeBuildError(RuntimeError):
    pass


def repository_root() -> Path:
    return Path(__file__).resolve().parent.parent


def native_manifest_path() -> Path:
    return repository_root() / "native" / "mojo_kernel_abi" / "Cargo.toml"


def build_native_kernel(*, release: bool = True) -> Path:
    cargo = shutil.which("cargo")
    if cargo is None:
        raise NativeBuildError("cargo not found in PATH")
    cargo_path = Path(cargo).expanduser()
    if not cargo_path.is_absolute():
        cargo_path = cargo_path.resolve(strict=True)
    _validate_tool_binary(cargo_path, tool_name="cargo")

    manifest = native_manifest_path()
    if not manifest.exists():
        raise NativeBuildError(f"native manifest not found: {manifest}")

    cmd = [os.fspath(cargo_path), "build", "--manifest-path", str(manifest)]
    if release:
        cmd.append("--release")

    env = os.environ.copy()
    env = _with_target_cpu_native(env)

    # `cmd` is an allowlisted tool path plus fixed args, with shell disabled.
    proc = subprocess.run(  # nosec B603
        cmd,
        cwd=str(repository_root()),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise NativeBuildError(
            "native kernel build failed.\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}\n"
        )

    profile = "release" if release else "debug"
    return _native_artifact_path(profile)


def build_native_kernel_with_sha256(*, release: bool = True) -> tuple[Path, str]:
    artifact = build_native_kernel(release=release)
    digest = _sha256_file(artifact)
    return artifact, digest


def _native_artifact_path(profile: str) -> Path:
    target_dir = repository_root() / "native" / "mojo_kernel_abi" / "target" / profile
    if sys.platform == "darwin":
        lib_name = "libmojo_kernel_abi.dylib"
    elif sys.platform == "win32":
        lib_name = "mojo_kernel_abi.dll"
    else:
        lib_name = "libmojo_kernel_abi.so"
    lib_path = target_dir / lib_name
    if not lib_path.exists():
        raise NativeBuildError(f"native kernel artifact missing: {lib_path}")
    return lib_path


def _sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(f, "sha256").hexdigest()
        hasher = hashlib.sha256()
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
        return hasher.hexdigest()


def _validate_tool_binary(path: Path, *, tool_name: str) -> None:
    resolved = path.resolve(strict=True)
    candidates = [path]
    if resolved != path:
        candidates.append(resolved)

    for candidate in candidates:
        if not candidate.is_file():
            raise NativeBuildError(f"{tool_name} path is not a file: {candidate}")
        st = candidate.stat()
        if st.st_mode & 0o022:
            raise NativeBuildError(
                f"refusing to run group/world-writable tool binary for {tool_name}: {candidate}"
            )
        if hasattr(os, "getuid"):
            uid = os.getuid()
            if st.st_uid not in (uid, 0):
                raise NativeBuildError(
                    f"refusing to run {tool_name} not owned by current user/root: {candidate}"
                )


def _with_target_cpu_native(env: dict[str, str]) -> dict[str, str]:
    """
    Opt into host-specific codegen for local benchmarking unless explicitly disabled.

    Set `MSPK_ENABLE_TARGET_CPU_NATIVE=0` to turn this off.
    """
    enabled = env.get("MSPK_ENABLE_TARGET_CPU_NATIVE", "1")
    if enabled.strip().lower() in {"0", "false", "no"}:
        return env

    existing = env.get("RUSTFLAGS", "").strip()
    flag = "-C target-cpu=native"
    if flag in existing:
        return env
    env["RUSTFLAGS"] = f"{existing} {flag}".strip()
    return env


class NativeRustKernelBackend(MojoSharedLibraryBackend):
    name = "native-rust-kernel"

    @classmethod
    def build_and_create(
        cls,
        *,
        release: bool = True,
        debug_buffer_size: int = 4096,
    ) -> "NativeRustKernelBackend":
        artifact, artifact_sha256 = build_native_kernel_with_sha256(release=release)
        return cls(
            artifact,
            allow_relative_path=True,
            debug_buffer_size=debug_buffer_size,
            expected_sha256=artifact_sha256,
        )


class NativeRustM31Backend(MojoSharedLibraryBackend):
    name = "native-rust-m31"

    @classmethod
    def build_and_create(
        cls,
        *,
        release: bool = True,
        debug_buffer_size: int = 4096,
    ) -> "NativeRustM31Backend":
        artifact, artifact_sha256 = build_native_kernel_with_sha256(release=release)
        return cls(
            artifact,
            allow_relative_path=True,
            debug_buffer_size=debug_buffer_size,
            expected_sha256=artifact_sha256,
        )
