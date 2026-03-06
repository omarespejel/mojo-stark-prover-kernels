"""Kernel backend abstractions and implementations."""

from __future__ import annotations

from collections import OrderedDict
import ctypes
from dataclasses import dataclass
import hashlib
import os
import stat
import sys
from pathlib import Path
from typing import Protocol

from .contracts import CommitLayerRequest, Hash32
from .debug import DebugTracer
from .m31_axpy import M31AxpyRequest, M31_PRIME
from .reference_blake2s_merkle import commit_layer as commit_layer_reference

EXPECTED_KERNEL_ABI_VERSION = 1


class BackendExecutionError(RuntimeError):
    """Raised when a backend cannot execute safely or correctly."""


class KernelBackend(Protocol):
    name: str

    def commit_layer(
        self, request: CommitLayerRequest, tracer: DebugTracer | None = None
    ) -> tuple[Hash32, ...]:
        ...


class M31AxpyBackend(Protocol):
    name: str

    def m31_axpy(
        self, request: M31AxpyRequest, tracer: DebugTracer | None = None
    ) -> tuple[int, ...]:
        ...


@dataclass
class _PreparedRequestBuffers:
    prev_layer_arr: object | None
    prev_layer_ptr: ctypes.POINTER(ctypes.c_uint8) | None
    prev_layer_len: int
    columns_arr: object
    columns_ptr: ctypes.POINTER(ctypes.c_uint32)
    n_columns: int
    n_rows: int
    out_len: int


@dataclass
class _PreparedM31RequestBuffers:
    a_arr: object
    a_ptr: ctypes.POINTER(ctypes.c_uint32)
    b_arr: object
    b_ptr: ctypes.POINTER(ctypes.c_uint32)
    c_arr: object
    c_ptr: ctypes.POINTER(ctypes.c_uint32)
    length: int
    out_len: int


class ReferenceKernelBackend:
    name = "reference-python"

    def commit_layer(
        self, request: CommitLayerRequest, tracer: DebugTracer | None = None
    ) -> tuple[Hash32, ...]:
        return commit_layer_reference(request, tracer=tracer)


class MojoSharedLibraryBackend:
    """
    Planned production backend for Mojo-compiled kernels.

    ABI (planned):
      int32 mojo_blake2s_commit_layer(
          uint32 log_size,
          uint8* prev_layer_bytes, uint32 prev_layer_len,
          uint32* columns_flat, uint32 n_columns, uint32 n_rows,
          uint8* out_hashes, uint32 out_hashes_len,
          uint32 debug_level,
          char* debug_buffer, uint32 debug_buffer_len
      )
    """

    name = "mojo-shared-lib"

    def __init__(
        self,
        shared_lib_path: Path | str,
        *,
        allow_relative_path: bool = False,
        debug_buffer_size: int = 4096,
        cache_max_entries: int = 8,
        expected_sha256: str | None = None,
    ) -> None:
        if debug_buffer_size < 128 or debug_buffer_size > 1_000_000:
            raise ValueError("debug_buffer_size must be in [128, 1_000_000]")
        if cache_max_entries < 0:
            raise ValueError("cache_max_entries cannot be negative")

        raw_path = Path(shared_lib_path)
        if not allow_relative_path and not raw_path.is_absolute():
            raise ValueError("shared_lib_path must be absolute unless allow_relative_path=True")

        expanded = raw_path.expanduser()
        if expanded.is_symlink():
            raise PermissionError(f"refusing to load symlinked shared library: {expanded}")

        resolved = expanded.resolve(strict=True)
        if not resolved.is_file():
            raise FileNotFoundError(f"shared library path is not a file: {resolved}")
        _validate_library_extension(resolved)
        _validate_file_ownership(resolved)
        _validate_parent_directory(resolved.parent)
        pinned_sha256 = (
            expected_sha256 if expected_sha256 is not None else os.environ.get("MSPK_KERNEL_SHA256")
        )
        if _env_flag_enabled("MSPK_KERNEL_STRICT") and (
            pinned_sha256 is None or not pinned_sha256.strip()
        ):
            raise BackendExecutionError(
                "MSPK_KERNEL_STRICT requires expected_sha256 or MSPK_KERNEL_SHA256 "
                "(artifact hash pinning)"
            )
        _validate_library_sha256(
            resolved,
            pinned_sha256,
        )

        st_mode = resolved.stat().st_mode
        if st_mode & 0o022:
            raise PermissionError(
                f"refusing to load group/world-writable shared library: {resolved}"
            )

        self._debug_buffer_size = debug_buffer_size
        self._cache_max_entries = cache_max_entries
        self._prepared_cache: OrderedDict[CommitLayerRequest, _PreparedRequestBuffers] = (
            OrderedDict()
        )
        self._m31_prepared_cache: OrderedDict[M31AxpyRequest, _PreparedM31RequestBuffers] = (
            OrderedDict()
        )
        self._shared_lib_path = resolved
        if os.name == "posix" and hasattr(ctypes, "RTLD_LOCAL"):
            self._lib = ctypes.CDLL(os.fspath(resolved), mode=ctypes.RTLD_LOCAL)
        else:
            self._lib = ctypes.CDLL(os.fspath(resolved))
        _validate_kernel_abi_version(self._lib)

        try:
            fn = self._lib.mojo_blake2s_commit_layer
        except AttributeError as exc:
            raise BackendExecutionError(
                "shared library missing symbol: mojo_blake2s_commit_layer"
            ) from exc

        fn.argtypes = [
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_uint32,
        ]
        fn.restype = ctypes.c_int32
        self._fn = fn

        self._m31_fn = None
        maybe_m31_fn = getattr(self._lib, "mojo_m31_axpy", None)
        if maybe_m31_fn is not None:
            maybe_m31_fn.argtypes = [
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_uint32),
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.POINTER(ctypes.c_char),
                ctypes.c_uint32,
            ]
            maybe_m31_fn.restype = ctypes.c_int32
            self._m31_fn = maybe_m31_fn
            _run_m31_kernel_self_test(maybe_m31_fn, debug_buffer_size=self._debug_buffer_size)

    @property
    def shared_lib_path(self) -> Path:
        return self._shared_lib_path

    def commit_layer(
        self, request: CommitLayerRequest, tracer: DebugTracer | None = None
    ) -> tuple[Hash32, ...]:
        request.validate()
        if tracer is not None:
            tracer.emit(1, f"mojo backend call start lib={self._shared_lib_path}")

        prepared = self._prepare_request_buffers(request)
        out_arr = (ctypes.c_uint8 * prepared.out_len)()
        out_ptr = ctypes.cast(out_arr, ctypes.POINTER(ctypes.c_uint8))

        debug_buf = (ctypes.c_char * self._debug_buffer_size)()
        debug_ptr = ctypes.cast(debug_buf, ctypes.POINTER(ctypes.c_char))

        rc = self._fn(
            ctypes.c_uint32(request.log_size),
            prepared.prev_layer_ptr,
            ctypes.c_uint32(prepared.prev_layer_len),
            prepared.columns_ptr,
            ctypes.c_uint32(prepared.n_columns),
            ctypes.c_uint32(prepared.n_rows),
            out_ptr,
            ctypes.c_uint32(prepared.out_len),
            ctypes.c_uint32(request.debug_level),
            debug_ptr,
            ctypes.c_uint32(self._debug_buffer_size),
        )

        debug_text = bytes(debug_buf).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        debug_text = _sanitize_debug_text(debug_text)
        if tracer is not None and debug_text:
            tracer.emit(2, f"mojo backend debug: {debug_text}")

        if rc != 0:
            raise BackendExecutionError(
                f"mojo kernel returned non-zero status ({rc}); details={debug_text or '<none>'}"
            )

        raw = bytes(out_arr)
        result = tuple(raw[i : i + 32] for i in range(0, len(raw), 32))

        if tracer is not None:
            tracer.emit(1, f"mojo backend call done hashes={len(result)}")
        return result

    def m31_axpy(
        self, request: M31AxpyRequest, tracer: DebugTracer | None = None
    ) -> tuple[int, ...]:
        request.validate()
        if self._m31_fn is None:
            raise BackendExecutionError("shared library missing symbol: mojo_m31_axpy")
        if tracer is not None:
            tracer.emit(1, f"mojo m31 call start lib={self._shared_lib_path}")

        prepared = self._prepare_m31_request_buffers(request)
        out_arr = (ctypes.c_uint32 * prepared.out_len)()
        out_ptr = ctypes.cast(out_arr, ctypes.POINTER(ctypes.c_uint32))

        debug_buf = (ctypes.c_char * self._debug_buffer_size)()
        debug_ptr = ctypes.cast(debug_buf, ctypes.POINTER(ctypes.c_char))

        rc = self._m31_fn(
            prepared.a_ptr,
            prepared.b_ptr,
            prepared.c_ptr,
            ctypes.c_uint32(prepared.length),
            ctypes.c_uint32(request.alpha),
            ctypes.c_uint32(request.beta),
            out_ptr,
            ctypes.c_uint32(prepared.out_len),
            ctypes.c_uint32(request.debug_level),
            debug_ptr,
            ctypes.c_uint32(self._debug_buffer_size),
        )

        debug_text = bytes(debug_buf).split(b"\0", 1)[0].decode("utf-8", errors="replace")
        debug_text = _sanitize_debug_text(debug_text)
        if tracer is not None and debug_text:
            tracer.emit(2, f"mojo m31 debug: {debug_text}")

        if rc != 0:
            raise BackendExecutionError(
                f"mojo m31 kernel returned non-zero status ({rc}); details={debug_text or '<none>'}"
            )

        # Converting directly from ctypes array is measurably faster than per-element int().
        result = tuple(out_arr)
        _validate_m31_result(result, expected_len=request.length)
        if tracer is not None:
            tracer.emit(1, f"mojo m31 call done len={len(result)}")
        return result

    def _prepare_request_buffers(self, request: CommitLayerRequest) -> _PreparedRequestBuffers:
        if self._cache_max_entries > 0:
            cached = self._prepared_cache.get(request)
            if cached is not None:
                self._prepared_cache.move_to_end(request)
                return cached

        prev_layer_arr = None
        prev_layer_ptr = None
        prev_layer_len = 0
        if request.prev_layer_hashes is not None:
            prev_raw = b"".join(request.prev_layer_hashes)
            prev_layer_len = len(prev_raw)
            prev_layer_arr = (ctypes.c_uint8 * prev_layer_len).from_buffer_copy(prev_raw)
            prev_layer_ptr = ctypes.cast(prev_layer_arr, ctypes.POINTER(ctypes.c_uint8))

        n_columns = len(request.columns)
        n_rows = request.n_rows
        columns_flat = [value for col in request.columns for value in col]
        columns_arr = (ctypes.c_uint32 * len(columns_flat))(*columns_flat)
        columns_ptr = ctypes.cast(columns_arr, ctypes.POINTER(ctypes.c_uint32))

        prepared = _PreparedRequestBuffers(
            prev_layer_arr=prev_layer_arr,
            prev_layer_ptr=prev_layer_ptr,
            prev_layer_len=prev_layer_len,
            columns_arr=columns_arr,
            columns_ptr=columns_ptr,
            n_columns=n_columns,
            n_rows=n_rows,
            out_len=n_rows * 32,
        )

        if self._cache_max_entries > 0:
            self._prepared_cache[request] = prepared
            self._prepared_cache.move_to_end(request)
            while len(self._prepared_cache) > self._cache_max_entries:
                self._prepared_cache.popitem(last=False)

        return prepared

    def _prepare_m31_request_buffers(self, request: M31AxpyRequest) -> _PreparedM31RequestBuffers:
        if self._cache_max_entries > 0:
            cached = self._m31_prepared_cache.get(request)
            if cached is not None:
                self._m31_prepared_cache.move_to_end(request)
                return cached

        a_arr = (ctypes.c_uint32 * request.length)(*request.a)
        b_arr = (ctypes.c_uint32 * request.length)(*request.b)
        c_arr = (ctypes.c_uint32 * request.length)(*request.c)

        prepared = _PreparedM31RequestBuffers(
            a_arr=a_arr,
            a_ptr=ctypes.cast(a_arr, ctypes.POINTER(ctypes.c_uint32)),
            b_arr=b_arr,
            b_ptr=ctypes.cast(b_arr, ctypes.POINTER(ctypes.c_uint32)),
            c_arr=c_arr,
            c_ptr=ctypes.cast(c_arr, ctypes.POINTER(ctypes.c_uint32)),
            length=request.length,
            out_len=request.length,
        )

        if self._cache_max_entries > 0:
            self._m31_prepared_cache[request] = prepared
            self._m31_prepared_cache.move_to_end(request)
            while len(self._m31_prepared_cache) > self._cache_max_entries:
                self._m31_prepared_cache.popitem(last=False)

        return prepared


def _sanitize_debug_text(text: str) -> str:
    sanitized = text.replace("\n", "\\n").replace("\r", "\\r")
    sanitized = "".join(ch for ch in sanitized if ch.isprintable())
    if len(sanitized) > 512:
        sanitized = sanitized[:512] + "...<truncated>"
    return sanitized


def _read_kernel_abi_version(lib: object) -> int:
    version_fn = getattr(lib, "mojo_kernel_abi_version", None)
    if version_fn is None:
        raise BackendExecutionError("shared library missing symbol: mojo_kernel_abi_version")

    # ctypes function pointers support explicit signature assignment;
    # plain Python callables used in tests do not.
    if hasattr(version_fn, "restype"):
        version_fn.restype = ctypes.c_uint32
    if hasattr(version_fn, "argtypes"):
        version_fn.argtypes = []

    try:
        version = int(version_fn())
    except Exception as exc:
        raise BackendExecutionError(
            f"failed to read kernel ABI version from shared library: {exc}"
        ) from exc
    return version


def _validate_kernel_abi_version(lib: object) -> None:
    version = _read_kernel_abi_version(lib)
    if version != EXPECTED_KERNEL_ABI_VERSION:
        raise BackendExecutionError(
            "shared library kernel ABI version mismatch: "
            f"expected {EXPECTED_KERNEL_ABI_VERSION}, got {version}"
        )


def _normalize_sha256_hex(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64:
        raise BackendExecutionError(
            f"expected kernel sha256 must be 64 hex chars, got length {len(normalized)}"
        )
    if not all(ch in "0123456789abcdef" for ch in normalized):
        raise BackendExecutionError("expected kernel sha256 must be hexadecimal")
    return normalized


def _sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        if hasattr(hashlib, "file_digest"):
            return hashlib.file_digest(f, "sha256").hexdigest()
        hasher = hashlib.sha256()
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
        return hasher.hexdigest()


def _validate_library_sha256(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None or not expected_sha256.strip():
        return
    expected = _normalize_sha256_hex(expected_sha256)
    actual = _sha256_file(path)
    if actual != expected:
        raise BackendExecutionError(
            f"shared library sha256 mismatch: expected {expected}, got {actual}"
        )


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _validate_m31_result(result: tuple[int, ...], *, expected_len: int) -> None:
    if len(result) != expected_len:
        raise BackendExecutionError(
            f"mojo m31 kernel output length mismatch: expected {expected_len}, got {len(result)}"
        )
    for idx, value in enumerate(result):
        if value < 0 or value >= M31_PRIME:
            raise BackendExecutionError(
                f"mojo m31 kernel returned non-canonical field value at index {idx}: {value}"
            )


def _run_m31_kernel_self_test(kernel_fn: object, *, debug_buffer_size: int) -> None:
    a_vals = (1, 2, 3, 4, 5, 6, 7, 8)
    b_vals = (11, 13, 17, 19, 23, 29, 31, 37)
    c_vals = (41, 43, 47, 53, 59, 61, 67, 71)
    alpha = 73
    beta = 79
    length = len(a_vals)

    a = (ctypes.c_uint32 * length)(*a_vals)
    b = (ctypes.c_uint32 * length)(*b_vals)
    c = (ctypes.c_uint32 * length)(*c_vals)
    out = (ctypes.c_uint32 * length)()
    debug_len = max(128, min(debug_buffer_size, 4096))
    debug_buf = (ctypes.c_char * debug_len)()

    rc = kernel_fn(
        ctypes.cast(a, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(b, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.cast(c, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(length),
        ctypes.c_uint32(alpha),
        ctypes.c_uint32(beta),
        ctypes.cast(out, ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_uint32(length),
        ctypes.c_uint32(0),
        ctypes.cast(debug_buf, ctypes.POINTER(ctypes.c_char)),
        ctypes.c_uint32(debug_len),
    )

    debug_text = bytes(debug_buf).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    debug_text = _sanitize_debug_text(debug_text)
    if rc != 0:
        raise BackendExecutionError(
            "mojo m31 kernel self-test call failed "
            f"(rc={rc}); details={debug_text or '<none>'}"
        )

    result = tuple(int(v) for v in out)
    _validate_m31_result(result, expected_len=length)
    expected = tuple(
        (alpha * av + beta * bv + cv) % M31_PRIME
        for av, bv, cv in zip(a_vals, b_vals, c_vals)
    )
    if result != expected:
        raise BackendExecutionError(
            "mojo m31 kernel self-test mismatch: "
            f"got={result[:4]}..., expected={expected[:4]}..."
        )


def _validate_library_extension(path: Path) -> None:
    if sys.platform == "darwin":
        expected_suffixes = {".dylib", ".so"}
    elif sys.platform == "win32":
        expected_suffixes = {".dll"}
    else:
        expected_suffixes = {".so"}
    if path.suffix.lower() not in expected_suffixes:
        raise PermissionError(
            f"unexpected shared library extension for platform: {path.name}"
        )


def _validate_file_ownership(path: Path) -> None:
    if not hasattr(os, "getuid"):
        return
    st = path.stat()
    uid = os.getuid()
    # Allow current user or root-owned read-only artifacts.
    if st.st_uid not in (uid, 0):
        raise PermissionError(
            f"refusing to load shared library not owned by current user/root: {path}"
        )


def _validate_parent_directory(parent: Path) -> None:
    current = parent
    while True:
        _validate_directory_component(current)
        if current == current.parent:
            break
        current = current.parent


def _validate_directory_component(path: Path) -> None:
    if path.is_symlink():
        raise PermissionError(f"refusing to load from symlinked directory: {path}")

    st = path.stat()
    if hasattr(os, "getuid"):
        uid = os.getuid()
        if st.st_uid not in (uid, 0):
            raise PermissionError(
                "refusing to load shared library from directory not owned by current user/root: "
                f"{path}"
            )

    writable = bool(st.st_mode & 0o022)
    if not writable:
        return

    # Allow root-owned sticky directories (for example `/tmp`) in the ancestor chain.
    sticky = bool(st.st_mode & stat.S_ISVTX)
    if sticky and st.st_uid == 0:
        return

    raise PermissionError(
        f"refusing to load shared library from group/world-writable directory: {path}"
    )
