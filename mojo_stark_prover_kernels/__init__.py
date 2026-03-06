"""Public API for mojo-stark-prover-kernels."""

from .backends import (
    BackendExecutionError,
    KernelBackend,
    M31AxpyBackend,
    MojoSharedLibraryBackend,
    ReferenceKernelBackend,
)
from .contracts import CommitLayerRequest, CommitLayerResult
from .debug import DebugTracer
from .differential import (
    DifferentialCaseResult,
    DifferentialSuiteResult,
    compare_case,
    run_randomized_suite,
)
from .native_backend import (
    NativeBuildError,
    NativeRustKernelBackend,
    NativeRustM31Backend,
    build_native_kernel,
    native_manifest_path,
)
from .m31_axpy import M31AxpyRequest, M31_PRIME, m31_axpy_reference
from .runner import ProverKernelRunner

__all__ = [
    "BackendExecutionError",
    "CommitLayerRequest",
    "CommitLayerResult",
    "DebugTracer",
    "DifferentialCaseResult",
    "DifferentialSuiteResult",
    "KernelBackend",
    "M31AxpyBackend",
    "M31AxpyRequest",
    "M31_PRIME",
    "MojoSharedLibraryBackend",
    "NativeBuildError",
    "NativeRustKernelBackend",
    "NativeRustM31Backend",
    "ProverKernelRunner",
    "ReferenceKernelBackend",
    "build_native_kernel",
    "compare_case",
    "m31_axpy_reference",
    "native_manifest_path",
    "run_randomized_suite",
]
