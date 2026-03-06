"""Production-style kernel execution runner with fallback strategy."""

from __future__ import annotations

import time

from .backends import BackendExecutionError, KernelBackend
from .contracts import CommitLayerRequest, CommitLayerResult
from .debug import DebugTracer


class ProverKernelRunner:
    def __init__(
        self,
        primary_backend: KernelBackend,
        fallback_backend: KernelBackend | None = None,
        *,
        default_debug_level: int = 1,
        debug_max_events: int = 20000,
    ) -> None:
        self._primary = primary_backend
        self._fallback = fallback_backend
        self._default_debug_level = default_debug_level
        self._debug_max_events = debug_max_events

    def commit_layer(self, request: CommitLayerRequest) -> CommitLayerResult:
        request.validate()
        tracer = DebugTracer(
            level=max(request.debug_level, self._default_debug_level),
            max_events=self._debug_max_events,
        )

        started = time.perf_counter()
        tracer.emit(1, f"runner start primary={self._primary.name}")
        try:
            hashes = self._primary.commit_layer(request, tracer=tracer)
            duration_ms = (time.perf_counter() - started) * 1000.0
            tracer.emit(1, f"runner done primary={self._primary.name}")
            return CommitLayerResult(
                layer_hashes=hashes,
                backend_name=self._primary.name,
                duration_ms=duration_ms,
                debug_events=tracer.snapshot(),
            )
        except Exception as primary_exc:
            tracer.emit(1, f"runner primary failure backend={self._primary.name} err={primary_exc}")
            if self._fallback is None:
                duration_ms = (time.perf_counter() - started) * 1000.0
                raise BackendExecutionError(
                    f"primary backend failed ({self._primary.name}); no fallback configured; "
                    f"duration_ms={duration_ms:.3f}"
                ) from primary_exc

        fallback_started = time.perf_counter()
        tracer.emit(1, f"runner fallback start backend={self._fallback.name}")
        try:
            hashes = self._fallback.commit_layer(request, tracer=tracer)
            duration_ms = (time.perf_counter() - started) * 1000.0
            fallback_duration_ms = (time.perf_counter() - fallback_started) * 1000.0
            tracer.emit(
                1,
                f"runner fallback done backend={self._fallback.name} "
                f"fallback_duration_ms={fallback_duration_ms:.3f}",
            )
            return CommitLayerResult(
                layer_hashes=hashes,
                backend_name=self._fallback.name,
                duration_ms=duration_ms,
                debug_events=tracer.snapshot(),
            )
        except Exception as fallback_exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            raise BackendExecutionError(
                f"both primary and fallback failed: primary={self._primary.name}, "
                f"fallback={self._fallback.name}, duration_ms={duration_ms:.3f}"
            ) from fallback_exc

