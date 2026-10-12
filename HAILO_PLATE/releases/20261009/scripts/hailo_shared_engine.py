"""Several InferModels on one Hailo USB VDevice, without CPU neural fallback.

Construction verifies every HEF before importing HailoRT or touching hardware.
Explicit open() acquires the single USB device. run() is serialized, returns
copied tensors and an audit event only after synchronous USB inference succeeds.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    key: str
    stage: str
    hef_path: Path
    hef_sha256: str
    input_shape: tuple
    input_dtype: str
    output_shapes: dict
    profile_hw: tuple
    input_name: str | None = None


def load_hailort():
    """Import only; ORT is DLL bootstrap, and no ORT session is constructed."""
    dll = None
    bootstrap = {}
    try:
        if os.name == "nt":
            import onnxruntime
            bootstrap["windows_dll_bootstrap"] = f"ORT {onnxruntime.__version__} imported before HailoRT DLL search path; zero ORT sessions"
            dll_path = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "HailoRT/bin"
            dll = os.add_dll_directory(str(dll_path))
        import hailo_platform as hp
        bootstrap["hailort"] = getattr(hp, "__version__", "unavailable")
        return hp, dll, bootstrap
    except BaseException:
        if dll is not None:
            dll.close()
        raise


class HailoSharedEngine:
    def __init__(self, specs, runtime_loader=load_hailort):
        specs = list(specs)
        self.specs = {spec.key: spec for spec in specs}
        if not self.specs or len(self.specs) != len(specs):
            raise ValueError("Expected a nonempty list of uniquely named Hailo models")
        self._contents = {}
        for spec in specs:
            if spec.input_dtype not in {"uint8", "float32"}:
                raise ValueError(f"Unsupported explicit host input format: {spec.input_dtype}")
            if not spec.input_shape or any(int(n) < 1 for n in spec.input_shape) or not spec.output_shapes:
                raise ValueError(f"Missing fixed stream shapes for {spec.key}")
            data = Path(spec.hef_path).read_bytes()
            if hashlib.sha256(data).hexdigest() != spec.hef_sha256:
                raise ValueError(f"HEF SHA-256 mismatch: {spec.key}")
            self._contents[spec.key] = data
        self._loader = runtime_loader
        self._lock = threading.RLock()
        self._entries = {}
        self._device = self._dll = self._hp = None
        self._closed = self._faulted = False
        self.creation_pid = os.getpid()
        self.engine_id = str(uuid.uuid4())
        self.run_sequence = 0
        self.device_id = None
        self.last_failure = None
        self.versions = {}

    def open(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Shared Hailo engine is closed")
            if self._device is not None:
                return self
            try:
                hp, self._dll, self.versions = self._loader()
                self._hp = hp
                ids = [str(value) for value in hp.Device.scan() if str(value).startswith("usb/")]
                if len(ids) != 1:
                    raise RuntimeError(f"Expected exactly one USB accelerator, found {ids}")
                params = hp.VDevice.create_params()
                params.device_ids = ids
                params.device_count = 1
                params.scheduling_algorithm = hp.HailoSchedulingAlgorithm.ROUND_ROBIN
                params.multi_process_service = False
                # Unique is the installed SDK default; no sharing with other processes.
                params.group_id = "UNIQUE"
                self._device = hp.VDevice(params=params)
                actual_ids = [str(value) for value in self._device.get_physical_devices_ids()]
                if actual_ids != ids:
                    raise RuntimeError(f"Unexpected physical device mapping: {actual_ids}")
                self.device_id = ids[0]
                for key, spec in self.specs.items():
                    entry = {"model": None, "configured": None, "bindings": None, "outputs": {}}
                    self._entries[key] = entry
                    model = entry["model"] = self._device.create_infer_model(self._contents[key])
                    model.set_batch_size(1)
                    if len(model.input_names) != 1 or set(model.output_names) != set(spec.output_shapes):
                        raise ValueError(f"HEF stream names differ from manifest: {key}")
                    input_stream = model.input()
                    if (tuple(input_stream.shape) != spec.input_shape
                            or spec.input_name is not None and input_stream.name != spec.input_name):
                        raise ValueError(f"HEF input shape/name mismatch: {key}: {input_stream.name} {input_stream.shape}")
                    input_stream.set_format_type(hp.FormatType.UINT8 if spec.input_dtype == "uint8" else hp.FormatType.FLOAT32)
                    for stream in model.outputs:
                        if tuple(stream.shape) != tuple(spec.output_shapes[stream.name]):
                            raise ValueError(f"HEF output shape differs from manifest: {key}/{stream.name}: {stream.shape}")
                        stream.set_format_type(hp.FormatType.FLOAT32)
                        entry["outputs"][stream.name] = np.empty(tuple(stream.shape), np.float32)
                    configured = entry["configured"] = model.configure()
                    configured.__enter__()
                    # Scheduler defaults are threshold=1, timeout=0ms; do not
                    # manually activate models while the scheduler is enabled.
                    entry["bindings"] = configured.create_bindings(output_buffers=entry["outputs"])
                return self
            except BaseException:
                try:
                    self.close()
                except Exception:
                    pass
                raise

    def run(self, key, tensor, timeout_ms=30000):
        with self._lock:
            if os.getpid() != self.creation_pid:
                raise RuntimeError("Hailo InferModels cannot cross process boundaries")
            if self._closed or self._device is None or self._faulted:
                raise RuntimeError("Shared Hailo engine is not ready, is closed, or a previous inference failed")
            if key not in self.specs:
                raise ValueError(f"Uncompiled Hailo model/profile: {key}")
            spec, entry = self.specs[key], self._entries[key]
            if (not isinstance(tensor, np.ndarray) or tuple(tensor.shape) != spec.input_shape
                    or tensor.dtype != np.dtype(spec.input_dtype) or not tensor.flags.c_contiguous):
                raise ValueError(f"{key} requires contiguous {spec.input_dtype} shape {spec.input_shape}; no resize/cast fallback")
            if not np.isfinite(tensor).all():
                raise ValueError("Nonfinite input")
            if not isinstance(timeout_ms, int) or timeout_ms <= 0:
                raise ValueError("timeout_ms must be positive")
            self.run_sequence += 1
            event = {"stage": spec.stage, "model_key": key, "profile_hw": list(spec.profile_hw),
                     "hef_sha256": spec.hef_sha256, "device": self.device_id,
                     "shared_engine_id": self.engine_id, "run_seq": self.run_sequence,
                     "input_shape": list(tensor.shape), "host_input_dtype": str(tensor.dtype),
                     "output_shapes": {name: list(shape) for name, shape in spec.output_shapes.items()},
                     "backend": "HAILO_USB", "cpu_neural_fallback": False}
            started_ns = time.perf_counter_ns()
            try:
                entry["bindings"].input().set_buffer(tensor)
                entry["configured"].run([entry["bindings"]], timeout=timeout_ms)
                finished_ns = time.perf_counter_ns()
                outputs = {name: value.copy() for name, value in entry["outputs"].items()}
                if not all(np.isfinite(value).all() for value in outputs.values()):
                    raise ValueError("Nonfinite USB model output")
            except BaseException as exc:
                self._faulted = True
                self.last_failure = {**event, "status": "FAILED", "completion_not_confirmed": True,
                                     "error": f"{type(exc).__name__}: {exc}"}
                raise
            event.update(status="COMPLETED", accelerator_used=True,
                         usb_inference_roundtrip_ms=(finished_ns-started_ns)/1e6,
                         timing_scope="Synchronous HailoRT input binding and run completion; includes SDK host format conversion and USB transfer; excludes output copying/decoding")
            return outputs, event

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            errors = []
            for entry in reversed(list(self._entries.values())):
                try:
                    if entry["configured"] is not None:
                        entry["configured"].__exit__(None, None, None)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    entry.update(configured=None, bindings=None, outputs={}, model=None)
            self._entries.clear()
            device, self._device = self._device, None
            dll, self._dll = self._dll, None
            try:
                if device is not None:
                    device.release()
            except Exception as exc:
                errors.append(exc)
            finally:
                if dll is not None:
                    try:
                        dll.close()
                    except Exception as exc:
                        errors.append(exc)
            self._hp = None
            self._contents.clear()
            if errors:
                raise RuntimeError("Shared Hailo cleanup failed: "+"; ".join(map(str, errors)))

    def __enter__(self):
        return self.open()

    def __exit__(self, *args):
        self.close()
