"""One resident VDevice; independent per-job buffers and asynchronous USB jobs.

Concurrent host requests do not imply independently addressable physical NPU cores.
The Hailo ROUND_ROBIN scheduler owns device execution. No model is duplicated.
"""
from __future__ import annotations
import os
import threading
import time
import numpy as np
from hailo_shared_engine import HailoSharedEngine
from all_usb_pipeline import AllUsbPipeline, pose_raw
from pose_ranges import preprocess as pose_range_input, restore_points


class AsyncSharedEngine(HailoSharedEngine):
    def open(self):
        super().open()
        self.active_requests = self.max_active_requests = 0
        for entry in self._entries.values():
            entry['submit_lock'] = threading.Lock()
            entry['async_capacity'] = entry['configured'].get_async_queue_size()
        return self

    def run(self, key, tensor, timeout_ms=30000):
        with self._lock:
            if os.getpid() != self.creation_pid or self._closed or self._faulted or self._device is None:
                raise RuntimeError('Shared asynchronous USB engine is unavailable')
            if key not in self.specs:
                raise ValueError(f'Uncompiled Hailo model/profile: {key}')
            spec, entry = self.specs[key], self._entries[key]
            if (not isinstance(tensor, np.ndarray) or tuple(tensor.shape) != spec.input_shape
                    or tensor.dtype != np.dtype(spec.input_dtype) or not tensor.flags.c_contiguous):
                raise ValueError(f'{key}: invalid input shape/format; no implicit resize or CPU fallback')
            if type(timeout_ms) is not int or timeout_ms <= 0:
                raise ValueError('timeout_ms must be positive')
            self.run_sequence += 1
            sequence = self.run_sequence
        if not np.isfinite(tensor).all():
            raise ValueError('Nonfinite input')
        outputs = {name: np.empty(shape, np.float32) for name, shape in spec.output_shapes.items()}
        completion = {}
        def callback(completion_info):
            # Retain the input/output/bindings in the SDK until this callback returns.
            completion.update(error=completion_info.exception, finished_ns=time.perf_counter_ns())
        started = time.perf_counter_ns()
        event = dict(stage=spec.stage, model_key=key, profile_hw=list(spec.profile_hw),
            hef_sha256=spec.hef_sha256, device=self.device_id, shared_engine_id=self.engine_id,
            run_seq=sequence, input_shape=list(tensor.shape), host_input_dtype=str(tensor.dtype),
            output_shapes={name:list(shape) for name,shape in spec.output_shapes.items()},
            backend='HAILO_USB', cpu_neural_fallback=False, asynchronous=True)
        counted = False
        try:
            # Only submission is serialized per model. No lock is held while inference runs.
            with entry['submit_lock']:
                entry['configured'].wait_for_async_ready(timeout_ms=timeout_ms)
                bindings = entry['configured'].create_bindings(output_buffers=outputs)
                bindings.input().set_buffer(tensor)
                with self._lock:
                    self.active_requests += 1
                    counted = True
                    self.max_active_requests = max(self.max_active_requests, self.active_requests)
                submitted = time.perf_counter_ns()
                job = entry['configured'].run_async([bindings], callback)
            job.wait(timeout_ms)
            if 'error' not in completion:
                raise RuntimeError('HailoRT job completed without a completion callback')
            if completion['error'] is not None:
                raise completion['error']
            if not all(np.isfinite(value).all() for value in outputs.values()):
                raise ValueError('Nonfinite USB model output')
        except BaseException as exc:
            with self._lock:
                self._faulted = True
                self.last_failure = {**event, 'status':'FAILED', 'error':f'{type(exc).__name__}: {exc}'}
            raise
        finally:
            if counted:
                with self._lock:
                    self.active_requests -= 1
        finished = completion['finished_ns']
        event.update(status='COMPLETED', accelerator_used=True,
            submitted_monotonic_ns=submitted, completed_monotonic_ns=finished,
            usb_inference_roundtrip_ms=(finished-started)/1e6,
            sdk_queue_wait_ms=(submitted-started)/1e6,
            timing_scope='Async submission through callback, including SDK scheduling and USB transfer; '
                         'overlap is host outstanding requests, not proof of physical NPU core parallelism')
        return outputs, event

    def metrics(self):
        with self._lock:
            return dict(shared_engine_id=self.engine_id, device=self.device_id,
                active_usb_requests=self.active_requests, peak_outstanding_usb_requests=self.max_active_requests,
                loaded_model_count=len(self.specs), model_copies=1,
                async_capacity_by_model={k:e['async_capacity'] for k,e in self._entries.items()},
                hef_file_bytes=sum(len(b) for b in self._contents.values()),
                device_memory_used_bytes=None, device_memory_total_bytes=None,
                device_memory_measurement='Unavailable: installed Windows HailoRT monitor and performance stats are unsupported')


class ConcurrentUsbPipeline(AllUsbPipeline):
    def __init__(self):
        self.local = threading.local()
        super().__init__(engine_factory=AsyncSharedEngine, ocr_vocabulary='plate41')

    def corners(self, crop):
        pixels, plan = pose_range_input(crop, self.pose_specs)
        height,width = plan['profile_hw']
        spec = self.pose_specs[(height,width)]
        outputs,event = self._run(spec,pixels)
        detections = pose_raw.nms(pose_raw.decode_heads(outputs,height,width))
        info = dict(backend='HAILO_USB',neural_run_seq=event['run_seq'],
            input_nchw=[1,3,height,width],native_crop_wh=[crop.shape[1],crop.shape[0]],
            input_transform=plan,per_corner_confidence=None)
        if not len(detections):
            return None,{**info,'valid':False,'reason':'no_pose_detection'}
        model_points = detections[0,5:].reshape(4,2)
        points = restore_points(model_points,plan)
        return points,{**info,'valid':True,'confidence':float(detections[0,4]),
            'corners_model_xy':model_points.tolist(),'corners_crop_xy':points.tolist()}

    @property
    def neural_stage_runs(self):
        if not hasattr(self.local, 'runs'):
            self.local.runs = []
        return self.local.runs

    @neural_stage_runs.setter
    def neural_stage_runs(self, value):
        self.local.runs = value

    @property
    def _last_accelerated(self):
        return getattr(self.local, 'accelerated', False)

    @_last_accelerated.setter
    def _last_accelerated(self, value):
        self.local.accelerated = value
