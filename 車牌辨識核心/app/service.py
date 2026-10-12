"""Durable local file exchange. No socket, HTTP server, or CPU neural fallback."""
from __future__ import annotations
import json
import os
from pathlib import Path
import signal
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from backend import InputError, read_photo
from file_events import DirectoryWakeup
from queue_store import InputLock, JobStore
from results import EventLog, atomic_bytes, atomic_json, now, text_result
from settings import COMPLETE, IMAGE_SUFFIXES, CORE_VERSION, basename_from_flag, identity
from memory_stats import snapshot as memory_snapshot


class FileCore:
    def __init__(self, settings, recognizer_factory, max_jobs=0):
        self.settings, self.factory, self.max_jobs = settings, recognizer_factory, max_jobs
        self.input = Path(settings['input_dir'])
        self.state_dir = Path(settings['state_dir'])
        self.stop = threading.Event()
        self.pulse_stop = threading.Event()
        self.state_lock = threading.Lock()
        self.log = EventLog(self.state_dir)
        self.recognizer = self.store = self.wakeup = None
        self.models = []
        self.protocol_id = None
        self.processed = 0
        self.fatal = None
        self.executor = None
        self.queue, self.futures = deque(), {}
        self.status = dict(state='starting', at=now(), pid=os.getpid(), input_ready=False,
            instance_id=uuid.uuid4().hex, input_dir=str(self.input), result_dir=settings['result_dir'],
            details_dir=settings['details_dir'], backend='HAILO_USB_ALL', api_server=False,
            max_inflight_images=settings['max_inflight_images'], max_queued_jobs=settings['max_queued_jobs'],
            queue_depth=0, active_jobs=[], peak_active_images=0,
            models_loaded_once=False, completed_jobs=0, failed_deliveries=0,
            current_job=None, cpu_neural_fallback=False, core_version=CORE_VERSION)

    def set_status(self, **values):
        with self.state_lock:
            self.status.update(values)

    def heartbeat(self):
        metrics = self.recognizer.metrics() if self.recognizer and hasattr(self.recognizer, 'metrics') else {}
        with self.state_lock:
            snapshot = {**self.status, 'at': now(), 'host_memory': memory_snapshot(), 'usb': metrics}
        atomic_json(self.state_dir / 'status.json', snapshot)

    def pulse(self):
        while not self.pulse_stop.wait(self.settings['heartbeat_seconds']):
            try:
                self.heartbeat()
            except Exception as exc:
                self.fatal = f'Heartbeat write failed: {exc}'
                self.stop.set()
                return

    def should_stop(self):
        return self.stop.is_set() or (self.state_dir / 'STOP').exists()

    def request_stop(self, *args):
        self.stop.set()

    def targets(self, job):
        return Path(job['json_path']), Path(job['txt_path'])

    def pending_marker(self, job):
        return self.input / job['marker_name']

    def claim(self, flag, basename):
        job = self.store.register(basename, flag.name, self.settings)
        if job is None:
            return None
        try:
            marker = self.pending_marker(job)
            if marker.exists():
                raise FileExistsError(f'Already processing: {marker.name}')
            flag.rename(marker)
        except Exception as exc:
            self.store.update(job['job_id'], state='failed', error=f'Claim failed: {exc}')
            self.log('CLAIM_FAILED', f'{flag.name}: {exc}', job_id=job['job_id'])
            return None
        return job  # Durable queued admission precedes actual execution.

    def base_payload(self, job):
        return dict(schema='plate-recognition/v1', job_id=job['job_id'], basename=job['basename'],
            image=None, source_sha256=None, original_size=None, status='error',
            models=self.models, protocol_id=self.protocol_id, detections=[], plates=[], errors=[], warnings=[],
            devices={stage: {'configured_backend': 'HAILO_USB', 'completed_runs': 0, 'cpu_neural_fallback': False}
                     for stage in ('yolo', 'pose', 'ocr')},
            backend='HAILO_USB_ALL', cpu_neural_fallback=False, execution_providers={},
            timing={'total_ms': None}, completed_at=None, admitted_at=job['admitted_at'],
            result_policy=self.settings['result_policy'], core_version=CORE_VERSION,
            timing_scope='total_ms: admission through atomic TXT publication, including admitted queue wait; '
                         'excludes pre-admission folder wait, startup, final JSON and logging. Null after recovery.',
            result_file=job['txt_path'], details_file=job['json_path'])

    def find_photo(self, basename):
        matches = [p for p in self.input.iterdir() if p.is_file() and p.stem.casefold() == basename.casefold()
                   and p.suffix.lower() in IMAGE_SUFFIXES]
        if len(matches) != 1:
            raise InputError('image_not_found' if not matches else 'image_not_unique',
                             '找不到同名照片' if not matches else '同一 basename 有多張支援格式的照片')
        return matches[0]

    def prepare(self, job, recovered=False, admitted_ns=None):
        work_started = time.perf_counter_ns()
        payload = self.base_payload(job)
        self.log('PROCESSING', f"{job['basename']} 開始辨識", job_id=job['job_id'], recovered=recovered)
        try:
            # Consumers reusing a basename must also clear their stale TXT before submission.
            Path(job['txt_path']).unlink(missing_ok=True)
            source = self.find_photo(job['basename'])
            payload['image'] = source.name
            started = time.perf_counter_ns()
            bgr, digest, orientation = read_photo(source, self.settings)
            decoded = time.perf_counter_ns()
            payload.update(source_sha256=digest, original_size=[bgr.shape[1], bgr.shape[0]], **orientation)
            recognized = self.recognizer.infer(bgr, source.name)
            payload.update(recognized)
            payload['timing']['image_read_decode_hash_ms'] = (decoded-started)/1e6
            payload['timing']['total_ms'] = None
            if self.settings['save_yolo_image']:
                try:
                    annotation = self.recognizer.annotate(bgr, payload)
                    path = Path(self.settings['yolo_dir']) / (job['basename']+'-yolo.jpg')
                    atomic_bytes(path, annotation)
                    payload['yolo_image'] = str(path)
                except Exception as exc:
                    payload['warnings'].append(dict(stage='annotation', code='annotation_failed', message=str(exc)))
        except Exception as exc:
            if isinstance(exc, InputError) and exc.source_sha256 is not None:
                payload['source_sha256'] = exc.source_sha256
            payload['status'] = 'error'
            payload['errors'].append(dict(stage='input_or_inference', code=getattr(exc, 'code', 'processing_failed'),
                                          message=f'{type(exc).__name__}: {exc}'))
        payload['recovered'] = recovered
        payload['timing']['queue_wait_ms'] = (work_started-admitted_ns)/1e6 if admitted_ns is not None else None
        payload['timing']['processing_ms'] = (time.perf_counter_ns()-work_started)/1e6
        if recovered:
            payload['warnings'].append(dict(code='inference_restarted_after_interruption', message='未完成的工作已重新辨識'))
        return payload

    def finish_work(self, job, payload, recovered=False, admitted_ns=None):
        text = text_result(payload['plates'], self.settings['result_policy'])
        # Commit the answer before publishing either output. Recovery reuses this payload.
        job = self.store.update(job['job_id'], state='publishing',
            payload=json.dumps(payload, ensure_ascii=False, allow_nan=False), result_text=text)
        self.deliver(job, admitted_ns=admitted_ns, recovered=recovered)

    def work(self, job, recovered=False, admitted_ns=None):
        """Serial path used for interrupted-job recovery before accepting new work."""
        self.set_status(current_job=job['job_id'])
        self.store.update(job['job_id'], state='running')
        payload = self.prepare(job, recovered, admitted_ns)
        self.finish_work(job, payload, recovered, admitted_ns)

    def drain(self, start_new=True):
        for future in list(self.futures):
            if future.done():
                job, admitted_ns = self.futures.pop(future)
                self.finish_work(job, future.result(), admitted_ns=admitted_ns)
        while start_new and not self.should_stop() and self.queue and len(self.futures) < self.settings['max_inflight_images']:
            job, admitted_ns = self.queue.popleft()
            self.store.update(job['job_id'], state='running')
            future = self.executor.submit(self.prepare, job, False, admitted_ns)
            self.futures[future] = (job, admitted_ns)
        self.set_status(queue_depth=len(self.queue), active_jobs=[j['job_id'] for j,_ in self.futures.values()],
            peak_active_images=max(self.status['peak_active_images'], len(self.futures)))

    def deliver(self, job, admitted_ns=None, recovered=False):
        try:
            payload = json.loads(job['payload'])
            if recovered:
                payload['recovered'] = True
                payload['timing']['total_ms'] = None
                payload['completed_at'] = None
                payload['warnings'].append(dict(code='publication_resumed', message='由已保存的答案續寫，未重跑模型'))
            json_path, txt_path = self.targets(job)
            atomic_json(json_path, payload)
            atomic_bytes(txt_path, job['result_text'].encode('utf-8'))
            published_ns = time.perf_counter_ns()
            payload['timing']['total_ms'] = ((published_ns-admitted_ns)/1e6 if admitted_ns is not None and not recovered else None)
            payload['completed_at'] = now()
            payload['result_txt_flush_fsync'] = True
            # Save final metadata before the second JSON replace, for crash recovery.
            self.store.update(job['job_id'], payload=json.dumps(payload, ensure_ascii=False, allow_nan=False))
            atomic_json(json_path, payload)
            self.pending_marker(job).unlink(missing_ok=True)
            self.store.update(job['job_id'], state='done', payload=None, result_text=None,
                              completed_at=payload['completed_at'], error=None)
        except Exception as exc:
            self.fail_delivery(job, exc)
            return
        self.processed += 1
        self.set_status(completed_jobs=self.processed, current_job=None)
        elapsed = payload['timing']['total_ms']
        elapsed_text = f'{elapsed:.1f} ms' if elapsed is not None else '恢復工作，耗時未計'
        for line in job['result_text'].splitlines():
            accepted, number, score = line.split('|')
            self.log('RESULT', f"{job['basename']} 車號={number} 信心={score} 通過={accepted} 狀態={payload['status']} 耗時={elapsed_text}",
                     job_id=job['job_id'], image=payload['image'], text=number,
                     accepted=accepted == 'true', confidence=float(score), status=payload['status'], total_ms=elapsed)
        for error in payload['errors']:
            self.log('RECOGNITION_ERROR', error['message'], job_id=job['job_id'], code=error['code'])
        self.heartbeat()

    def fail_delivery(self, job, exc):
        marker = self.pending_marker(job)
        failed = self.input / (job['flag_name']+'.failed-'+job['job_id'][:8])
        try:
            if marker.exists():
                marker.rename(failed)
                job['marker_name'] = failed.name
        except OSError:
            pass
        self.store.update(job['job_id'], state='publish_failed', marker_name=job['marker_name'], error=f'{type(exc).__name__}: {exc}')
        self.set_status(current_job=None, failed_deliveries=self.status['failed_deliveries']+1)
        self.log('PUBLISH_FAILED', f"{job['basename']} 寫檔失敗：{exc}；答案已保留，重啟後續寫", job_id=job['job_id'])

    def recover(self):
        pending = self.store.pending()
        known = {j['marker_name'].casefold() for j in pending}
        for marker in sorted(self.input.iterdir()):
            if not marker.is_file() or not marker.name.lower().endswith('complete.txt.processing') or marker.name.casefold() in known:
                continue
            flag_name = marker.name[:-len('.processing')]
            try:
                basename = basename_from_flag(flag_name)
            except ValueError:
                continue
            orphan = self.store.register(basename, flag_name, self.settings, marker.name)
            if orphan:
                pending.append(orphan)
        for job in pending:
            if self.should_stop():
                break
            if job['payload'] is not None:
                self.log('RECOVER', f"{job['basename']} 續寫已保存的答案", job_id=job['job_id'])
                self.deliver(job, recovered=True)
                continue
            marker, flag = self.pending_marker(job), self.input / job['flag_name']
            if not marker.exists() and flag.exists():
                flag.rename(marker)
            if not marker.exists():
                self.store.update(job['job_id'], state='failed', error='Recovery marker missing; no result fabricated')
                self.log('RECOVERY_FAILED', f"{job['basename']} 找不到接單旗標，沒有產生結果", job_id=job['job_id'])
                continue
            self.store.update(job['job_id'], state='running')
            self.work(job, recovered=True)

    def scan(self):
        for flag in sorted(self.input.iterdir(), key=lambda p: p.name.casefold()):
            if (self.should_stop() or (self.max_jobs and self.processed+len(self.queue)+len(self.futures) >= self.max_jobs)
                    or (self.executor and len(self.queue) >= self.settings['max_queued_jobs'])):
                break
            if not flag.is_file():
                continue
            try:
                basename = basename_from_flag(flag.name)
            except ValueError as exc:
                failed = flag.with_name(flag.name+'.failed-'+uuid.uuid4().hex[:8])
                try:
                    flag.rename(failed)
                except OSError:
                    pass
                self.log('INVALID_FLAG', f'{flag.name}: {exc}')
                continue
            if basename is None:
                continue
            admitted_ns = time.perf_counter_ns()
            job = self.claim(flag, basename)
            if job is not None:
                if self.executor is None:
                    self.work(job, admitted_ns=admitted_ns)
                else:
                    self.queue.append((job, admitted_ns))
                    self.log('QUEUED', f"{basename} 已接單，等待分派", job_id=job['job_id'])

    def run(self):
        # Acquire before writing status: a second instance must not clobber the owner's heartbeat.
        with InputLock(self.input):
            previous = None
            thread = None
            try:
                if (self.state_dir / 'STOP').exists():
                    raise RuntimeError(f'請先刪除停止旗標再啟動：{self.state_dir / "STOP"}')
                if threading.current_thread() is threading.main_thread():
                    previous = signal.signal(signal.SIGINT, self.request_stop)
                self.heartbeat()
                thread = threading.Thread(target=self.pulse, daemon=True, name='core-heartbeat')
                thread.start()
                self.store = JobStore(self.state_dir)
                self.log('LOADING', '載入 YOLO／POSE／OCR 至 USB，模型僅載入一次；不啟動 API Server')
                started = time.perf_counter_ns()
                self.recognizer = self.factory(self.settings)
                self.models = self.recognizer.models
                self.protocol_id = identity(self.settings, self.models)
                self.wakeup = DirectoryWakeup(self.input, self.settings['scan_interval_seconds'], self.settings['poll_interval_seconds'])
                self.set_status(state='recovering', models=self.models, protocol_id=self.protocol_id,
                    models_loaded_once=True, model_load_ms=(time.perf_counter_ns()-started)/1e6,
                    watcher=self.wakeup.mode)
                self.recover()
                self.executor = ThreadPoolExecutor(max_workers=self.settings['max_inflight_images'], thread_name_prefix='photo')
                if not self.should_stop():
                    self.set_status(state='running', input_ready=True)
                    self.heartbeat()
                    self.log('INPUT_READY', f'照片及 Complete 旗標請放：{self.input}')
                    self.log('OUTPUT', f"TXT：{self.settings['result_dir']}；JSON：{self.settings['details_dir']}")
                    self.log('QUEUE', f"同時處理上限 {self.settings['max_inflight_images']} 張；等待上限 {self.settings['max_queued_jobs']} 筆；共用一份 USB 模型")
                    self.log('POSE_RANGE', '裁圖選最小可容納的 USB 尺寸；只補右／下邊，過大時等比例縮小，四角換算回原裁圖')
                    memory = memory_snapshot()
                    if memory.get('available'):
                        self.log('MEMORY', f"電腦 RAM {memory['process_working_set_bytes']/1048576:.1f} MiB；"
                            f"提交量 {memory['process_private_bytes']/1048576:.1f} MiB；USB 棒 RAM 用量無 Windows SDK 讀值")
                while not self.should_stop() and not (self.max_jobs and self.processed >= self.max_jobs):
                    self.drain()
                    self.scan()
                    self.drain()
                    if not self.should_stop() and not (self.max_jobs and self.processed >= self.max_jobs):
                        if self.futures:
                            # Wake promptly for completed answers; input is rescanned at least every 50ms while busy.
                            wait(self.futures, timeout=.05, return_when=FIRST_COMPLETED)
                        else:
                            self.wakeup.wait(self.stop)
                if self.fatal:
                    raise RuntimeError(self.fatal)
                self.set_status(state='stopping', input_ready=False)
                self.log('STOPPING', '停止接單；完成執行中的照片，尚未開始的工作保留到下次啟動')
                while self.futures:
                    wait(self.futures, timeout=.25, return_when=FIRST_COMPLETED)
                    self.drain(start_new=False)
                return 0
            except Exception as exc:
                self.set_status(state='error', input_ready=False, error=f'{type(exc).__name__}: {exc}')
                self.log('ERROR', str(exc))
                return 1
            finally:
                self.pulse_stop.set()
                if thread:
                    thread.join(timeout=6)
                if self.executor:
                    self.executor.shutdown(wait=True)
                if self.wakeup:
                    self.wakeup.close()
                if self.recognizer:
                    # Capture final counters while the device still exists.
                    self.set_status(final_usb_metrics=self.recognizer.metrics() if hasattr(self.recognizer, 'metrics') else {})
                    try:
                        self.recognizer.close()
                    except Exception as exc:
                        self.set_status(state='error', error=f'USB cleanup failed: {exc}', models_released=False)
                        self.log('CLOSE_FAILED', str(exc))
                    self.recognizer = None
                if self.store:
                    self.store.close()
                if previous is not None:
                    signal.signal(signal.SIGINT, previous)
                self.set_status(state='error' if self.status['state']=='error' else 'stopped',
                                input_ready=False, models_released=self.status.get('models_released', True), current_job=None, active_jobs=[])
                self.heartbeat()
                self.log('STOPPED', '核心已停止；照片與結果檔案保留')
                if self.status.get('models_released') is False:
                    return 1
