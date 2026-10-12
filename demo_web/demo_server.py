"""Local-only recording bridge to the unchanged resident USB core.

No inference runs at startup. POST /api/run accepts only three fixed photo IDs.
Every submission uses the original client's unique file-protocol job; no result
cache is used. Historical test results are never loaded by this server.
"""
from __future__ import annotations

import datetime as dt
import ctypes
from ctypes import wintypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

from demo_config import ROOT, RUNTIME, CORE_ROOT, load_config, prepare_runtime

CONFIG = load_config()
APP = CONFIG['app']
SOURCE_ROOT = APP.parent.parent
SAMPLES = CONFIG['samples']
PORT = CONFIG['port']
HOSTS = {f'127.0.0.1:{PORT}', f'localhost:{PORT}'}
MAX_RECORDING = 100 * 1024 * 1024
MAX_SAMPLE = 20 * 1024 * 1024
sys.path.insert(0, str(APP))
from client import submit
from settings import load_settings


def now():
    return dt.datetime.now().astimezone().isoformat(timespec='milliseconds')


def read_json_once(path):
    # The core atomically replaces heartbeat/result files. Ordinary Windows
    # Python opens omit FILE_SHARE_DELETE and can briefly block that rename.
    # Every bridge JSON read explicitly shares deletion with the producer.
    if os.name != 'nt':
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    import msvcrt
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                               wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    handle = api.CreateFileW(str(Path(path)), 0x80000000, 0x1 | 0x2 | 0x4,
                             None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        api.CloseHandle(handle)
        raise
    with os.fdopen(fd, 'r', encoding='utf-8-sig') as stream:
        contents = stream.read()
    return json.loads(contents)


def read_json(path):
    deadline = time.monotonic() + 2.0
    while True:
        try:
            return read_json_once(path)
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def check_ready(settings, status=None):
    if (Path(settings['state_dir']) / 'STOP').exists():
        raise RuntimeError('Demo core stop flag is present')
    status = status if status is not None else read_json(Path(settings['state_dir']) / 'status.json')
    age = (dt.datetime.now().astimezone() - dt.datetime.fromisoformat(status['at'])).total_seconds()
    if status.get('state') != 'running' or not status.get('input_ready') or not 0 <= age < 20:
        raise RuntimeError('Demo core is not ready or its heartbeat is stale')
    if os.name != 'nt':
        raise RuntimeError('Hardware demo execution is supported on Windows only')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    api.GetExitCodeProcess.restype = wintypes.BOOL
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = api.OpenProcess(0x1000, False, int(status['pid']))
    if not handle:
        raise RuntimeError('Demo core process is offline')
    try:
        code = wintypes.DWORD()
        if not api.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
            raise RuntimeError('Demo core process is offline')
    finally:
        api.CloseHandle(handle)


def safe_error(value):
    text = str(value).replace(str(SOURCE_ROOT), '[local workspace]').replace(str(ROOT), '[demo]')
    text = re.sub(r'''[A-Za-z]:[\\/][^"'\r\n]*''', '[local file]', text)
    return text[:500]


class Demo:
    def __init__(self):
        prepare_runtime()
        self.settings = load_settings(CORE_ROOT)
        self.lock = threading.RLock()
        self.jobs = {}
        self.pending = queue.Queue()
        self.stop = threading.Event()
        self.started_at = now()
        self._core_cache = {}
        self._core_cache_at = 0.0
        self.pnp = {'present': None, 'status': 'checking', 'name': None, 'checked_at': None}

    def start(self):
        threading.Thread(target=self.worker, name='demo-serial-jobs', daemon=True).start()
        threading.Thread(target=self.check_device, name='demo-pnp-monitor', daemon=True).start()

    def check_device(self):
        if os.name != 'nt':
            with self.lock:
                self.pnp = {'present': None, 'status': 'unsupported', 'name': None,
                            'checked_at': now(), 'error': 'Windows hardware runtime required'}
            return
        powershell = Path(os.environ.get('SystemRoot', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
        command = ("$d=@(Get-PnpDevice -PresentOnly -ErrorAction Stop | "
                   "Where-Object {$_.InstanceId -match 'VID_0B05&PID_1D6F'}); "
                   "[pscustomobject]@{present=($d.Count -gt 0);status=($d | Select-Object -First 1 -ExpandProperty Status);"
                   "name=($d | Select-Object -First 1 -ExpandProperty FriendlyName)} | ConvertTo-Json -Compress")
        while not self.stop.is_set():
            try:
                result = subprocess.run([str(powershell), '-NoProfile', '-NonInteractive', '-Command', command],
                    capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode:
                    raise RuntimeError('Windows PnP query failed')
                value = json.loads(result.stdout.decode('utf-8-sig', errors='replace'))
                value['checked_at'] = now()
                with self.lock:
                    self.pnp = value
            except Exception as exc:
                with self.lock:
                    self.pnp = {'present': None, 'status': 'unknown', 'name': None,
                                'checked_at': now(), 'error': safe_error(exc)}
            self.stop.wait(5)

    def core_state(self):
        try:
            with self.lock:
                if time.monotonic() - self._core_cache_at > 0.75:
                    self._core_cache = read_json(Path(self.settings['state_dir']) / 'status.json')
                    self._core_cache_at = time.monotonic()
                value = dict(self._core_cache)
            age = (dt.datetime.now().astimezone() - dt.datetime.fromisoformat(value['at'])).total_seconds()
            try:
                check_ready(self.settings, status=value)
                ready = True
            except Exception:
                ready = False
            return value, ready, age
        except Exception:
            return {}, False, None

    def public_job(self, job):
        return {key: value for key, value in job.items() if not key.startswith('_')}

    def state(self):
        core, ready, age = self.core_state()
        with self.lock:
            jobs = [self.public_job(row) for row in self.jobs.values()]
            pnp = dict(self.pnp)
        active = any(row['status'] in ('queued', 'running') for row in jobs)
        device_ready = ready and pnp.get('present') is True and pnp.get('status') == 'OK'
        status = 'busy' if ready and active else 'ready' if device_ready else 'error' if core.get('state') == 'error' else 'loading'
        return {'status': status, 'core_ready': ready, 'device_ready': device_ready,
                'now': now(), 'server_started_at': self.started_at,
                'device': {**pnp, 'source': 'Windows PnP; ready also requires a live core heartbeat'},
                'core': {'state': core.get('state', 'not_started'), 'heartbeat_at': core.get('at'),
                         'heartbeat_age_seconds': age, 'models_loaded': len(core.get('models', [])),
                         'models_loaded_once': core.get('models_loaded_once', False),
                         'active_jobs': len(core.get('active_jobs', [])), 'queue_depth': core.get('queue_depth', 0),
                         'completed_jobs': core.get('completed_jobs', 0),
                         'backend': core.get('backend'),
                         'error': safe_error(core.get('error', '')) or None},
                'jobs': jobs, 'sample_ids': list(SAMPLES),
                'sample_available': {key: file.is_file() and file.stat().st_size <= MAX_SAMPLE
                                     for key, file in SAMPLES.items()},
                'timing_note': 'processing_ms excludes queueing and final publication; total_ms ends at atomic TXT publication; client_elapsed_ms includes submission and result polling.',
                'result_note': 'Each run submits a new job. Completed results are recorded outcomes of this demo session, not cached inference responses.'}

    def enqueue(self, sample):
        if sample not in (*SAMPLES, 'all'):
            raise ValueError('Unknown sample ID')
        if not self.state()['device_ready']:
            raise RuntimeError('The core and USB device are not ready')
        selected = list(SAMPLES) if sample == 'all' else [sample]
        for item in selected:
            if not SAMPLES[item].is_file():
                raise ValueError(f'Missing {item} sample; add your JPEG to demo_web/samples')
            if not 1 <= SAMPLES[item].stat().st_size <= MAX_SAMPLE:
                raise ValueError('Each sample must be between 1 byte and 20 MiB')
        with self.lock:
            if sum(row['status'] in ('queued', 'running') for row in self.jobs.values()) + len(selected) > 12:
                raise RuntimeError('Demo queue is full')
            # Keep browser responses bounded; old output files remain local for inspection.
            while len(self.jobs) + len(selected) > 200:
                old = next((key for key, row in self.jobs.items()
                            if row['status'] in ('completed', 'error')), None)
                if old is None:
                    break
                del self.jobs[old]
            added = []
            for item in selected:
                job_id = uuid.uuid4().hex
                job = {'id': job_id, 'sample': item, 'status': 'queued',
                       'input_url': f'/sample/{item}.jpg', 'result_url': None,
                       'plate_text': None, 'processing_ms': None, 'total_ms': None,
                       'client_elapsed_ms': None, 'fallback': None, 'stages': [],
                       'error': None, 'queued_at': now(), 'submitted_at': None,
                       'completed_at': None, 'replay': False, 'result_source': 'fresh_demo_run'}
                self.jobs[job_id] = job
                self.pending.put(job_id)
                added.append({'id': job_id, 'sample': item, 'status': 'queued'})
            return {'accepted': True, 'jobs': added}

    def worker(self):
        while not self.stop.is_set():
            try:
                job_id = self.pending.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                with self.lock:
                    job = self.jobs[job_id]
                    job['status'] = 'running'
                    sample = job['sample']
                check_ready(self.settings)
                started = time.perf_counter_ns()
                submission_time = now()
                submitted = submit(SAMPLES[sample], self.settings)
                with self.lock:
                    job.update(submitted_at=submission_time, _submission=submitted)
                detail_path = Path(submitted['result_json'])
                txt_path = Path(submitted['result_txt'])
                flag = Path(submitted['image']).with_name(submitted['basename'] + 'Complete.txt')
                deadline = time.monotonic() + 180
                while not self.stop.is_set():
                    if time.monotonic() > deadline:
                        raise TimeoutError('No complete result within 180 seconds; no result has been fabricated')
                    if detail_path.is_file() and txt_path.is_file() and not flag.exists() and not flag.with_name(flag.name + '.processing').exists():
                        result = read_json(detail_path)
                        if result.get('completed_at'):
                            if result.get('source_sha256') != submitted['source_sha256']:
                                raise RuntimeError('Result source hash does not match the submitted image')
                            if result.get('basename') != submitted['basename']:
                                raise RuntimeError('Result identity does not match the newly submitted job')
                            break
                    state, ready, age = self.core_state()
                    if not ready and state.get('state') in ('error', 'stopped'):
                        raise RuntimeError('Core stopped before a complete result was delivered')
                    self.stop.wait(0.05)
                else:
                    raise RuntimeError('Demo server stopped before result delivery')
                elapsed = (time.perf_counter_ns() - started) / 1e6
                events = result.get('neural_stage_runs', [])
                stages = [{key: event.get(key) for key in
                           ('stage', 'model_key', 'backend', 'status', 'cpu_neural_fallback',
                            'usb_inference_roundtrip_ms', 'run_seq', 'asynchronous')} for event in events]
                for stage in stages:
                    stage['source_stage'] = stage['stage']
                    if stage['stage'] == 'yolo':
                        stage['stage'] = 'plate'
                all_usb = bool(events) and all(event.get('backend') == 'HAILO_USB' and
                    event.get('status') == 'COMPLETED' and event.get('cpu_neural_fallback') is False for event in events)
                annotated = Path(self.settings['yolo_dir']) / (submitted['basename'] + '-yolo.jpg')
                errors = result.get('errors', [])
                with self.lock:
                    job.update(status='completed' if result.get('status') == 'ok' and all_usb else 'error',
                        result_url=f'/result/{job_id}/annotated.jpg' if annotated.is_file() else None,
                        plate_text='\n'.join(str(p.get('text', '')) for p in result.get('plates', [])),
                        plates=[{'text': p.get('text'), 'confidence': p.get('ocr_confidence'),
                                 'accepted': p.get('accepted')} for p in result.get('plates', [])],
                        processing_ms=result.get('timing', {}).get('processing_ms'),
                        total_ms=result.get('timing', {}).get('total_ms'), client_elapsed_ms=elapsed,
                        fallback=result.get('cpu_neural_fallback'), stages=stages,
                        all_observed_events_usb=all_usb, backend=result.get('backend'),
                        core_job_id=result.get('job_id'), completed_at=result.get('completed_at'),
                        error='; '.join(safe_error(e.get('message', e.get('code', ''))) for e in errors) or (None if all_usb else 'USB completion evidence is incomplete'),
                        replay=True, view_mode='completed_run_review', _annotated=annotated)
                (RUNTIME / 'demo_jobs.json').write_text(json.dumps(self.state()['jobs'], ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps({'event': 'demo_completed', 'id': job_id, 'sample': sample,
                                  'status': job['status'], 'at': now()}, ensure_ascii=False), flush=True)
            except Exception as exc:
                with self.lock:
                    self.jobs[job_id].update(status='error', error=safe_error(exc), completed_at=now(), replay=True)
                print(json.dumps({'event': 'demo_error', 'id': job_id, 'error': safe_error(exc)}, ensure_ascii=False), flush=True)
            finally:
                self.pending.task_done()


DEMO = None


class Handler(BaseHTTPRequestHandler):
    server_version = 'PlateLinkLocalDemo/1'

    def local_request(self):
        if self.headers.get('Host') not in HOSTS:
            return False
        origin = self.headers.get('Origin')
        site = self.headers.get('Sec-Fetch-Site')
        return site not in ('cross-site', 'same-site') and (not origin or origin in {f'http://{host}' for host in HOSTS})

    def respond(self, status, data, kind='application/json; charset=utf-8'):
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.local_request():
            return self.respond(403, {'error': 'Local same-origin requests only'})
        path = urlsplit(self.path).path
        if path == '/api/state':
            return self.respond(200, DEMO.state())
        if path in ('/', '/index.html'):
            index = ROOT / 'index.html'
            if not index.is_file():
                return self.respond(503, {'error': 'Frontend is being prepared'})
            return self.respond(200, index.read_bytes(), 'text/html; charset=utf-8')
        matched = re.fullmatch(r'/sample/(military|diplomatic|foreign)\.jpg', path)
        if matched:
            sample = SAMPLES[matched[1]]
            if not sample.is_file():
                return self.respond(404, {'error': 'Add your own JPEG under demo_web/samples'})
            if sample.stat().st_size > MAX_SAMPLE:
                return self.respond(413, {'error': 'Sample exceeds 20 MiB'})
            return self.respond(200, sample.read_bytes(), 'image/jpeg')
        matched = re.fullmatch(r'/result/([a-f0-9]{32})/annotated\.jpg', path)
        if matched:
            with DEMO.lock:
                file = DEMO.jobs.get(matched[1], {}).get('_annotated')
            if file and file.is_file():
                return self.respond(200, file.read_bytes(), 'image/jpeg')
        return self.respond(404, {'error': 'Not found'})

    def do_POST(self):
        if not self.local_request():
            return self.respond(403, {'error': 'Local same-origin requests only'})
        path = urlsplit(self.path).path
        try:
            length = int(self.headers.get('Content-Length', '-1'))
            if path == '/api/recording':
                if not self.headers.get('Content-Type', '').lower().startswith('video/webm'):
                    return self.respond(415, {'error': 'Expected video/webm'})
                if not 1 <= length <= MAX_RECORDING:
                    return self.respond(413, {'error': 'Recording must be between 1 byte and 100 MiB'})
                folder = RUNTIME / 'recordings'
                folder.mkdir(exist_ok=True)
                name = 'live-' + dt.datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '-' + uuid.uuid4().hex[:8] + '.webm'
                destination = folder / name
                remaining = length
                self.connection.settimeout(60)
                with destination.open('xb') as stream:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError('Recording upload ended before Content-Length bytes')
                        stream.write(chunk)
                        remaining -= len(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                return self.respond(201, {'saved_path': 'runtime/recordings/' + name, 'bytes': length})
            if path == '/api/run':
                if not self.headers.get('Content-Type', '').lower().startswith('application/json'):
                    return self.respond(415, {'error': 'Expected application/json'})
                if not 1 <= length <= 1024:
                    return self.respond(413, {'error': 'Invalid JSON body size'})
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict) or set(request) != {'sample'}:
                    raise ValueError('Expected exactly one sample field')
                return self.respond(202, DEMO.enqueue(request['sample']))
            return self.respond(404, {'error': 'Not found'})
        except RuntimeError as exc:
            return self.respond(409, {'error': safe_error(exc)})
        except Exception as exc:
            return self.respond(400, {'error': safe_error(exc)})

    def log_message(self, format, *args):
        # Avoid flooding the log with frontend heartbeat polling.
        if '/api/state' not in self.requestline:
            print(f'{now()} {format % args}', flush=True)


def main():
    global DEMO
    DEMO = Demo()
    DEMO.start()
    server = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    server.daemon_threads = True
    def watch_stop():
        while not DEMO.stop.wait(0.5):
            if (RUNTIME / 'STOP_DEMO').exists():
                server.shutdown()
                return
    threading.Thread(target=watch_stop, daemon=True).start()
    print(json.dumps({'event': 'demo_server_ready', 'url': f'http://127.0.0.1:{PORT}',
                      'pid': os.getpid(), 'inference_submitted': False}), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        DEMO.stop.set()
        server.server_close()


if __name__ == '__main__':
    main()
