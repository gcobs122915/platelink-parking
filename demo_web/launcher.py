"""Foreground supervisor. Child process handles belong only to this demo run."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from demo_config import ROOT, RUNTIME, CORE_ROOT, load_config, prepare_runtime


def acquire_lock():
    RUNTIME.mkdir(exist_ok=True)
    stream = (RUNTIME / 'launcher.lock').open('a+b')
    stream.seek(0, 2)
    if stream.tell() == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        stream.close()
        raise RuntimeError('A demo launcher already owns this runtime directory')
    return stream


def request_stop():
    prepare_runtime()
    (RUNTIME / 'STOP_DEMO').write_text('stop this demo only\n', encoding='utf-8')
    config = json.loads((CORE_ROOT / 'config.json').read_text(encoding='utf-8-sig'))
    state = (CORE_ROOT / config['state_dir']).resolve()
    state.mkdir(exist_ok=True)
    (state / 'STOP').write_text('stop this demo core only\n', encoding='utf-8')


def supervise(with_core):
    config = load_config()
    if with_core and os.name != 'nt':
        raise RuntimeError('The validated hardware runtime is Windows-only')
    if with_core and importlib.util.find_spec('hailo_platform') is None:
        raise RuntimeError('Install matching HailoRT 5.3.2 wheel/driver and model bundle first; see README')
    lock = acquire_lock()
    children, logs = [], []
    stopping = False
    def on_signal(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    try:
        prepare_runtime()
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', config['port']))
        for flag in (RUNTIME / 'STOP_DEMO', CORE_ROOT / 'state/STOP'):
            flag.unlink(missing_ok=True)
        scripts = [('core', 'run_isolated_core.py')] if with_core else []
        scripts.append(('server', 'demo_server.py'))
        for name, script in scripts:
            stdout = (RUNTIME / f'{name}.stdout.log').open('ab')
            stderr = (RUNTIME / f'{name}.stderr.log').open('ab')
            logs.extend([stdout, stderr])
            process = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', '-u', str(ROOT / script)],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            children.append((name, process))
        (RUNTIME / 'processes.json').write_text(json.dumps({
            'supervisor_pid': os.getpid(), 'started_at': time.time(),
            'children': {name: child.pid for name, child in children}
        }, indent=2), encoding='utf-8')
        print(f'Open http://127.0.0.1:{config["port"]} in Edge or Chrome.', flush=True)
        print('Keep this terminal open. Ctrl+C or "python demo_web/launcher.py stop" stops this demo.', flush=True)
        print('No image is submitted automatically. Read runtime/*.log if model loading fails.', flush=True)
        while not stopping and not (RUNTIME / 'STOP_DEMO').exists():
            exited = [(name, p.returncode) for name, p in children if p.poll() is not None]
            if exited:
                raise RuntimeError(f'Demo child exited: {exited}; inspect runtime logs')
            time.sleep(0.25)
    finally:
        if children:
            request_stop()
            # Allow the core to finish in-flight work, then stop only our owned children.
            deadline = time.monotonic() + 30
            for name, process in children:
                try:
                    process.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    print(f'{name} did not stop within 30 seconds; terminating this demo child.', flush=True)
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        for stream in logs:
            stream.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'serve', 'stop', 'check'))
    args = parser.parse_args()
    try:
        if args.action == 'stop':
            request_stop()
            print('Demo-local stop flags written. No external process was signaled.')
        elif args.action == 'check':
            config = load_config()
            print(f'Core source found: {config["app"]}')
            print(f'Loopback port: {config["port"]}')
            print('Samples: ' + ', '.join(f'{key}={file.is_file()}' for key, file in config['samples'].items()))
            print('This checks portable paths only; it does not validate models, driver or USB inference.')
        else:
            supervise(args.action == 'start')
    except (RuntimeError, ValueError, OSError) as exc:
        print(f'Demo error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
