"""Launch the original core with bounded retries for demo-local Windows I/O.

This does not modify source files, models, inference, result data or timings.
The extra retry handles transient access-denied/sharing violations while a
local observer reads an atomically replaced file. All retry targets must be
inside this demo's isolated core directory. Actual wait time remains measured.
"""
from pathlib import Path
import sys
import time

from demo_config import CORE_ROOT, load_config, prepare_runtime

prepare_runtime()
APP = load_config()['app']
sys.path.insert(0, str(APP))
import results

_original_atomic_bytes = results.atomic_bytes


def demo_atomic_bytes(path, data):
    target = Path(path).resolve()
    if not target.is_relative_to(CORE_ROOT):
        raise ValueError('Demo writer refuses a target outside the isolated core root')
    deadline = time.monotonic() + 2.0
    attempts = 0
    while True:
        try:
            value = _original_atomic_bytes(target, data)
            if attempts:
                print(f'[DEMO_IO_RETRY] atomic publication succeeded after {attempts} transient Windows sharing failures', flush=True)
            return value
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or time.monotonic() >= deadline:
                raise
            attempts += 1
            time.sleep(0.01)


results.atomic_bytes = demo_atomic_bytes

if __name__ == '__main__':
    import run
    sys.argv = [str(APP / 'run.py'), '--core-root', str(CORE_ROOT)]
    raise SystemExit(run.main())
