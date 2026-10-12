"""Local file submission helper; no inference or network calls."""
import argparse
import datetime as dt
import hashlib
import os
from pathlib import Path
import sys
import uuid
import json
from results import atomic_bytes
from settings import IMAGE_SUFFIXES, load_settings


def check_ready(settings):
    if (Path(settings['state_dir'])/'STOP').exists():
        raise RuntimeError('STOP 旗標存在；請先移除再啟動核心')
    status = json.loads((Path(settings['state_dir'])/'status.json').read_text(encoding='utf-8'))
    age = (dt.datetime.now().astimezone()-dt.datetime.fromisoformat(status['at'])).total_seconds()
    if status['state'] != 'running' or not status.get('input_ready') or not 0 <= age < 20:
        raise RuntimeError('核心未就緒或心跳已過期；請先開 start.cmd，等待 [INPUT_READY]')
    # A fresh heartbeat must also belong to a process still alive.
    pid = int(status['pid'])
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.GetExitCodeProcess.restype = wintypes.BOOL
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = api.OpenProcess(0x1000, False, pid)
        if not handle:
            raise RuntimeError('核心程序已離線')
        try:
            code = wintypes.DWORD()
            if not api.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 259:
                raise RuntimeError('核心程序已離線')
        finally:
            api.CloseHandle(handle)
    else:
        os.kill(pid, 0)


def submit(source, settings):
    source = Path(source).resolve()
    if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f'不支援的照片路徑：{source}')
    if source.stat().st_size > settings['max_file_bytes']:
        raise ValueError('照片超過大小上限')
    with source.open('rb') as stream:
        contents = stream.read(settings['max_file_bytes']+1)
    if len(contents) > settings['max_file_bytes']:
        raise ValueError('照片超過大小上限')
    basename = 'manual_'+uuid.uuid4().hex
    target = Path(settings['input_dir'])/(basename+source.suffix.lower())
    upload = target.with_name(target.name+'.upload')
    with upload.open('xb') as stream:
        stream.write(contents)
        stream.flush()
        os.fsync(stream.fileno())
    upload.rename(target)
    flag = target.parent/(basename+'Complete.txt')
    with flag.open('xb') as stream:
        stream.flush()
        os.fsync(stream.fileno())
    return dict(basename=basename, image=str(target), source_sha256=hashlib.sha256(contents).hexdigest(),
                result_txt=str(Path(settings['result_dir'])/(basename+'Result.txt')),
                result_json=str(Path(settings['details_dir'])/(basename+'Result.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photos', type=Path, nargs='*')
    parser.add_argument('--core-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    try:
        settings = load_settings(args.core_root)
        if args.stop:
            path = Path(settings['state_dir'])/'STOP'
            atomic_bytes(path, b'')
            print(f'已送出停止旗標：{path}\n核心會完成目前工作再停止；下次啟動前請刪除此 STOP。', flush=True)
            return 0
        if not args.photos:
            print('先開 start.cmd，再把一張或多張照片拖到 send.cmd。\n'
                  '或手動將 demo001.jpg 放進 input，最後建立 demo001Complete.txt。', flush=True)
            return 0
        check_ready(settings)
        for path in args.photos:
            result = submit(path, settings)
            print(f"[SUBMITTED] {path.name}\n照片：{result['image']}\n"
                  f"車號：{result['result_txt']}\n明細：{result['result_json']}", flush=True)
        return 0
    except Exception as exc:
        print(f'[ERROR] {exc}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
