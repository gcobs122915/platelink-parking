"""Atomic protocol publication primitives. TXT is UTF-8 without BOM, LF."""
from __future__ import annotations
import datetime as dt
import json
import math
import os
from pathlib import Path
import uuid
import threading


def now():
    return dt.datetime.now().astimezone().isoformat(timespec='milliseconds')


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8'))


def normalize_text(value):
    return ''.join(c for c in str(value).upper() if not c.isspace() and c != '-')


def text_result(plates, policy='all'):
    selected = plates
    if policy == 'largest' and plates:
        def rank(p):
            x1, y1, x2, y2 = p['box']
            return ((x2 - x1) * (y2 - y1), p['detector_confidence'])
        selected = [max(plates, key=rank)]
    lines = []
    for plate in selected:
        text = plate['text']
        score = float(plate['ocr_confidence'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid OCR confidence')
        if not text or any(c in text for c in '|\r\n'):
            lines.append('false|######|0\n')
        else:
            lines.append(f"{str(bool(plate['accepted'])).lower()}|{text}|{score:.6f}\n")
    return ''.join(lines) if lines else 'false|######|0\n'


class EventLog:
    def __init__(self, state_dir):
        self.path = Path(state_dir) / 'events.jsonl'
        self.lock = threading.Lock()

    def __call__(self, event, message='', **fields):
        timestamp = now()
        record = {'at': timestamp, 'event': event, 'message': message, **fields}
        with self.lock:
            print(f'[{timestamp[11:23]}] [{event}] {message}', flush=True)
            with self.path.open('a', encoding='utf-8', newline='\n') as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
