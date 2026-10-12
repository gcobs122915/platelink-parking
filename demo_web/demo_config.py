"""Portable demo paths; runtime writes stay in this demo's own directory."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / 'runtime'
CORE_ROOT = RUNTIME / 'core'
SAMPLE_ROOT = ROOT / 'samples'
DEFAULTS = {'app_dir': '../車牌辨識核心/app', 'port': 8766,
            'samples': {'military': 'military.jpg', 'diplomatic': 'diplomatic.jpg', 'foreign': 'foreign.jpg'}}


def load_config():
    path = ROOT / 'demo.config.json'
    supplied = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError('demo.config.json contains unknown settings')
    value = {**DEFAULTS, **supplied}
    if type(value['port']) is not int or not 1024 <= value['port'] <= 65535:
        raise ValueError('port must be an integer from 1024 to 65535')
    if not isinstance(value['app_dir'], str) or not value['app_dir'].strip():
        raise ValueError('app_dir must identify the existing core app directory')
    app = (ROOT / value['app_dir']).resolve()
    if not (app / 'client.py').is_file() or not (app / 'settings.py').is_file():
        raise ValueError('Core app_dir is incomplete; preserve the repository sibling layout')
    if not isinstance(value['samples'], dict) or set(value['samples']) != set(DEFAULTS['samples']):
        raise ValueError('samples must map military, diplomatic and foreign to local JPG filenames')
    samples = {}
    for key, filename in value['samples'].items():
        if not isinstance(filename, str):
            raise ValueError('Sample filenames must be strings')
        target = (SAMPLE_ROOT / filename).resolve()
        if not target.is_relative_to(SAMPLE_ROOT.resolve()) or target.suffix.lower() not in ('.jpg', '.jpeg'):
            raise ValueError('Samples must be JPEG files inside demo_web/samples')
        samples[key] = target
    return {'app': app, 'port': value['port'], 'samples': samples}


def prepare_runtime():
    CORE_ROOT.mkdir(parents=True, exist_ok=True)
    config = CORE_ROOT / 'config.json'
    if not config.exists():
        # Fixed relative directories prevent this wrapper writing into a live core.
        config.write_text(json.dumps({
            'input_dir': 'input', 'result_dir': 'input', 'details_dir': 'output/details',
            'yolo_dir': 'output/yolo', 'state_dir': 'state', 'backend': 'hailo-all',
            'ocr_vocabulary': 'plate41', 'result_policy': 'all', 'save_yolo_image': True,
            'max_inflight_images': 1, 'max_queued_jobs': 12,
        }, indent=2), encoding='utf-8')
    data = json.loads(config.read_text(encoding='utf-8-sig'))
    for name in ('input_dir', 'result_dir', 'details_dir', 'yolo_dir', 'state_dir'):
        value = data.get(name)
        if not isinstance(value, str) or not (CORE_ROOT / value).resolve().is_relative_to(CORE_ROOT.resolve()):
            raise ValueError('Demo core paths must remain inside demo_web/runtime/core')
    return CORE_ROOT
