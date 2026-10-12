"""File-exchange settings; paths are relative to the core, never to the cwd."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

IMAGE_SUFFIXES = {'.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp'}
COMPLETE = 'Complete.txt'
CORE_VERSION = 'usb-file/20261011-pose-ranges-v1'
DEFAULTS = dict(input_dir='input', result_dir='input', details_dir='output/details',
                yolo_dir='output/yolo', state_dir='state', backend='hailo-all',
                ocr_vocabulary='plate41', result_policy='all', ocr_min_confidence=.5,
                ocr_retry_threshold=.95, save_yolo_image=False,
                max_file_bytes=104857600, max_pixels=40000000,
                scan_interval_seconds=2., poll_interval_seconds=.25,
                heartbeat_seconds=2., max_inflight_images=4, max_queued_jobs=128,
                pose_input_policy='smallest_fit_downscale_v1')


def load_settings(root):
    root = Path(root).resolve()
    supplied = json.loads((root / 'config.json').read_text(encoding='utf-8-sig'))
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError('config.json contains unknown settings')
    values = {**DEFAULTS, **supplied}
    if values['backend'] != 'hailo-all' or values['ocr_vocabulary'] != 'plate41':
        raise ValueError('This core requires all-USB inference and the plate41 model')
    if values['pose_input_policy'] != 'smallest_fit_downscale_v1':
        raise ValueError('Unsupported POSE range policy')
    for key, limit in (('max_inflight_images', 8), ('max_queued_jobs', 4096)):
        if type(values[key]) is not int or not 1 <= values[key] <= limit:
            raise ValueError(f'{key} must be an integer between 1 and {limit}')
    if values['result_policy'] not in {'all', 'largest'}:
        raise ValueError('result_policy must be all or largest')
    for key in ('ocr_min_confidence', 'ocr_retry_threshold'):
        if type(values[key]) not in (int, float) or not 0 <= values[key] <= 1:
            raise ValueError(f'{key} must be between 0 and 1')
    for key, upper in (('max_file_bytes', 104857600), ('max_pixels', 40000000)):
        if type(values[key]) is not int or not 0 < values[key] <= upper:
            raise ValueError(f'{key} must be a positive integer <= {upper}')
    for key in ('scan_interval_seconds', 'poll_interval_seconds', 'heartbeat_seconds'):
        if type(values[key]) not in (int, float) or not .01 <= values[key] <= 5:
            raise ValueError(f'{key} must be 0.01..5 seconds')
    if type(values['save_yolo_image']) is not bool:
        raise ValueError('save_yolo_image must be true or false')
    for key in ('input_dir', 'result_dir', 'details_dir', 'yolo_dir', 'state_dir'):
        if not isinstance(values[key], str) or not values[key].strip():
            raise ValueError(f'{key} must be a directory path')
        values[key] = str((root / values[key]).resolve())
        Path(values[key]).mkdir(parents=True, exist_ok=True)
    values['core_root'] = str(root)
    return values


def identity(settings, models):
    data = json.dumps({'settings': settings, 'models': models, 'core_version': CORE_VERSION},
                      ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return 'usb-file-20261011-' + hashlib.sha256(data).hexdigest()


def valid_basename(value):
    if not value or value in {'.', '..'} or value[-1] in ' .':
        return False
    if any(ord(c) < 32 or c in '<>:"/\\|?*' for c in value):
        return False
    reserved = {'CON', 'PRN', 'AUX', 'NUL', 'CONIN$', 'CONOUT$'}
    reserved.update(f'{prefix}{number}' for prefix in ('COM', 'LPT') for number in '123456789¹²³')
    return value.split('.')[0].upper() not in reserved


def basename_from_flag(name):
    if not name.lower().endswith(COMPLETE.lower()):
        return None
    value = name[:-len(COMPLETE)]
    if not valid_basename(value):
        raise ValueError('Complete flag requires a valid, nonempty basename')
    return value
