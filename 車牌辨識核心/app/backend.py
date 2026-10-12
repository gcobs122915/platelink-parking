"""File-protocol adapter around the validated, resident all-USB model primitives."""
from __future__ import annotations
import hashlib
import io
import math
from pathlib import Path
import sys
import time
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps

SCRIPTS = Path(__file__).resolve().parents[2] / 'HAILO_PLATE/releases/20261009/scripts'
sys.path.insert(0, str(SCRIPTS))
from reference_pipeline import eligible_bbox, expanded_crop, warp_corners
from results import normalize_text


class InputError(ValueError):
    def __init__(self, code, message, source_sha256=None):
        super().__init__(message)
        self.code, self.source_sha256 = code, source_sha256


def read_photo(path, settings):
    path = Path(path)
    if path.stat().st_size > settings['max_file_bytes']:
        raise InputError('image_file_too_large', '圖片超過檔案大小上限')
    with path.open('rb') as stream:
        contents = stream.read(settings['max_file_bytes']+1)
    if len(contents) > settings['max_file_bytes']:
        raise InputError('image_file_too_large', '圖片超過檔案大小上限')
    digest = hashlib.sha256(contents).hexdigest()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                if image.width * image.height > settings['max_pixels']:
                    raise ValueError('圖片超過像素上限')
                # EXIF changes coordinates, never the hash of the original bytes.
                orientation = image.getexif().get(274, 1)
                corrected = ImageOps.exif_transpose(image)
                pixels = np.asarray(corrected.convert('RGB'))
                bgr = np.ascontiguousarray(pixels[:, :, ::-1])
    except Exception as exc:
        raise InputError('image_decode_failed', f'{type(exc).__name__}: {exc}', digest) from exc
    return bgr, digest, {'exif_orientation': orientation, 'exif_corrected': orientation in range(2, 9)}


def ocr_candidate(value, source):
    score = float(value['confidence'])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError('OCR returned invalid confidence')
    raw = str(value['text'])
    return {'raw_text': raw, 'text': normalize_text(raw), 'ocr_confidence': score, 'source': source}


def usable(text):
    return bool(text) and not any(c in text for c in '|\r\n')


def recognize_crop(ocr, corrected, raw, rectified, settings):
    first = ocr_candidate(ocr.recognize(corrected), 'rectified' if rectified else 'original_crop')
    candidates, chosen, notices = [first], first, []
    retry = {'attempted': False, 'succeeded': None, 'threshold': settings['ocr_retry_threshold']}
    if rectified and first['ocr_confidence'] < settings['ocr_retry_threshold']:
        retry['attempted'] = True
        try:
            second = ocr_candidate(ocr.recognize(raw), 'original_crop')
            candidates.append(second)
            retry['succeeded'] = True
            if usable(second['text']) and second['ocr_confidence'] > first['ocr_confidence']:
                chosen = second
        except Exception as exc:
            retry.update(succeeded=False, error=f'{type(exc).__name__}: {exc}')
            notices.append({'stage': 'ocr_retry', 'code': 'original_crop_retry_failed', 'message': retry['error']})
    return {**chosen, 'accepted': usable(chosen['text']) and chosen['ocr_confidence'] >= settings['ocr_min_confidence'],
            'ocr_source': chosen['source'], 'ocr_candidates': candidates, 'ocr_retry': retry}, notices


class UsbRecognizer:
    def __init__(self, settings):
        from async_usb import ConcurrentUsbPipeline
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        self.settings = settings
        self.pipeline = ConcurrentUsbPipeline()
        self.models = [dict(name=m['key'], stage=m['stage'], hef_sha256=m['hef_sha256'],
                            source_model_sha256=m['source_model_sha256'], profile_hw=m.get('profile_hw'),
                            input_policy=settings['pose_input_policy'] if m['stage']=='pose' else m.get('preprocess'))
                       for m in self.pipeline.manifest['models']]
        self.vocabulary = self.pipeline.ocr_vocabulary_metadata

    def infer(self, bgr, filename):
        start = time.perf_counter_ns()
        pipeline = self.pipeline
        pipeline.neural_stage_runs = []
        pipeline._last_accelerated = False
        result = dict(detections=[], plates=[], errors=[], warnings=[], timing={},
                      backend='HAILO_USB_ALL', cpu_neural_fallback=False,
                      execution_providers={}, ocr_vocabulary=self.vocabulary)
        try:
            boxes, scores, classes = pipeline.detect(bgr)
        except Exception as exc:
            result['errors'].append({'stage': 'yolo', 'code': 'detection_failed', 'message': f'{type(exc).__name__}: {exc}'})
            boxes, scores, classes = [], [], []
        result['timing']['yolo_ms'] = (time.perf_counter_ns()-start)/1e6
        for index, (bbox, score, class_id) in enumerate(zip(boxes, scores, classes), 1):
            box = [float(v) for v in bbox]
            sides = [box[2]-box[0], box[3]-box[1]]
            eligible = eligible_bbox(bbox)
            result['detections'].append(dict(index=index, box=box, detector_confidence=float(score),
                class_id=int(class_id), short_side=min(sides), long_side=max(sides),
                accepted_size=bool(eligible), exclusion_reason=None if eligible else 'short_side_must_exceed_30_and_long_side_50'))
            if not eligible:
                continue
            stage = 'crop'
            try:
                raw, bounds = expanded_crop(bgr, bbox)
                pose_start = time.perf_counter_ns()
                stage = 'pose'
                # Variable crop sizes are fitted to the resident POSE profiles.
                # A successfully run POSE with invalid geometry uses raw pixels.
                points, pose = pipeline.corners(raw)
                if points is None:
                    corrected, geometry = raw, {'applied': False, 'reason': pose['reason']}
                else:
                    corrected, geometry = warp_corners(raw, points)
                pose_ms = (time.perf_counter_ns()-pose_start)/1e6
                if not geometry['applied']:
                    result['warnings'].append(dict(stage='pose', detection_index=index,
                        code='original_crop_fallback', message=geometry['reason']))
                stage = 'ocr'
                ocr_start = time.perf_counter_ns()
                recognized, notices = recognize_crop(pipeline.ocr, corrected, raw, geometry['applied'], self.settings)
                ocr_ms = (time.perf_counter_ns()-ocr_start)/1e6
                result['warnings'].extend({**n, 'detection_index': index} for n in notices)
                crop_points = points.tolist() if points is not None else None
                original_points = [[float(x+bounds[0]), float(y+bounds[1])] for x,y in crop_points] if crop_points else None
                result['plates'].append(dict(index=index, box=box, crop_bounds=bounds,
                    detector_confidence=float(score), corners=original_points,
                    rectifier=dict(used=geometry['applied'], method='pose_perspective' if geometry['applied'] else 'original_crop',
                        points=crop_points, corner_order='TL,TR,BR,BL', geometry=geometry,
                        input_transform=pose.get('input_transform'),
                        pose_detection_confidence=pose.get('confidence'), per_corner_confidence=None),
                    **recognized, timing=dict(pose_and_warp_ms=pose_ms, ocr_with_retry_ms=ocr_ms)))
            except Exception as exc:
                result['errors'].append(dict(stage=stage, detection_index=index,
                    code='unsupported_profile' if 'not compiled' in str(exc) else 'plate_processing_failed',
                    message=f'{type(exc).__name__}: {exc}'))
        events = [dict(event) for event in pipeline.neural_stage_runs]
        if any(e.get('backend') != 'HAILO_USB' or e.get('status') != 'COMPLETED' or e.get('cpu_neural_fallback') is not False for e in events):
            raise RuntimeError('Actual neural execution is not a completed USB operation')
        result['neural_stage_runs'] = events
        result['devices'] = {stage: {'configured_backend': 'HAILO_USB', 'device': pipeline.engine.device_id,
            'completed_runs': sum(e['stage'] == stage for e in events), 'cpu_neural_fallback': False}
            for stage in ('yolo', 'pose', 'ocr')}
        result['shared_engine_id'] = pipeline.engine.engine_id
        result['models_reused'] = True
        result['timing']['inference_and_geometry_ms'] = (time.perf_counter_ns()-start)/1e6
        result['status'] = ('partial' if result['plates'] else 'error') if result['errors'] else 'ok'
        return result

    def annotate(self, bgr, result):
        canvas = bgr.copy()
        for detection in result['detections']:
            x1,y1,x2,y2 = map(round, detection['box'])
            cv2.rectangle(canvas, (x1,y1), (x2,y2), (60,200,80), 2)
        for plate in result['plates']:
            if plate['corners']:
                cv2.polylines(canvas, [np.asarray(plate['corners'], np.int32)], True, (255,160,40), 2)
        ok, encoded = cv2.imencode('.jpg', canvas)
        if not ok:
            raise ValueError('Unable to encode YOLO annotation')
        return encoded.tobytes()

    def close(self):
        self.pipeline.close()

    def metrics(self):
        return {**self.pipeline.engine.metrics(), 'pose_input_policy': self.settings['pose_input_policy'],
                'pose_profiles_hw': [list(p) for p in sorted(self.pipeline.pose_specs)]}
