"""Exact-profile YOLO11 raw-head preprocessing/decoding, without ML frameworks."""
from __future__ import annotations

import numpy as np

HEIGHT, WIDTH = 736, 1280
STRIDES = (8, 16, 32)
END_NODES = [f"/model/model.23/cv{branch}.{level}/cv{branch}.{level}.2/Conv"
             for level in range(3) for branch in (2, 3)]
HEAD_SPECS = [(stride, channels) for stride in STRIDES for channels in (64, 1)]


def preprocess(bgr):
    """Ultralytics LetterBox(imgsz=1280,auto=True,stride=32), then BGR->RGB.

    Return uint8 NHWC pixels; division by 255 is deliberately NOT done here.
    The compiled network inserts that normalization once.
    """
    import cv2
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected nonempty uint8 BGR image")
    h, w = bgr.shape[:2]
    if min(h, w) < 1:
        raise ValueError("Empty image")
    ratio = min(1280 / h, 1280 / w)
    scaled_w, scaled_h = round(w * ratio), round(h * ratio)
    dw, dh = (1280 - scaled_w) % 32, (1280 - scaled_h) % 32
    left, right = round(dw / 2 - .1), round(dw / 2 + .1)
    top, bottom = round(dh / 2 - .1), round(dh / 2 + .1)
    resized = cv2.resize(bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR) if (scaled_w, scaled_h) != (w, h) else bgr
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(padded[:, :, ::-1]), {"ratio": ratio, "left": left, "top": top,
        "original_hw": [h, w], "profile_hw": list(padded.shape[:2])}


def ordered_heads(outputs, layout="NHWC"):
    """Order six outputs by spatial shape and channel count, not fragile HN names."""
    if layout not in {"NCHW", "NHWC"}:
        raise ValueError(layout)
    values = list(outputs.values()) if isinstance(outputs, dict) else list(outputs)
    if len(values) != 6:
        raise ValueError(f"Expected six raw heads, got {len(values)}")
    ordered = {}
    for output in values:
        value = np.asarray(output, dtype=np.float32)
        if value.ndim == 3:
            value = value[None]
        if value.ndim != 4 or value.shape[0] != 1 or not np.isfinite(value).all():
            raise ValueError(f"Invalid head shape/values: {value.shape}")
        if layout == "NHWC":
            value = value.transpose(0, 3, 1, 2)
        match = [(s, c) for s, c in HEAD_SPECS if value.shape == (1, c, HEIGHT // s, WIDTH // s)]
        if len(match) != 1 or match[0] in ordered:
            raise ValueError(f"Unexpected or duplicate raw head: {value.shape}")
        ordered[match[0]] = value
    return [ordered[spec] for spec in HEAD_SPECS]


def decode_heads(outputs, layout="NHWC"):
    """Original DFL(16 bins), sigmoid, grid decode -> [1,5,19320] xywh+score."""
    ordered = ordered_heads(outputs, layout)
    predictions = []
    bins = np.arange(16, dtype=np.float32)[None, None, :, None]
    for level, stride in enumerate(STRIDES):
        reg, cls = ordered[2 * level:2 * level + 2]
        h, w = reg.shape[2:]
        logits = reg.reshape(1, 4, 16, -1)
        shifted = logits - logits.max(axis=2, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=2, keepdims=True)
        distances = (probabilities * bins).sum(axis=2)
        yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
        anchors = np.stack([xx.ravel() + .5, yy.ravel() + .5])[None]
        top_left, bottom_right = anchors - distances[:, :2], anchors + distances[:, 2:]
        boxes = np.concatenate([(top_left + bottom_right) / 2, bottom_right - top_left], axis=1) * stride
        scores = 1 / (1 + np.exp(-np.clip(cls.reshape(1, 1, -1), -80, 80)))
        predictions.append(np.concatenate([boxes, scores], axis=1))
    return np.concatenate(predictions, axis=2)


def nms(decoded, confidence=.25, iou=.7, max_det=50):
    """Single-class Ultralytics-style continuous-coordinate NMS; output xyxy+score."""
    rows = np.asarray(decoded)[0].T
    rows = rows[rows[:, 4] > confidence]
    if not len(rows):
        return np.empty((0, 5), dtype=np.float32)
    rows = rows.copy()
    xy, wh = rows[:, :2].copy(), rows[:, 2:4].copy()
    rows[:, :2], rows[:, 2:4] = xy - wh / 2, xy + wh / 2
    order = np.argsort(-rows[:, 4], kind="stable")[:30000]
    picked = []
    while len(order) and len(picked) < max_det:
        index = order[0]
        picked.append(index)
        others = order[1:]
        intersection_wh = np.maximum(0, np.minimum(rows[index, 2:4], rows[others, 2:4]) - np.maximum(rows[index, :2], rows[others, :2]))
        intersection = intersection_wh.prod(axis=1)
        area = np.maximum(0, rows[index, 2:4] - rows[index, :2]).prod()
        areas = np.maximum(0, rows[others, 2:4] - rows[others, :2]).prod(axis=1)
        overlap = intersection / np.maximum(area + areas - intersection, 1e-12)
        order = others[overlap <= iou]
    return rows[picked]


def original_boxes(detections, metadata):
    result = np.asarray(detections).copy()
    h, w = metadata["original_hw"]
    profile_h, profile_w = metadata["profile_hw"]
    # Match Ultralytics scale_boxes' inverse-coordinate rounding, which can
    # differ by one pixel from LetterBox's physical border for some sizes.
    gain = min(profile_h / h, profile_w / w)
    pad_x = round((profile_w - w * gain) / 2 - .1)
    pad_y = round((profile_h - h * gain) / 2 - .1)
    result[:, [0, 2]] = (result[:, [0, 2]] - pad_x) / gain
    result[:, [1, 3]] = (result[:, [1, 3]] - pad_y) / gain
    result[:, [0, 2]] = result[:, [0, 2]].clip(0, w)
    result[:, [1, 3]] = result[:, [1, 3]].clip(0, h)
    return result
