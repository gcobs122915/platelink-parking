"""Original YOLO11 pose raw heads; exact native-pad profiles, no resizing."""
from __future__ import annotations

import numpy as np

STRIDES = (8, 16, 32)
HEAD_SPECS = [(s, c) for s in STRIDES for c in (64, 1, 8)]
END_NODES = [f"/model/model.23/cv{b}.{i}/cv{b}.{i}.2/Conv"
             for i in range(3) for b in (2, 3, 4)]
# Exact fixed DFL projection from the supplied pose checkpoint/validated ONNX.
# Ultralytics initializes this non-trainable projection to arange(16), but this
# checkpoint stores small FP32 deviations. Retain its actual values verbatim.
DFL_WEIGHTS = np.asarray([0.,1.,2.,3.000000238418579,4.,4.999999523162842,
    6.000000476837158,7.,8.,9.000000953674316,9.999999046325684,10.999998092651367,
    12.000000953674316,13.000005722045898,14.,14.999909400939941],dtype=np.float32)


def preprocess(bgr, height, width):
    import cv2
    if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected nonempty uint8 BGR native crop")
    h, w = bgr.shape[:2]
    native = ((h + 31) // 32 * 32, (w + 31) // 32 * 32)
    if min(h, w) < 1 or native != (height, width) or height * width > 1048576:
        raise ValueError(f"Native padded shape {native} differs from exact profile {(height, width)}")
    image = cv2.copyMakeBorder(bgr, 0, height-h, 0, width-w, cv2.BORDER_CONSTANT, value=(114,114,114))
    return np.ascontiguousarray(image[:, :, ::-1])


def ordered_heads(outputs, height, width, layout="NHWC"):
    if layout not in {"NHWC", "NCHW"}:
        raise ValueError(layout)
    values = list(outputs.values()) if isinstance(outputs, dict) else list(outputs)
    if len(values) != 9:
        raise ValueError(f"Expected nine pose heads, got {len(values)}")
    found = {}
    for output in values:
        value = np.asarray(output, dtype=np.float32)
        if value.ndim == 3:
            value = value[None]
        if value.ndim != 4 or value.shape[0] != 1 or not np.isfinite(value).all():
            raise ValueError(f"Invalid pose head {value.shape}")
        if layout == "NHWC":
            value = value.transpose(0,3,1,2)
        matches = [spec for spec in HEAD_SPECS if value.shape == (1,spec[1],height//spec[0],width//spec[0])]
        if len(matches) != 1 or matches[0] in found:
            raise ValueError(f"Unexpected or duplicate pose head {value.shape}")
        found[matches[0]] = value
    return [found[spec] for spec in HEAD_SPECS]


def decode_heads(outputs, height, width, layout="NHWC"):
    heads = ordered_heads(outputs, height, width, layout)
    predictions = []
    bins = DFL_WEIGHTS[None,None,:,None]
    for i, stride in enumerate(STRIDES):
        reg, cls, keypoints = heads[3*i:3*i+3]
        h, w = reg.shape[2:]
        logits = reg.reshape(1,4,16,-1)
        probabilities = np.exp(logits - logits.max(axis=2,keepdims=True))
        probabilities /= probabilities.sum(axis=2,keepdims=True)
        distance = (probabilities * bins).sum(axis=2)
        yy,xx = np.meshgrid(np.arange(h,dtype=np.float32),np.arange(w,dtype=np.float32),indexing="ij")
        anchors = np.stack([xx.ravel()+.5, yy.ravel()+.5])[None]
        left, right = anchors-distance[:,:2], anchors+distance[:,2:]
        boxes = np.concatenate([(left+right)/2,right-left],axis=1)*stride
        scores = 1/(1+np.exp(-np.clip(cls.reshape(1,1,-1),-80,80)))
        points = ((keypoints.reshape(1,4,2,-1)*2 + (anchors[:,None]-.5))*stride).reshape(1,8,-1)
        predictions.append(np.concatenate([boxes,scores,points],axis=1))
    return np.concatenate(predictions,axis=2)


def nms(decoded):
    """Same single-class conf .25 / IoU .5 / max_det 1 pose selection."""
    rows = np.asarray(decoded)[0].T
    rows = rows[rows[:,4] > .25]
    if not len(rows):
        return np.empty((0,13),dtype=np.float32)
    # max_det=1 makes the first confidence-sorted row the only retained row.
    row = rows[np.argsort(-rows[:,4],kind="stable")[:1]].copy()
    xy, wh = row[:,:2].copy(),row[:,2:4].copy()
    row[:,:2],row[:,2:4] = xy-wh/2,xy+wh/2
    return row
