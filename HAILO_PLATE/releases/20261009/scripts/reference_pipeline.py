"""CPU reference for the exact 2026-10-09 bundle. This never uses the accelerator.

Run in the existing WSL torch environment with --ocr-python pointing to the
existing Paddle environment. Imports of torch/Paddle are lazy so preprocessing
and contract tests do not require both frameworks in one Python installation.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

RELEASE = Path(__file__).resolve().parents[1]
WORKSPACE = RELEASE.parents[2]
DEFAULT_BUNDLE = WORKSPACE / "PlateRecognition_Models_20261009"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
EXPECTED_HASHES = {
    "models/yolo/yolo11l.pt": "4078b0ac1dc4d37c4fc1553393ad1e10139b09010e2d636392a399f46e4679de",
    "models/pose/yolo11l-pose.pt": "96e894abf28b8c13f4b4bd9b23d98ec929ed60b3e8ddf4d72d17b682ac676ce6",
    "models/ocr/inference.json": "cc01748691bf4118c27d8c0cf79507f19adc0b8824be2d9f63eba87713e3c750",
    "models/ocr/inference.pdiparams": "24aff12ad420fb488bd7cdff550ffcb016127d36182fabfb0496b8ec00212e28",
    "models/ocr/inference.yml": "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
    "models/ocr/pretrained_character_dict.txt": "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
}


def verify_bundle(bundle):
    records = {}
    for name, expected in EXPECTED_HASHES.items():
        path = Path(bundle) / name
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise ValueError(f"Source model SHA-256 mismatch: {path}")
        records[name] = actual
    return records


def validate_bgr(bgr):
    if (not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8
            or bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 1):
        raise ValueError("Expected a nonempty uint8 BGR image")


def pose_input(bgr, max_pixels=1048576):
    """Native pixels, RGB, 114 bottom/right padding to stride 32; no resizing."""
    validate_bgr(bgr)
    height, width = bgr.shape[:2]
    padded_h, padded_w = math.ceil(height / 32) * 32, math.ceil(width / 32) * 32
    if padded_h * padded_w > max_pixels:
        raise ValueError(f"POSE padded pixels {padded_h * padded_w} exceed {max_pixels}")
    padded = cv2.copyMakeBorder(bgr, 0, padded_h - height, 0, padded_w - width,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.


def ocr_input(bgr):
    """BGR resize to 48 high, normalize, then zero-pad on the right to 320."""
    validate_bgr(bgr)
    height, width = bgr.shape[:2]
    target_width = min(320, math.ceil(48 * width / height))
    resized = cv2.resize(bgr, (target_width, 48), interpolation=cv2.INTER_LINEAR)
    output = np.zeros((1, 3, 48, 320), dtype=np.float32)
    output[0, :, :, :target_width] = (resized.astype(np.float32).transpose(2, 0, 1) / 255. - .5) / .5
    return output


def load_characters(bundle):
    return [""] + (Path(bundle) / "models/ocr/pretrained_character_dict.txt").read_text(encoding="utf-8").splitlines() + [" "]


def decode_ctc(probabilities, characters, *, probability_sum_atol=1e-3):
    if (isinstance(probability_sum_atol, bool) or not np.isfinite(probability_sum_atol)
            or not 0 < probability_sum_atol <= .02):
        raise ValueError("OCR probability sum tolerance must satisfy 0 < atol <= 0.02")
    values = np.asarray(probabilities)
    if (values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != len(characters)
            or not np.isfinite(values).all()):
        raise ValueError(f"OCR output must be [1,T,{len(characters)}] finite probabilities; got {values.shape}")
    if (values.min() < -1e-5 or values.max() > 1.00001
            or not np.allclose(values.sum(axis=-1), 1., atol=probability_sum_atol, rtol=0)):
        raise ValueError("Expected OCR Softmax probabilities, not logits")
    ids = values[0].argmax(axis=-1)
    positions = [i for i, value in enumerate(ids) if value != 0 and (i == 0 or value != ids[i - 1])]
    scores = [float(values[0, i, ids[i]]) for i in positions]
    return {"text": "".join(characters[ids[i]] for i in positions),
            "confidence": float(np.mean(scores)) if scores else 0.,
            "character_confidences": scores}


def eligible_bbox(bbox):
    x1, y1, x2, y2 = map(float, bbox)
    width, height = x2 - x1, y2 - y1
    return min(width, height) > 30 and max(width, height) > 50


def expanded_crop(bgr, bbox):
    """Expand each side by 3% of that box dimension, outward-round, and clip."""
    x1, y1, x2, y2 = map(float, bbox)
    dx, dy = (x2 - x1) * .03, (y2 - y1) * .03
    height, width = bgr.shape[:2]
    bounds = [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
              min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]
    a, b, c, d = bounds
    return bgr[b:d, a:c].copy(), bounds


def warp_corners(bgr, points):
    """TL/TR/BR/BL geometric transform, with explicit raw-crop fallback."""
    validate_bgr(bgr)
    corners = np.asarray(points, dtype=np.float32)
    def fallback(reason):
        return bgr.copy(), {"applied": False, "reason": reason, "corner_order": "TL,TR,BR,BL"}
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return fallback("invalid_corner_shape_or_values")
    height, width = bgr.shape[:2]
    if (np.any(corners[:, 0] < 0) or np.any(corners[:, 0] > width - 1)
            or np.any(corners[:, 1] < 0) or np.any(corners[:, 1] > height - 1)):
        return fallback("corners_outside_native_crop")
    if not cv2.isContourConvex(corners) or cv2.contourArea(corners, oriented=True) <= 1.:
        return fallback("nonconvex_degenerate_or_reversed_corners")
    tl, tr, br, bl = corners
    # Reject an upside-down/reflected semantic order rather than guessing a reorder.
    if (float(tl[1] + tr[1]) >= float(bl[1] + br[1])
            or float(tl[0] + bl[0]) >= float(tr[0] + br[0])):
        return fallback("invalid_tl_tr_br_bl_order")
    out_w = max(2, round(max(float(np.linalg.norm(tr - tl)), float(np.linalg.norm(br - bl)))))
    out_h = max(2, round(max(float(np.linalg.norm(bl - tl)), float(np.linalg.norm(br - tr)))))
    if out_w * out_h > 1048576:
        return fallback("warp_pixel_limit")
    target = np.asarray([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    matrix = cv2.getPerspectiveTransform(corners, target)
    if not np.isfinite(matrix).all() or abs(float(np.linalg.det(matrix))) < 1e-10:
        return fallback("singular_perspective_matrix")
    corrected = cv2.warpPerspective(bgr, matrix, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    return corrected, {"applied": True, "reason": "pose_four_corners", "corner_order": "TL,TR,BR,BL",
                       "output_wh": [out_w, out_h], "matrix": matrix.tolist()}


def recognize_with_retry(ocr, corrected, raw, rectified):
    first = ocr.recognize(corrected)
    attempts = [{"source": "rectified" if rectified else "raw", **first}]
    if rectified and first["confidence"] < .95:
        attempts.append({"source": "raw_retry", **ocr.recognize(raw)})
    best = max(attempts, key=lambda row: row["confidence"])
    return {**best, "accepted": bool(best["text"] and best["confidence"] >= .5),
            "attempts": attempts, "min_confidence": .5, "retry_threshold": .95}


class PaddleOcr:
    def __init__(self, bundle, threads=4):
        import paddle
        source = Path(bundle) / "models/ocr"
        config = paddle.inference.Config(str(source / "inference.json"), str(source / "inference.pdiparams"))
        config.disable_gpu()
        config.disable_glog_info()
        config.disable_mkldnn()
        config.set_cpu_math_library_num_threads(threads)
        config.switch_ir_optim(False)
        self.predictor = paddle.inference.create_predictor(config)
        names = self.predictor.get_input_names()
        if len(names) != 1 or len(self.predictor.get_output_names()) != 1:
            raise ValueError("Expected one OCR input and one output")
        self.input = self.predictor.get_input_handle(names[0])
        self.output_name = self.predictor.get_output_names()[0]
        self.characters = load_characters(bundle)
        self.version = paddle.__version__

    def infer(self, tensor):
        if tensor.shape != (1, 3, 48, 320) or tensor.dtype != np.float32:
            raise ValueError("Expected OCR tensor [1,3,48,320] float32")
        self.input.reshape(tensor.shape)
        self.input.copy_from_cpu(tensor)
        self.predictor.run()
        output = self.predictor.get_output_handle(self.output_name).copy_to_cpu()
        return decode_ctc(output, self.characters)

    def recognize(self, bgr):
        return self.infer(ocr_input(bgr))

    def close(self):
        pass


class OcrWorker:
    """Keep Paddle resident in a separate existing Python, without installing packages."""
    def __init__(self, python, bundle, threads=4):
        self.process = subprocess.Popen([str(python), "-u", str(Path(__file__).resolve()), "--ocr-worker",
                                         "--bundle", str(bundle), "--threads", str(threads)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                        encoding="utf-8", bufsize=1)
        ready = self._receive()
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(ready.get("error", "Paddle worker failed to start"))
        self.version = ready["paddle_version"]

    def _receive(self):
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"Paddle worker exited with {self.process.poll()}")
            if line.startswith("REFERENCE_RESULT "):
                response = json.loads(line[len("REFERENCE_RESULT "):])
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response
            print(line.rstrip(), file=sys.stderr)

    def recognize(self, bgr):
        payload = base64.b64encode(ocr_input(bgr).tobytes()).decode("ascii")
        self.process.stdin.write(json.dumps({"tensor": payload}) + "\n")
        self.process.stdin.flush()
        return self._receive()

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        self.process.stdout.close()


def ocr_worker(bundle, threads):
    def send(value):
        print("REFERENCE_RESULT " + json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)
    try:
        ocr = PaddleOcr(bundle, threads)
        send({"ready": True, "paddle_version": ocr.version})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                tensor = np.frombuffer(base64.b64decode(request["tensor"], validate=True), dtype=np.float32).reshape(1, 3, 48, 320)
                send(ocr.infer(tensor))
            except Exception as exc:
                send({"error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        send({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    return 0


class ReferencePipeline:
    backend = "CPU_REFERENCE"

    def __init__(self, bundle=DEFAULT_BUNDLE, ocr_python=None, threads=4, runtime_dir=None):
        self.bundle = Path(bundle).resolve()
        self.hashes = verify_bundle(self.bundle)
        runtime_dir = Path(runtime_dir or RELEASE / "results/reference_runtime")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(runtime_dir))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        import torch
        import ultralytics
        from ultralytics import YOLO
        torch.set_num_threads(threads)
        self.torch = torch
        self.detector = YOLO(str(self.bundle / "models/yolo/yolo11l.pt"))
        self.pose = YOLO(str(self.bundle / "models/pose/yolo11l-pose.pt"))
        if list(self.pose.model.model[-1].kpt_shape) != [4, 2]:
            raise ValueError("POSE must have kpt_shape=[4,2], TL/TR/BR/BL with no point confidences")
        self.pose.model.float().eval().to("cpu")
        self.ocr = OcrWorker(ocr_python, self.bundle, threads) if ocr_python else PaddleOcr(self.bundle, threads)
        self.versions = {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                         "paddle": self.ocr.version, "opencv": cv2.__version__, "numpy": np.__version__}

    def corners(self, crop):
        try:
            from ultralytics.utils.nms import non_max_suppression
        except ImportError:  # Ultralytics 8.3 keeps NMS in ops.
            from ultralytics.utils.ops import non_max_suppression
        try:
            tensor = pose_input(crop)
        except ValueError as exc:
            return None, {"valid": False, "reason": str(exc)}
        with self.torch.inference_mode():
            output = self.pose.model(self.torch.from_numpy(tensor))
            prediction = output[0] if isinstance(output, tuple) else output
            detections = non_max_suppression(prediction, conf_thres=.25, iou_thres=.5,
                                             nc=len(self.pose.names), max_det=1)[0]
        if len(detections) == 0:
            return None, {"valid": False, "reason": "no_pose_detection", "input_nchw": list(tensor.shape)}
        row = detections[0].cpu().numpy()
        points = row[6:].reshape(4, 2)
        return points, {"valid": True, "confidence": float(row[4]), "input_nchw": list(tensor.shape),
                        "native_crop_wh": [crop.shape[1], crop.shape[0]], "corners_crop_xy": points.tolist(),
                        "per_corner_confidence": None}

    def process(self, bgr, filename="image"):
        validate_bgr(bgr)
        started = time.perf_counter()
        boxes, scores, classes = self.detect(bgr)
        detected = time.perf_counter()
        return self.process_detections(bgr, boxes, scores, classes, filename, started, detected)

    def detect(self, bgr):
        result = self.detector.predict(bgr, imgsz=1280, rect=True, half=False, conf=.25, iou=.7,
                                       max_det=50, device="cpu", verbose=False, save=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        return boxes, scores, classes

    def process_detections(self, bgr, boxes, scores, classes, filename, started, detected):
        plates, crops, rejected = [], [], []
        for bbox, score, class_id in zip(boxes, scores, classes):
            if not eligible_bbox(bbox):
                rejected.append({"bbox_xyxy": bbox.tolist(), "reason": "strict_size_filter_short_gt30_long_gt50"})
                continue
            crop, bounds = expanded_crop(bgr, bbox)
            pose_start = time.perf_counter()
            points, pose_info = self.corners(crop)
            if points is None:
                corrected = crop.copy()
                geometry = {"applied": False, "reason": pose_info["reason"], "corner_order": "TL,TR,BR,BL"}
            else:
                corrected, geometry = warp_corners(crop, points)
            pose_ms = (time.perf_counter() - pose_start) * 1000
            ocr_start = time.perf_counter()
            recognized = recognize_with_retry(self.ocr, corrected, crop, geometry["applied"])
            ocr_ms = (time.perf_counter() - ocr_start) * 1000
            plates.append({"index": len(plates) + 1, "bbox_xyxy": bbox.tolist(), "detector_confidence": float(score),
                           "class_id": int(class_id), "crop_xyxy": bounds, "pose": pose_info,
                           "rectification": geometry, "ocr": recognized,
                           "timing_ms": {"pose_and_warp_cpu": pose_ms, "ocr_with_retry_cpu": ocr_ms}})
            crops.append((crop, corrected))
        return {"filename": filename, "backend": self.backend, "accelerator_used": False,
                "image_wh": [bgr.shape[1], bgr.shape[0]], "plates": plates, "rejected_detections": rejected,
                "timing_ms": {"yolo_cpu": (detected - started) * 1000,
                              "total_cpu": (time.perf_counter() - started) * 1000}}, crops

    def close(self):
        self.ocr.close()


def write_image(path, bgr):
    ok, encoded = cv2.imencode(Path(path).suffix, bgr)
    if not ok:
        raise ValueError(f"Unable to encode image: {path}")
    Path(path).write_bytes(encoded.tobytes())


def collect_images(source):
    source = Path(source).resolve()
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Input file must be a supported image")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", help="Image file or folder (non-recursive)")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output", type=Path, default=RELEASE / "results/cpu_reference")
    parser.add_argument("--ocr-python", type=Path, help="Separate Python with Paddle installed; kept resident")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Maximum files; 0 means all")
    parser.add_argument("--ocr-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.threads <= 32:
        parser.error("--threads must be 1..32")
    if args.ocr_worker:
        return ocr_worker(args.bundle, args.threads)
    if args.input is None:
        parser.error("input is required")
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    files = collect_images(args.input)
    if args.limit:
        files = files[:args.limit]
    if not files:
        parser.error("No images found")
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    start = time.perf_counter()
    pipeline = ReferencePipeline(args.bundle, args.ocr_python, args.threads, args.output / "runtime_settings")
    report = {"backend": "CPU_REFERENCE", "accelerator_used": False, "accelerator_deployment_complete": False,
              "source_bundle": str(args.bundle.resolve()), "source_sha256": pipeline.hashes, "versions": pipeline.versions,
              "startup_ms": (time.perf_counter() - start) * 1000,
              "geometry_policy": "Reconstructed where the model-only README omits details: expand bbox by 3% per side, floor/ceil outward, clip image; warp width/height use rounded max opposite-edge lengths; reject out-of-crop, nonconvex, reversed or upside-down semantic corners; invalid pose falls back to raw crop. Exact original application parity is not established.",
              "accuracy_validated": False, "images": []}
    errors = 0
    try:
        for index, source in enumerate(files, 1):
            try:
                bgr = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                result, crops = pipeline.process(bgr, source.name)
                directory = args.output / f"{index:04d}_{source.stem}"
                directory.mkdir(parents=True, exist_ok=True)
                annotated = bgr.copy()
                for plate, (raw, corrected) in zip(result["plates"], crops):
                    number = plate["index"]
                    raw_path, rect_path = directory / f"plate_{number:02d}_raw.png", directory / f"plate_{number:02d}_rectified.png"
                    write_image(raw_path, raw)
                    write_image(rect_path, corrected)
                    plate["artifacts"] = {"raw_crop": str(raw_path.resolve()), "rectified_crop": str(rect_path.resolve())}
                    x1, y1, x2, y2 = map(round, plate["bbox_xyxy"])
                    color = (60, 200, 80) if plate["ocr"]["accepted"] else (0, 180, 255)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"#{number} OCR {plate['ocr']['confidence']:.3f}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                annotated_path = directory / "annotated.jpg"
                write_image(annotated_path, annotated)
                result["annotated_image"] = str(annotated_path.resolve())
                (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
                report["images"].append(result)
                print(json.dumps({"image": source.name, "backend": "CPU_REFERENCE", "plates": [{"text": p["ocr"]["text"], "confidence": p["ocr"]["confidence"], "accepted": p["ocr"]["accepted"]} for p in result["plates"]]}, ensure_ascii=False), flush=True)
            except Exception as exc:
                errors += 1
                report["images"].append({"filename": source.name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"ERROR {source.name}: {exc}", file=sys.stderr, flush=True)
    finally:
        pipeline.close()
        report["status"] = "PASS" if errors == 0 else "FAILED"
        report["error_count"] = errors
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"CPU reference report: {args.output / 'report.json'}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
