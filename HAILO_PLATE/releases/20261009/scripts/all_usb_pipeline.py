"""All neural models on Hailo USB; image geometry and text decoding stay on host.

This backend cannot start without the full, verified HEF manifest. Exact native
POSE profiles are selected by right/bottom stride padding only. Unsupported
profiles fail; no ONNX/PyTorch/Paddle session or CPU neural fallback exists here.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

from hailo_shared_engine import HailoSharedEngine, ModelSpec
from reference_pipeline import (DEFAULT_BUNDLE, RELEASE, ReferencePipeline, decode_ctc,
                                ocr_input, validate_bgr, verify_bundle)
from ocr_vocabulary import load_ocr_vocabulary
from ocr_usb_metadata import verified_prepared_graph

sys.path.insert(0, str(RELEASE / "hailo_scripts"))
import pose_raw
import yolo_raw

DEFAULT_MANIFEST = RELEASE / "compiler/all_usb_manifest.json"


def load_all_usb_manifest(path, source_hashes, vocabulary=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"全 USB 模型尚未備妥：需要 YOLO、POSE 與完整 OCR 的 HEF；不會退回 CPU。\nAll-USB deployment is not ready: missing {path}. YOLO, every declared native POSE profile and complete OCR HEFs are required; no CPU fallback.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    vocabulary = vocabulary or load_ocr_vocabulary("plate41")
    if manifest.get("ocr_vocabulary") != vocabulary.name or vocabulary.name != "plate41":
        raise ValueError("All-USB deployment requires the user-selected plate41 OCR vocabulary")
    if (manifest.get("schema_version") != 1 or manifest.get("hardware_arch") != "hailo10h"
            or manifest.get("status") not in {"READY_FOR_HARDWARE_VALIDATION", "HARDWARE_VALIDATED"}
            or manifest.get("cpu_neural_fallback") is not False):
        raise ValueError("All-USB manifest is incomplete or incompatible; no CPU fallback")
    source_names = {"yolo": "models/yolo/yolo11l.pt", "pose": "models/pose/yolo11l-pose.pt",
                    "ocr": "models/ocr/inference.pdiparams"}
    specs, records = [], {}
    counts = {name: 0 for name in source_names}
    pose_profiles = set()
    for item in manifest["models"]:
        stage, key = item["stage"], item["key"]
        if stage not in counts or key in records:
            raise ValueError("All-USB model stages/keys are invalid or duplicate")
        if (item.get("neural_network_complete") is not True or item.get("cpu_neural_layers") != []
                or item.get("source_model_sha256") != source_hashes[source_names[stage]]):
            raise ValueError(f"{key}: incomplete device neural graph or wrong original weights")
        shape, dtype = tuple(item["input"]["shape"]), item["input"]["dtype"]
        hw, outputs = tuple(item["profile_hw"]), item["outputs"]
        if len(shape) != 3 or shape != (*hw, 3):
            raise ValueError(f"{key}: expected one HWC image stream")
        if stage in {"yolo", "pose"}:
            if dtype != "uint8" or item.get("preprocess") != "rgb_uint8_div255_in_hef":
                raise ValueError(f"{key}: expected raw RGB UINT8 with /255 exactly once in HEF")
            channels = (64, 1) if stage == "yolo" else (64, 1, 8)
            wanted = sorted((hw[0]//stride, hw[1]//stride, channels_n)
                            for stride in (8,16,32) for channels_n in channels)
            if sorted(tuple(value) for value in outputs.values()) != wanted:
                raise ValueError(f"{key}: raw neural heads are incomplete or have wrong shapes")
            expected_decoder = "yolo11_raw6" if stage == "yolo" else "yolo11_pose_raw9"
            if item.get("decoder") != expected_decoder:
                raise ValueError(f"{key}: unsupported raw head decoder")
            if stage == "yolo" and hw != (736,1280):
                raise ValueError("Original YOLO compiled profile must be 736x1280")
            if stage == "pose":
                if any(n < 32 or n % 32 for n in hw) or hw[0]*hw[1] > 1048576 or hw in pose_profiles:
                    raise ValueError(f"Invalid or duplicate native POSE profile: {hw}")
                pose_profiles.add(hw)
        else:
            if (shape != (48,320,3) or dtype != "float32"
                    or item.get("preprocess") != "bgr_float32_normalized_minus1_plus1_right_zero"
                    or item.get("decoder") != "ctc_softmax_tc" or len(outputs) != 1
                    or tuple(next(iter(outputs.values()))) not in {(40,vocabulary.classes), (1,40,vocabulary.classes)}
                    or item.get("vocabulary") != vocabulary.metadata):
                raise ValueError("OCR must contain the verified 42-class classifier and softmax with exact normalized BGR input")
            if any(field in item for field in ("graph_derivation", "source_onnx_sha256", "compiled_onnx_sha256")):
                validation = item.get("validation_data", {})
                verified_prepared_graph({
                    "source_onnx_sha256": item.get("source_onnx_sha256"),
                    "prepared_onnx_sha256": item.get("compiled_onnx_sha256"),
                    "graph_derivation": item.get("graph_derivation"),
                    "validation_npy_sha256": validation.get("validation_bgr_normalized.npy"),
                    "expected_npy_sha256": validation.get("onnx_expected.npy"),
                }, {"graph_derivation": item.get("graph_derivation")}, vocabulary)
        hef_path = Path(item["hef"])
        if not hef_path.is_absolute():
            hef_path = path.parent / hef_path
        if "probability_contract" in item:
            if stage != "ocr":
                raise ValueError("Only the complete OCR device graph may have a probability contract")
            from approve_ocr_probability_contract import probability_contract_metadata, sha, verify_probability_contract
            contract = item["probability_contract"]
            approval_path = (path.parent / contract["report"]).resolve()
            if contract.get("schema_version") not in {1, 2} or sha(approval_path) != contract.get("report_sha256"):
                raise ValueError("OCR probability approval report hash/schema mismatch")
            approval = verify_probability_contract(approval_path, item["hef_sha256"], full_evidence=False)
            if (any(contract.get(name) != value for name, value in probability_contract_metadata(approval).items())
                    or approval.get("source_model_sha256") != item["source_model_sha256"]
                    or approval.get("ocr_vocabulary_metadata") != vocabulary.metadata):
                raise ValueError("OCR manifest probability tolerance/source differs from approval")
        specs.append(ModelSpec(key, stage, hef_path, item["hef_sha256"], shape, dtype,
                               {name: tuple(value) for name,value in outputs.items()}, hw, item["input"].get("name")))
        records[key] = item
        counts[stage] += 1
    if counts["yolo"] != 1 or counts["ocr"] != 1 or counts["pose"] < 1:
        raise ValueError("All-USB needs exactly one YOLO, one full OCR and a native POSE profile bank")
    required = {tuple(hw) for hw in manifest.get("required_pose_profiles", [])}
    if not required.issubset(pose_profiles):
        raise ValueError("The declared native POSE profile bank is incomplete")
    return manifest, specs, records


class HailoOcr:
    def __init__(self, owner, spec, bundle):
        self.owner, self.spec = owner, spec
        self.characters = owner.vocabulary.characters
        self.probability_contract = owner.model_records[spec.key].get("probability_contract")
        self.probability_sum_atol = (self.probability_contract["row_sum_atol"] if self.probability_contract else 1e-3)

    def recognize(self, bgr):
        pixels = np.ascontiguousarray(ocr_input(bgr)[0].transpose(1,2,0))
        outputs, event = self.owner._run(self.spec, pixels)
        probabilities = next(iter(outputs.values())).reshape(1,40,len(self.characters))
        result = decode_ctc(probabilities, self.characters, probability_sum_atol=self.probability_sum_atol)
        result.update(neural_backend="HAILO_USB", neural_run_seq=event["run_seq"],
                      probability_sum_atol=self.probability_sum_atol,
                      probability_contract=self.probability_contract,
                      probability_row_sum_range=[float(probabilities.sum(axis=-1).min()),
                                                 float(probabilities.sum(axis=-1).max())])
        return result

    def close(self):
        pass  # The owner closes the one shared engine after all model users.


class AllUsbPipeline(ReferencePipeline):
    backend = "HAILO_USB_ALL"

    def __init__(self, bundle=DEFAULT_BUNDLE, manifest_path=None, engine_factory=HailoSharedEngine,
                 ocr_vocabulary=None):
        self.bundle = Path(bundle).resolve()
        self._last_accelerated = False
        self.neural_stage_runs = []
        self.engine = self.ocr = None
        self.detection_adapter = None
        self.hashes = verify_bundle(self.bundle)
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST).resolve()
        self.vocabulary = load_ocr_vocabulary(ocr_vocabulary or "plate41", release=RELEASE, bundle=self.bundle)
        self.ocr_vocabulary = self.vocabulary.name
        self.ocr_vocabulary_metadata = self.vocabulary.metadata
        self.manifest, specs, self.model_records = load_all_usb_manifest(self.manifest_path, self.hashes, self.vocabulary)
        self.execution_providers = {}
        self.onnx_hashes = {}
        self.hef_hashes = {spec.key: spec.hef_sha256 for spec in specs}
        self.model_backends = {stage: "HAILO_USB" for stage in ("yolo","pose","ocr")}
        self.yolo_spec = next(spec for spec in specs if spec.stage == "yolo")
        self.ocr_spec = next(spec for spec in specs if spec.stage == "ocr")
        self.pose_specs = {spec.profile_hw: spec for spec in specs if spec.stage == "pose"}
        try:
            # All files/hashes are checked by construction before open scans USB.
            self.engine = engine_factory(specs)
            self.ocr = HailoOcr(self, self.ocr_spec, self.bundle)
            self.engine.open()
            self.versions = {"opencv": cv2.__version__, "numpy": np.__version__, **self.engine.versions}
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _run(self, spec, tensor):
        outputs, event = self.engine.run(spec.key, tensor)
        self.neural_stage_runs.append(event)
        self._last_accelerated = True
        return outputs, event

    def detect(self, bgr):
        pixels, geometry = yolo_raw.preprocess(bgr)
        if tuple(pixels.shape) != self.yolo_spec.input_shape:
            raise ValueError(f"Original YOLO rect profile {pixels.shape[:2]} is not compiled; no resizing or CPU fallback")
        outputs, _ = self._run(self.yolo_spec, pixels)
        detections = yolo_raw.original_boxes(yolo_raw.nms(yolo_raw.decode_heads(outputs)), geometry)
        return detections[:,:4], detections[:,4], np.zeros(len(detections),np.int64)

    def corners(self, crop):
        validate_bgr(crop)
        height, width = ((n+31)//32*32 for n in crop.shape[:2])
        profile = (height,width)
        if profile not in self.pose_specs:
            raise ValueError(f"Native POSE profile {profile} is not compiled. Supported: {sorted(self.pose_specs)}; no extra padding, resize or CPU fallback")
        spec = self.pose_specs[profile]
        outputs, event = self._run(spec, pose_raw.preprocess(crop,height,width))
        detections = pose_raw.nms(pose_raw.decode_heads(outputs,height,width))
        info = {"backend": "HAILO_USB", "neural_run_seq": event["run_seq"],
                "input_nchw": [1,3,height,width], "native_crop_wh": [crop.shape[1],crop.shape[0]],
                "per_corner_confidence": None}
        if not len(detections):
            return None, {**info, "valid": False, "reason": "no_pose_detection"}
        points = detections[0,5:].reshape(4,2)
        return points, {**info, "valid": True, "confidence": float(detections[0,4]), "corners_crop_xy": points.tolist()}

    def process(self, bgr, filename="image"):
        self._last_accelerated = False
        self.neural_stage_runs = []
        result, crops = super().process(bgr, filename)
        expected = {"yolo": 1, "pose": len(result["plates"]),
                    "ocr": sum(len(row["ocr"]["attempts"]) for row in result["plates"])}
        actual = {stage: sum(event["stage"] == stage for event in self.neural_stage_runs) for stage in expected}
        if actual != expected:
            raise RuntimeError(f"Neural USB audit count mismatch: {actual} vs {expected}")
        result.update(backend=self.backend, accelerator_used=self._last_accelerated,
                      ocr_vocabulary=self.ocr_vocabulary, ocr_vocabulary_metadata=self.ocr_vocabulary_metadata,
                      model_backends=dict(self.model_backends), execution_providers={},
                      all_models_accelerated=all(actual.values()), all_executed_neural_models_accelerated=True,
                      neural_stage_runs=[dict(event) for event in self.neural_stage_runs],
                      neural_stage_status={stage: {"completed_runs": count, "skipped": count == 0,
                                                 "skip_reason": "No eligible detected plates" if count == 0 else None}
                                           for stage,count in actual.items()},
                      pose_backend="HAILO_USB", ocr_backend="HAILO_USB",
                      detector={"backend": "HAILO_USB", "accelerator_used": True, "hef_sha256": self.yolo_spec.hef_sha256},
                      host_postprocessing=["image read/resize/letterbox/crop", "fixed DFL/sigmoid/grid/keypoint decode and NMS",
                                           "perspective warp", "CTC/acceptance/retry selection", "UTF-8 TXT write/flush/fsync"],
                      cpu_neural_fallback=False)
        result["timing_ms"]["yolo_hailo"] = result["timing_ms"].pop("yolo_cpu")
        result["timing_ms"]["total_pipeline"] = result["timing_ms"].pop("total_cpu")
        for row in result["plates"]:
            row["timing_ms"]["pose_and_warp"] = row["timing_ms"].pop("pose_and_warp_cpu")
            row["timing_ms"]["ocr_with_retry"] = row["timing_ms"].pop("ocr_with_retry_cpu")
        return result, crops

    def close(self):
        self.ocr = None
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.close()
