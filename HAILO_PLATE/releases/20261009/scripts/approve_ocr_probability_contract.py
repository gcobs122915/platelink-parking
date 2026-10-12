"""Approve an explicit quantized-device probability contract from saved USB evidence.

No inference, output normalization, softmax, or editing of failed compiler reports
is performed. The original contract caps both simulator/device tolerance at 0.02.
The explicitly selected physical-device-only contract caps device tolerance at
0.01; it preserves the simulator row-sum failure and independently rechecks its
saved raw tensors for every other gate. Neither contract proves general accuracy.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath

import numpy as np

from reference_pipeline import decode_ctc
from ocr_vocabulary import load_ocr_vocabulary

REPORT_NAME = "hardware_probability_contract_report.json"
STATUS = "PASS_DEVICE_PROBABILITY_CONTRACT"
PHYSICAL_KIND = "PHYSICAL_DEVICE_SOFTMAX"
PHYSICAL_SCOPE = "physical_device_only"
ROW_SUM_ERROR = "ValueError: Complete OCR softmax probability range/row-sum contract failed; inspect recorded statistics before any tolerance decision"
PHYSICAL_POLICY = {
    "device_row_sum_atol_cap": .01, "confidence_difference_limit": .1,
    "minimum_repeats_per_sample": 3,
    "scope": "Six source-isolated numerical holdouts; shared full-pipeline validation is still required",
    "tolerance_basis": "Explicit project policy, not a vendor accuracy guarantee or a claim of an SDK bug",
    "official_sources": [
        {"path": "HAILO_PLATE/reference/dfc-5.3/hailo_dataflow_compiler_v5.3.0_user_guide.pdf",
         "pdf_pages": [8, 28], "printed_pages": [5, 25],
         "statement": "The quantized emulator approximates hardware and is not bit-exact"},
        {"path": "/opt/hailo-dfc-5.3.0/lib/python3.10/site-packages/hailo_model_optimization/acceleras/atomic_ops/activation_op.py",
         "lines": [1247, 1278],
         "statement": "Piecewise-linear numeric simulation documents potentially different rounding after the offset"},
        {"path": "/opt/hailo-dfc-5.3.0/lib/python3.10/site-packages/hailo_model_optimization/acceleras/atomic_ops/base_atomic_op.py",
         "lines": [1276, 1281],
         "statement": "Native, bit-exact, and numeric simulation modes use distinct execution paths"},
    ],
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_tolerance(value, *, physical_device_only=False):
    cap = .01 if physical_device_only else .02
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= cap:
        raise ValueError(f"An explicit finite row-sum atol in (0, {cap}] is required")
    return float(value)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_simulation(quantized, expected, characters, atol):
    if (quantized.get("status") != "FAILED" or quantized.get("strict_probability_contract_passed") is not False
            or quantized.get("ctc_and_score_contract_passed") is not True
            or quantized.get("error") != ROW_SUM_ERROR):
        raise ValueError("The only permitted quantized validation failure is the original FP32 row-sum guard")
    checks = quantized.get("checks", [])
    if len(checks) != 6 or [row["sample"] for row in checks] != list(range(6)):
        raise ValueError("All six simulation checks are required")
    errors = []
    for index, row in enumerate(checks):
        stats = row["actual_probabilities"]
        sums = np.asarray(stats["row_sums"], dtype=np.float64)
        reference = decode_ctc(expected[index:index+1], characters)
        actual = row["actual"]
        if (sums.shape != (40,) or not np.isfinite(sums).all() or np.max(np.abs(sums-1)) > atol
                or stats.get("probability_range_valid") is not True
                or not -1e-6 <= stats["min"] <= stats["max"] <= 1.000001
                or row.get("text_equal") is not True or actual["text"] != reference["text"]
                or row["reference"]["text"] != reference["text"]
                or not math.isfinite(actual["confidence"])
                or abs(actual["confidence"] - reference["confidence"]) > .1):
            raise ValueError(f"Simulation sample {index} fails the explicit probability/CTC/confidence contract")
        errors.append(float(np.max(np.abs(sums-1))))
    if not any(error > 1e-3 for error in errors):
        raise ValueError("No original row-sum failure was demonstrated")
    return errors


def check_physical_simulation(quantized, expected, characters, model_dir):
    """Recheck raw simulator values without accepting its failed probability sum.

    Direct greedy CTC reads unchanged finite values for diagnostic comparison;
    no larger decoder tolerance, probability normalization, or softmax is used.
    """
    from probe_ocr_usb import diagnostic_greedy
    if (quantized.get("status") != "FAILED" or quantized.get("strict_probability_contract_passed") is not False
            or quantized.get("ctc_and_score_contract_passed") is not True
            or quantized.get("error") != ROW_SUM_ERROR):
        raise ValueError("Only a simulator row-sum failure is eligible for a physical-device contract")
    filename = quantized.get("raw_outputs_npy", "")
    if (not isinstance(filename, str) or not filename or Path(filename).name != filename
            or PureWindowsPath(filename).name != filename or PureWindowsPath(filename).is_absolute()):
        raise ValueError("Saved simulator raw output must be a filename inside the selected model directory")
    raw_path = (model_dir / filename).resolve()
    if not raw_path.is_relative_to(model_dir) or sha(raw_path) != quantized.get("raw_outputs_npy_sha256"):
        raise ValueError("Saved simulator raw output hash/path changed")
    raw = np.load(raw_path, allow_pickle=False)
    if (raw.shape != expected.shape or raw.dtype != np.float32 or not np.isfinite(raw).all()
            or raw.min() < -1e-6 or raw.max() > 1.000001):
        raise ValueError("Raw simulator shape/dtype/finite/probability range contract failed")
    checks = quantized.get("checks", [])
    if len(checks) != 6 or [row.get("sample") for row in checks] != list(range(6)):
        raise ValueError("All six simulation checks are required")
    samples = []
    for index, row in enumerate(checks):
        values = raw[index:index+1]
        reference, actual = decode_ctc(expected[index:index+1], characters), diagnostic_greedy(values, characters)
        difference = actual["confidence"] - reference["confidence"]
        if actual["text"] != reference["text"] or abs(difference) > .1:
            raise ValueError(f"Raw simulator sample {index} fails CTC text/confidence <= 0.1")
        sums = values.sum(-1).ravel()
        stats, reported = row["actual_probabilities"], row["actual"]
        recorded_sums = np.asarray(stats["row_sums"], dtype=np.float64)
        if (row.get("text_equal") is not True or reported["text"] != actual["text"]
                or row["reference"]["text"] != reference["text"]
                or not math.isfinite(reported["confidence"])
                or abs(reported["confidence"]-actual["confidence"]) > 1e-6
                or not math.isfinite(row["reference"]["confidence"])
                or abs(row["reference"]["confidence"]-reference["confidence"]) > 1e-6
                or not math.isfinite(row["confidence_difference"])
                or abs(row["confidence_difference"]-difference) > 1e-6
                or stats.get("probability_range_valid") is not True
                or not math.isfinite(stats["min"]) or not math.isfinite(stats["max"])
                or abs(stats["min"]-float(values.min())) > 1e-6
                or abs(stats["max"]-float(values.max())) > 1e-6
                or recorded_sums.shape != (40,) or not np.isfinite(recorded_sums).all()
                or not np.allclose(recorded_sums, sums, rtol=0, atol=1e-6)):
            raise ValueError(f"Simulator sample {index} summary differs from saved raw output")
        error = float(np.abs(sums-1).max())
        samples.append({"sample": index, "text": actual["text"], "confidence": actual["confidence"],
                        "confidence_difference": difference, "non_row_contract_passed": True,
                        "strict_probability_sum_passed": error <= 1e-3,
                        "row_sums": sums.tolist(), "maximum_row_sum_error": error})
    if not any(row["maximum_row_sum_error"] > 1e-3 for row in samples):
        raise ValueError("No original simulator row-sum failure was demonstrated")
    return samples, raw_path


def evaluate(probe_report, model_dir, row_sum_atol, *, physical_device_only=False):
    # Lazy import avoids importing a hardware wrapper in lightweight runtime checks.
    from probe_ocr_usb import load_probe_spec
    atol = validate_tolerance(row_sum_atol, physical_device_only=physical_device_only)
    model_dir, probe_report = Path(model_dir).resolve(), Path(probe_report).resolve()
    spec, prepared = load_probe_spec(model_dir)
    if sha(spec.hef_path) != spec.hef_sha256:
        raise ValueError("Selected HEF changed after compilation")
    compilation = read_json(model_dir / "compile_report.json")
    if (compilation.get("status") != "DIAGNOSTIC_HEF_NOT_HARDWARE_VALIDATED"
            or compilation.get("diagnostic_probability_only") is not True
            or compilation.get("quantized_fp32_row_sum_guard_passed") is not False):
        raise ValueError("This exception requires a diagnostic compilation preserving the failed FP32 guard")
    probe = read_json(probe_report)
    raw_path = probe_report.parent / "raw_outputs.npz"
    if (probe.get("status") != "DIAGNOSTIC_COMPLETED" or probe.get("diagnostic_only") is not True
            or probe.get("deployment_approved") is not False or probe.get("hardware_used") is not True
            or probe.get("cpu_neural_fallback") is not False or probe.get("execution_providers") != {}
            or probe.get("models_closed_usb_released") is not True
            or probe.get("all_six_ctc_and_score_passed") is not True
            or probe.get("hef_sha256") != spec.hef_sha256
            or Path(probe["model_dir"]).resolve() != model_dir
            or not str(probe.get("device", "")).startswith("usb/") or not probe.get("shared_engine_id")):
        raise ValueError("A completed, released, genuine USB OCR probe is required")
    bindings = {"compile_report_sha256": "compile_report.json",
                "quantized_validation_report_sha256": "validate-quantized_report.json",
                "validation_input_sha256": "validation_bgr_normalized.npy",
                "expected_output_sha256": "onnx_expected.npy"}
    if any(probe.get(field) != sha(model_dir / filename) for field, filename in bindings.items()):
        raise ValueError("Probe evidence no longer matches the selected model/data")
    expected = np.load(model_dir / "onnx_expected.npy")
    inputs = np.load(model_dir / "validation_bgr_normalized.npy", mmap_mode="r")
    vocabulary = load_ocr_vocabulary("plate41")
    if (probe.get("ocr_vocabulary") != vocabulary.name
            or probe.get("ocr_vocabulary_metadata") != vocabulary.metadata):
        raise ValueError("Probe OCR vocabulary differs from the selected classifier")
    if expected.shape != (6,40,vocabulary.classes) or inputs.shape != (6,48,320,3) or prepared.get("validation_count") != 6:
        raise ValueError("Exactly six original holdouts are required")
    characters = vocabulary.characters
    quantized = read_json(model_dir / "validate-quantized_report.json")
    if physical_device_only:
        simulation_samples, simulation_raw_path = check_physical_simulation(quantized, expected, characters, model_dir)
        simulation_errors = [row["maximum_row_sum_error"] for row in simulation_samples]
    else:
        simulation_errors = check_simulation(quantized, expected, characters, atol)
    samples = probe.get("samples", [])
    if len(samples) != 6 or [row["index"] for row in samples] != list(range(6)):
        raise ValueError("Probe must contain exactly six complete samples")
    repeat_counts = {len(row.get("events", [])) for row in samples}
    minimum_repeats = 3 if physical_device_only else 2
    if len(repeat_counts) != 1 or min(repeat_counts) < minimum_repeats:
        raise ValueError(f"Every sample needs the same {minimum_repeats} or more completed USB repeats")
    repeats = repeat_counts.pop()
    events = [event for row in samples for event in row["events"]]
    if events != probe.get("neural_stage_runs") or [e.get("run_seq") for e in events] != list(range(1, 6*repeats+1)):
        raise ValueError("USB runs must be complete, consecutive, and independently audited")
    for event in events:
        if (event.get("stage") != "ocr" or event.get("status") != "COMPLETED"
                or event.get("backend") != "HAILO_USB" or event.get("accelerator_used") is not True
                or event.get("cpu_neural_fallback") is not False or event.get("hef_sha256") != spec.hef_sha256
                or event.get("device") != probe["device"] or event.get("shared_engine_id") != probe["shared_engine_id"]):
            raise ValueError("USB execution evidence has the wrong model/device/engine/backend")
    results = []
    with np.load(raw_path) as raw:
        if set(raw.files) != {f"sample{i}_repeat{r}" for i in range(6) for r in range(repeats)}:
            raise ValueError("Missing or unexpected raw USB repeats")
        for index, sample in enumerate(samples):
            if not all(sample.get(key) is True for key in ("repeatability_bit_exact", "text_and_score_passed", "probability_range_passed")):
                raise ValueError("Probe did not pass all non-row-sum gates")
            reference, first, row_errors = decode_ctc(expected[index:index+1], characters), None, []
            for repeat in range(repeats):
                values = raw[f"sample{index}_repeat{repeat}"]
                if (values.shape != (1,40,vocabulary.classes) or values.dtype != np.float32 or not np.isfinite(values).all()
                        or values.min() < -1e-6 or values.max() > 1.000001):
                    raise ValueError("Raw USB output shape/dtype/probability range is invalid")
                if first is None:
                    first = values
                elif not np.array_equal(first, values):
                    raise ValueError("Raw USB repeats are not bit-exact")
                decoded = decode_ctc(values, characters, probability_sum_atol=atol)
                if decoded["text"] != reference["text"] or abs(decoded["confidence"]-reference["confidence"]) > .1:
                    raise ValueError("Raw USB CTC text/confidence differs from the original model")
                row_errors.append(float(np.abs(values.sum(-1)-1).max()))
            results.append({"sample": index, "text": reference["text"], "repeats": repeats,
                            "maximum_row_sum_error": max(row_errors), "bit_exact": True})
    files = {"hef": spec.hef_path, "quantized_har": model_dir / "quantized.har",
             "probe_report": probe_report, "probe_raw_outputs": raw_path,
             **{name: model_dir / name for name in ("prepare_report.json", "optimize_report.json",
                 "validate-quantized_report.json", "compile_report.json", "output_manifest.json",
                 "validation_bgr_normalized.npy", "onnx_expected.npy")}}
    result = {"schema_version": 1, "status": STATUS, "contract_kind": "EXPLICIT_QUANTIZED_DEVICE_SOFTMAX",
            "ocr_vocabulary": vocabulary.name, "ocr_vocabulary_metadata": vocabulary.metadata,
            "row_sum_atol": atol, "row_sum_rtol": 0, "output_values_modified": False,
            "model_dir": str(model_dir), "hef_sha256": spec.hef_sha256,
            "source_model_sha256": prepared["source_files_sha256"]["inference.pdiparams"],
            "original_quantized_validation_status": "FAILED", "original_compile_status": compilation["status"],
            "original_fp32_row_sum_atol": 1e-3, "simulation_six_checks_passed": True,
            "simulation_maximum_row_sum_error": max(simulation_errors), "all_six_samples_passed": True,
            "all_repeats_bit_exact": True, "sample_count": 6, "repeats_per_sample": repeats,
            "device": probe["device"], "shared_engine_id": probe["shared_engine_id"],
            "samples": results, "full_pipeline_deployment_approved": False,
            "evidence": {key: {"path": str(Path(path).resolve()), "sha256": sha(path)} for key, path in files.items()}}
    if physical_device_only:
        result.update(schema_version=2, contract_kind=PHYSICAL_KIND, scope=PHYSICAL_SCOPE,
                      simulation_six_checks_passed=False, simulation_non_row_contract_passed=True,
                      simulation_probability_contract_passed=False,
                      simulation_samples=simulation_samples,
                      physical_device_maximum_row_sum_error=max(row["maximum_row_sum_error"] for row in results),
                      device_versions=probe.get("versions", {}), policy=PHYSICAL_POLICY)
        result["evidence"]["simulation_raw_outputs"] = {"path": str(simulation_raw_path), "sha256": sha(simulation_raw_path)}
    return result


def probability_contract_metadata(approved):
    """Small manifest/runtime metadata; full simulator files remain offline."""
    result = {"schema_version": approved["schema_version"], "row_sum_atol": approved["row_sum_atol"]}
    if approved["schema_version"] == 2:
        result.update({name: approved[name] for name in (
            "contract_kind", "scope", "original_quantized_validation_status",
            "simulation_probability_contract_passed", "simulation_maximum_row_sum_error",
            "physical_device_maximum_row_sum_error")})
    return result


def verify_probability_contract(report_path, expected_hef_sha256, *, full_evidence=True):
    """Runtime checks hash-bound manifest evidence; builder rechecks raw files."""
    report_path = Path(report_path).resolve()
    value = read_json(report_path)
    physical = value.get("schema_version") == 2
    validate_tolerance(value.get("row_sum_atol"), physical_device_only=physical)
    if physical:
        if (value.get("contract_kind") != PHYSICAL_KIND or value.get("scope") != PHYSICAL_SCOPE
                or value.get("simulation_six_checks_passed") is not False
                or value.get("simulation_non_row_contract_passed") is not True
                or value.get("simulation_probability_contract_passed") is not False
                or value.get("original_quantized_validation_status") != "FAILED"
                or not isinstance(value.get("simulation_maximum_row_sum_error"), (int, float))
                or not math.isfinite(value["simulation_maximum_row_sum_error"])
                or value["simulation_maximum_row_sum_error"] <= 1e-3
                or not isinstance(value.get("physical_device_maximum_row_sum_error"), (int, float))
                or not math.isfinite(value["physical_device_maximum_row_sum_error"])
                or not 0 <= value["physical_device_maximum_row_sum_error"] <= value["row_sum_atol"]):
            raise ValueError("Invalid physical-device-only probability approval")
    elif (value.get("schema_version") != 1 or value.get("contract_kind") != "EXPLICIT_QUANTIZED_DEVICE_SOFTMAX"
          or value.get("simulation_six_checks_passed") is not True):
        raise ValueError("Invalid original quantized-device probability approval")
    if (value.get("status") != STATUS
            or value.get("hef_sha256") != expected_hef_sha256 or value.get("row_sum_rtol") != 0
            or value.get("output_values_modified") is not False or value.get("sample_count") != 6
            or value.get("repeats_per_sample", 0) < (3 if physical else 2)
            or not all(value.get(key) is True for key in ("all_six_samples_passed", "all_repeats_bit_exact"))):
        raise ValueError("Invalid explicit device probability approval")
    if full_evidence:
        model_dir = Path(value["model_dir"]).resolve()
        if report_path != model_dir / REPORT_NAME:
            raise ValueError("Probability approval must remain beside its selected model")
        for item in value["evidence"].values():
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Probability approval evidence hash changed")
        fresh = evaluate(value["evidence"]["probe_report"]["path"], model_dir, value["row_sum_atol"], physical_device_only=physical)
        if any(value.get(key) != content for key, content in fresh.items()):
            raise ValueError("Probability approval does not match reevaluated source/USB evidence")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-report", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--row-sum-atol", type=float, required=True)
    parser.add_argument("--physical-device-only", action="store_true",
                        help="Separate schema2 device contract (atol <= 0.01, >=3 USB repeats); simulator row-sum failure is retained")
    args = parser.parse_args()
    report = evaluate(args.probe_report, args.model_dir, args.row_sum_atol, physical_device_only=args.physical_device_only)
    report["created_at"] = dt.datetime.now().astimezone().isoformat()
    output = args.model_dir / REPORT_NAME
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"], "output": str(output), "row_sum_atol": report["row_sum_atol"]}))


if __name__ == "__main__":
    main()
