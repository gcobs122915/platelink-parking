"""Bind a compiled OCR graph to the exact user-authorized classifier subset."""
import hashlib
import json
import math
from pathlib import Path

from ocr_vocabulary import load_ocr_vocabulary
from reference_pipeline import RELEASE


POOL7_DEPTHWISE_SHA256 = "4c49cc16ac368f0da9c61d3f914b4aebb976d0f909e3e60333817282a98585ad"


def _bound_artifact(relative, digest):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Graph derivation requires a release-relative artifact")
    path = (RELEASE / relative).resolve()
    if (not path.is_relative_to(RELEASE.resolve()) or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
        raise ValueError("Graph derivation artifact is missing, changed, or outside the release")
    return path


def verified_prepared_graph(prepared, stream, vocabulary):
    """Allow the original graph or this exact audited fixed-operator rewrite.

    This proves graph/vocabulary provenance only. It never approves numerical
    quantization, compiled kernels, physical inference, or deployment.
    """
    original = vocabulary.onnx_sha256
    if prepared.get("source_onnx_sha256") != original:
        raise ValueError("Prepared OCR has a different original classifier source")
    derivation = prepared.get("graph_derivation")
    if prepared.get("prepared_onnx_sha256") == original:
        if derivation is not None or stream.get("graph_derivation") is not None:
            raise ValueError("Unmodified OCR must not declare a different graph derivation")
        return
    if (not isinstance(derivation, dict) or derivation != stream.get("graph_derivation")
            or derivation.get("kind") != "AVGPOOL7_FIXED_DEPTHWISE_3X2"
            or derivation.get("source_onnx_sha256") != original
            or derivation.get("derived_onnx_sha256") != POOL7_DEPTHWISE_SHA256
            or prepared.get("prepared_onnx_sha256") != POOL7_DEPTHWISE_SHA256):
        raise ValueError("Unsupported or inconsistent OCR graph derivation")
    _bound_artifact(derivation.get("derived_onnx"), POOL7_DEPTHWISE_SHA256)
    path = _bound_artifact(derivation.get("equivalence_report"), derivation.get("equivalence_report_sha256"))
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (evidence.get("status") != "PASS_FLOAT_EQUIVALENCE_DIAGNOSTIC_ONLY"
            or evidence.get("source_model_sha256") != original
            or evidence.get("candidate_model_sha256") != POOL7_DEPTHWISE_SHA256
            or evidence.get("validation_array_sha256") != prepared.get("validation_npy_sha256")
            or evidence.get("existing_expected_sha256") != prepared.get("expected_npy_sha256")
            or any(evidence.get(key) is not True for key in (
                "source_model_unchanged", "unchanged_original_nodes_except_one",
                "fixed_operator_rewrite_only", "graph_interface_unchanged"))
            or evidence.get("learned_weights_changed") is not False
            or evidence.get("input_resolution_changed") is not False):
        raise ValueError("Fixed-operator rewrite does not retain original graph/weights and holdouts")
    initializers = evidence.get("unchanged_original_initializers", [])
    if (evidence.get("original_initializer_count") != 257 or len(initializers) != 257
            or len({row.get("name") for row in initializers}) != 257
            or not all(row.get("byte_identical") is True for row in initializers)):
        raise ValueError("All original OCR initializers must remain byte-identical")
    pool, replacement = evidence.get("original_pool", {}), evidence.get("replacement", {})
    if (pool.get("name") != "AveragePool.0"
            or pool.get("attributes") != {"kernel_shape": [3, 2], "strides": [3, 2],
                "pads": [0, 0, 0, 0], "ceil_mode": 0, "count_include_pad": 0}
            or replacement.get("op_type") != "Conv" or replacement.get("group") != 768
            or replacement.get("weight_shape") != [768, 1, 3, 2]
            or replacement.get("coefficient_float32") != 0.1666666716337204
            or replacement.get("bias_present") is not False
            or replacement.get("attributes") != {"dilations": [1, 1], "group": 768,
                "kernel_shape": [3, 2], "pads": [0, 0, 0, 0], "strides": [3, 2]}):
        raise ValueError("Only the validated fixed one-sixth depthwise operator is allowed")
    checks = evidence.get("samples", [])
    def close_stat(value, shape, limit):
        maximum = value.get("maxabs")
        return (value.get("shape") == shape and value.get("allclose") is True
                and type(maximum) in (int, float) and math.isfinite(maximum) and 0 <= maximum <= limit)
    if (evidence.get("validation_count") != 6 or len(checks) != 6
            or [row.get("sample") for row in checks] != list(range(6))
            or not all(row.get("ctc_equal") is True and row.get("all_frame_ids_equal") is True
                and row.get("original", {}).get("text") == row.get("candidate", {}).get("text")
                and close_stat(row.get("final_probability", {}), [40, 42], 3e-6)
                and close_stat(row.get("pool_outputs", {}), [768, 1, 40], 3e-6) for row in checks)
            or any(evidence.get("gates", {}).get(key, {}).get("array_equal") is not True for key in (
                "original_capture_final", "original_vs_existing_expected"))):
        raise ValueError("Six fixed-operator float equivalence holdouts must pass")


def verified_compiled_vocabulary(prepared, stream):
    vocabulary = load_ocr_vocabulary("plate41")
    report_path = RELEASE / "validation/ocr_plate41/export_report.json"
    exported = json.loads(report_path.read_text(encoding="utf-8"))
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    if (prepared.get("classes") != vocabulary.classes
            or prepared.get("export_report_sha256") != report_sha
            or stream.get("classes") != vocabulary.classes
            or stream.get("blank_index") != 0
            or stream.get("classifier_subset_authorized_by_user") is not True):
        raise ValueError("Compiled OCR does not match the verified user-authorized 42-class model")
    verified_prepared_graph(prepared, stream, vocabulary)
    for field in ("dictionary", "dictionary_sha256", "mapping", "mapping_sha256"):
        if prepared.get(field) != exported[field] or stream.get(field) != exported[field]:
            raise ValueError(f"Compiled OCR vocabulary {field} changed")
    mapping = json.loads((RELEASE / exported["mapping"]).read_text(encoding="utf-8"))
    indices = [row["original_index"] for row in mapping["classes"]]
    alphabet = "".join(vocabulary.characters)
    if any(value.get("original_class_indices") != indices or value.get("deployment_alphabet") != alphabet
           for value in (prepared, stream)):
        raise ValueError("Compiled OCR class order differs from the selected original classifier columns")
    return vocabulary
