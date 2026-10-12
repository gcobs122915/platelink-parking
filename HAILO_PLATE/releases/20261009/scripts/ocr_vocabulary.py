"""Hash-bound OCR vocabulary/model selection without creating inference sessions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = RELEASE.parents[2] / "PlateRecognition_Models_20261009"
PLATE41 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ軍使外試臨"
SOURCE_HASHES = {
    "inference.json": "cc01748691bf4118c27d8c0cf79507f19adc0b8824be2d9f63eba87713e3c750",
    "inference.pdiparams": "24aff12ad420fb488bd7cdff550ffcb016127d36182fabfb0496b8ec00212e28",
    "inference.yml": "991b700facf5b50a7de193468207d5f4255b538dde0d312ae3b7c7a9b6873129",
    "pretrained_character_dict.txt": "b5f2bfe2bdd9448429e3e82b51c789775d9b42f2403d082b00662eb77e401c5d",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_path(release, report, field, exact_relative):
    if report.get(field) != exact_relative:
        raise ValueError(f"Unexpected OCR artifact path for {field}")
    path = release / exact_relative
    if sha(path) != report.get(field + "_sha256"):
        raise ValueError(f"OCR artifact SHA-256 mismatch: {path}")
    return path


@dataclass(frozen=True)
class OcrVocabulary:
    name: str
    characters: tuple[str, ...]
    classes: int
    onnx_path: Path
    onnx_sha256: str
    metadata: dict


def load_ocr_vocabulary(name="original", *, release=RELEASE, bundle=DEFAULT_BUNDLE):
    """Validate the selected export, alphabet and provenance, without ORT/USB.

    Metadata paths are relative and hashes are platform-independent. Deployment
    manifests can bind this complete metadata dictionary for either vocabulary.
    """
    if name not in {"original", "plate41"}:
        raise ValueError("ocr_vocabulary must be original or plate41")
    release, bundle = Path(release).resolve(), Path(bundle).resolve()
    source = bundle / "models/ocr"
    for filename, expected in SOURCE_HASHES.items():
        if sha(source / filename) != expected:
            raise ValueError(f"Original OCR source changed: {filename}")
    full_chars = [""] + (source / "pretrained_character_dict.txt").read_text(encoding="utf-8").splitlines() + [" "]
    if len(full_chars) != 18710 or full_chars[0] != "":
        raise ValueError("Original OCR dictionary must have 18710 classes including blank and space")
    original_report = read(release / "validation/ocr_onnx_export.json")
    if (original_report.get("status") != "PASS_FLOAT_EXPORT"
            or original_report.get("sources_sha256") != SOURCE_HASHES):
        raise ValueError("Original OCR export/source validation did not pass")
    original_onnx = verified_path(release, original_report, "onnx", "models/onnx/ppocrv6_medium_v3_48x320.onnx")
    report_relative = "validation/ocr_onnx_export.json"
    report, onnx, characters = original_report, original_onnx, full_chars
    dictionary, dictionary_root = "models/ocr/pretrained_character_dict.txt", "source_bundle"
    dictionary_sha256 = SOURCE_HASHES["pretrained_character_dict.txt"]
    mapping_relative = mapping_sha256 = characters_json_sha256 = None
    if name == "plate41":
        report_relative = "validation/ocr_plate41/export_report.json"
        report = read(release / report_relative)
        if (report.get("status") != "PASS_SUBSET_FLOAT_EXPORT" or report.get("sources_sha256") != SOURCE_HASHES
                or report.get("source_onnx_sha256") != sha(original_onnx)
                or report.get("source_export_report_sha256") != sha(release / "validation/ocr_onnx_export.json")
                or report.get("classes") != 42 or report.get("blank_index") != 0
                or report.get("space_included") is not False
                or report.get("input_shape") != [1, 3, 48, 320] or report.get("output_shape") != [1, 40, 42]
                or report.get("all_nodes_byte_identical") is not True
                or report.get("all_non_head_initializers_byte_identical") is not True
                or report.get("source_bundle_modified") is not False or report.get("retrained") is not False):
            raise ValueError("Subset export shape/source/slicing contract did not pass")
        checks = report.get("checks", [])
        if (report.get("validation_count") != 6 or len(checks) != 6
                or [row.get("sample") for row in checks] != list(range(6))
                or any(any(row.get(key) is not True for key in ("passed", "subset_vs_selected_logits_softmax_allclose",
                                                               "ctc_matches_selected_logits_gold", "all_frame_ids_match_gold")) for row in checks)):
            raise ValueError("Subset requires six complete selected-logit Softmax validation checks")
        onnx = verified_path(release, report, "onnx", "models/onnx/ppocrv6_medium_v3_plate41_48x320.onnx")
        dictionary = "models/ocr_plate41/characters.txt"
        dictionary_path = verified_path(release, report, "dictionary", dictionary)
        chars_json = verified_path(release, report, "characters_json", "models/ocr_plate41/characters.json")
        mapping_relative = "models/ocr_plate41/mapping.json"
        mapping_path = verified_path(release, report, "mapping", mapping_relative)
        characters = [""] + list(PLATE41)
        characters_record = read(chars_json)
        if (dictionary_path.read_text(encoding="utf-8").splitlines() != list(PLATE41)
                or characters_record.get("characters") != characters or characters_record.get("classes") != 42
                or characters_record.get("blank_index") != 0 or characters_record.get("space_included") is not False):
            raise ValueError("Subset dictionary must exactly match blank plus the 41 requested characters")
        original_indices = []
        for character in characters:
            positions = [i for i, value in enumerate(full_chars) if value == character]
            if len(positions) != 1:
                raise ValueError("Each requested OCR class must identify one unique original class")
            original_indices.append(positions[0])
        mapping = read(mapping_path)
        expected_rows = [{"new_index": i, "original_index": old, "character": characters[i]} for i, old in enumerate(original_indices)]
        head = mapping.get("head_slicing", {})
        if (mapping.get("classes") != expected_rows or mapping.get("class_count") != 42
                or mapping.get("blank_index") != 0 or mapping.get("space_included") is not False
                or mapping.get("visible_characters") != PLATE41
                or mapping.get("sources_sha256") != SOURCE_HASHES
                or mapping.get("source_onnx_sha256") != report["source_onnx_sha256"]
                or mapping.get("source_dictionary_sha256") != SOURCE_HASHES["pretrained_character_dict.txt"]
                or head != report.get("head_slicing") or head.get("original_indices") != original_indices
                or head.get("weight_name") != "linear_8.w_0_0" or head.get("bias_name") != "linear_8.b_0_0"
                or head.get("source_weight_shape") != [192, 18710] or head.get("selected_weight_shape") != [192, 42]
                or head.get("source_bias_shape") != [18710] or head.get("selected_bias_shape") != [42]
                or head.get("weight_axis") != 1 or head.get("bias_axis") != 0
                or head.get("selection_byte_exact") is not True):
            raise ValueError("Subset original-class mapping or exact head-slicing evidence changed")
        dictionary_root, dictionary_sha256 = "release", report["dictionary_sha256"]
        mapping_sha256, characters_json_sha256 = report["mapping_sha256"], report["characters_json_sha256"]
    metadata = {
        "name": name, "classes": len(characters), "blank_index": 0, "space_included": name == "original",
        "visible_characters": PLATE41 if name == "plate41" else None,
        "characters_sha256": hashlib.sha256(json.dumps(characters, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest(),
        "dictionary": dictionary, "dictionary_root": dictionary_root, "dictionary_sha256": dictionary_sha256,
        "characters_json_sha256": characters_json_sha256,
        "mapping": mapping_relative, "mapping_sha256": mapping_sha256,
        "onnx": report["onnx"], "onnx_sha256": report["onnx_sha256"],
        "export_report": report_relative, "export_report_sha256": sha(release / report_relative),
        "source_dictionary_sha256": SOURCE_HASHES["pretrained_character_dict.txt"],
        "output_semantics": report["output_semantics"],
    }
    return OcrVocabulary(name, tuple(characters), len(characters), onnx, report["onnx_sha256"], metadata)
