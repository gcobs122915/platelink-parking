# PlateLink Parking — ASUS UGen AI League

Team: 全球雲通智慧整合

Private repository: https://github.com/gcobs122915/platelink-parking

**SOURCE REVIEW PACKAGE — not a clone-and-run hardware distribution.**

This snapshot contains the existing Windows file-exchange license-plate recognition core and selected shared Python dependencies. Its neural pipeline detects plates, estimates four corners, rectifies crops, and recognizes text using ASUS UGen300 / Hailo-10H. The host performs image processing, geometry, fixed decoding and file I/O. The implemented core has a resident shared device, eight model profiles, concurrent request buffers, a persistent queue, restart recovery, and POSE crop fitting.

No new ParkingMapApp integration, parking-space-to-plate association, vehicle-tracking migration, or migration of the occupancy detector to UGen is implemented in this package. It does not start an HTTP API server.

## Contents

The original sibling layout is retained because imports and launchers depend on it:

```text
車牌辨識核心/app/                    File protocol, queue and asynchronous adapter
車牌辨識核心/tests/                  Three fixture-based unit-test modules
車牌辨識核心/config.example.json     Safe relative-path configuration
車牌辨識核心/{start,send,stop}.cmd    Unmodified original launchers
HAILO_PLATE/releases/20261009/scripts/       Selected shared runtime source
HAILO_PLATE/releases/20261009/hailo_scripts/  Fixed YOLO/POSE decoders
```

Copied Python files and launchers are unchanged. Plate strings and fake HEF bytes in the unit tests are synthetic fixtures. No actual vehicle photos, raw recognition reports, model weights, HEFs, databases, vendor packages, or credentials are included. No software license has been added or inferred.

## Review setup

The inspected environment used Windows 11 and Python 3.13.5. Install `requirements-review.txt` in a separate review environment. The original `車牌辨識核心/requirements.txt` only supplements an existing environment with Pillow and is not a complete setup specification.

From the package root, the safe tests can be run with:

```powershell
python -B -m unittest discover -s '車牌辨識核心/tests' -p 'test_*.py' -v
```

These tests use temporary folders and explicit model/device doubles; they do not need USB, HailoRT, weights or original photos. See `VERIFICATION.md` for checks actually performed. A fresh installation from the pinned dependencies was not tested during packaging.

## Hardware prerequisites — separate bundle required

Do not run `start.cmd` before installing the separately authorized runtime assets. The unchanged launchers expect `HAILO_PLATE/.venv/Scripts/python.exe`. Hardware execution requires HailoRT Windows runtime/driver and matching Python wheel **5.3.2**, a UGen300 device, the original source-model bundle, and exact hash-bound deployment/provenance assets. Copy `config.example.json` to `config.json` only when preparing a runtime installation; its paths are relative to the core directory.

The runtime references these asset categories, none of which are included:

```text
PlateRecognition_Models_20261009/models/yolo/yolo11l.pt
PlateRecognition_Models_20261009/models/pose/yolo11l-pose.pt
PlateRecognition_Models_20261009/models/ocr/
HAILO_PLATE/releases/20261009/compiler/all_usb_manifest.json
HAILO_PLATE/releases/20261009/compiler/<eight HEF paths listed in the manifest>
HAILO_PLATE/releases/20261009/compiler/ocr_pooldw_softmax_analytic_probe_package/hardware_probability_contract_report.json
HAILO_PLATE/releases/20261009/compiler/ocr_41chars_48x320_pool7_depthwise_candidate/{complete_ocr_pool7_depthwise.onnx,report.json}
HAILO_PLATE/releases/20261009/models/onnx/{ppocrv6_medium_v3_48x320.onnx,ppocrv6_medium_v3_plate41_48x320.onnx}
HAILO_PLATE/releases/20261009/models/ocr_plate41/{characters.txt,characters.json,mapping.json}
HAILO_PLATE/releases/20261009/validation/ocr_onnx_export.json
HAILO_PLATE/releases/20261009/validation/ocr_plate41/export_report.json
```

The source OCR directory requires `inference.json`, `inference.pdiparams`, `inference.yml`, and `pretrained_character_dict.txt`. Source bundle placement must be alongside `HAILO_PLATE` and `車牌辨識核心`. Provenance reports may contain original sample data or local paths; editing their contents breaks existing hash bindings. Do not disable validation to make an incomplete package appear runnable.

`approve_ocr_probability_contract.py` is included for runtime validation with `full_evidence=False`. Its standalone diagnostic/re-approval mode and the standalone CPU reference CLI need additional tooling outside this package. ONNX Runtime is used only for Windows DLL loading order in the all-USB path, not CPU neural inference.

## Limits and submission status

The compiled YOLO input remains 736×1280 after preprocessing; unsupported whole-image proportions can fail. POSE fitting does not remove this limitation. This packaging check is not a hardware benchmark, independent labeled accuracy study, long-duration stress test, or integrated parking-management acceptance test.

The GitHub owner is `gcobs122915`. This source-review snapshot is prepared for the existing **Private** repository linked above. Reviewer access has not been granted; confirm the organizer's actual reviewer account before granting access. A private URL alone does not grant access. See `私人提交說明.txt` for GitHub access options.
