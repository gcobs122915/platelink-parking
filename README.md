# PlateLink Parking

Team: 全球雲通智慧整合

Private project repository for the ASUS UGen AI League.

## Current status

- Implemented locally: license-plate detection, four-corner POSE estimation, and OCR inference on ASUS UGen300 / Hailo-10H.
- The host handles image processing, geometric correction, decoding, application logic, and file I/O.
- Planned: migrate the existing GPU-based vehicle-detection YOLO to UGen and integrate parking-space occupancy management with plate recognition.
- Source code upload is pending; a curated source-review package has been prepared locally.
- Model weights and hardware runtime dependencies will be documented separately. This repository is not yet a runnable deployment.
