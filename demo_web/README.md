# PlateLink local UGen demonstration

This is the portable source version of the local test webpage used for the V3
demonstration recording. It submits new photographs to the existing USB plate
core and displays the resulting text, stage evidence, timing and annotated image.
It never starts inference automatically or substitutes prerecorded results.

**Source is public; runtime assets and real photographs are not included.**
The webpage is served on `127.0.0.1` only. GitHub's source page or GitHub Pages
cannot access the USB device through this Python service. Hosting this folder
as static files does not provide working recognition.

## What is implemented

- Existing USB plate detection, four-corner estimation and OCR, with host image
  preparation, perspective correction and text decoding.
- Fresh, unique file-protocol jobs, source-hash verification, real status polling,
  bounded queue, and explicit errors when hardware or evidence is unavailable.
- Optional browser screen recording saved under `runtime/recordings/` locally;
  the browser asks you to choose a tab/window. No audio is captured.

Vehicle/occupancy YOLO migration, space-to-plate association and ParkingMapApp
integration remain planned. The military/diplomatic/foreign buttons select
user-provided examples; they do not report a predicted plate category. Completed
history entries are labeled saved runs. Click **Run on UGen** for fresh inference.

## Prerequisites

The previously observed hardware environment was Windows 11, Python 3.13.5,
ASUS UGen300 / Hailo-10H and HailoRT Windows driver/runtime plus matching Python
wheel **5.3.2**. Use an environment supported by your vendor release. Python
dependencies are described in the repository's `requirements-review.txt`.

You must separately supply the authorized model/provenance bundle described in
the root README: eight HEFs, manifest, source-model hashes, OCR vocabulary,
ONNX and probability-contract validation assets. The existing core validates
these bindings. Do not bypass those checks or assume installing Python packages
alone makes this source package runnable.

Preserve the sibling layout:

```text
platelink-parking/
  demo_web/                        this folder
  車牌辨識核心/app/                existing core source
  HAILO_PLATE/releases/20261009/    shared source + separately supplied assets
  PlateRecognition_Models_20261009/ separately supplied source-model bundle
```

## Prepare and run (PowerShell, from repository root)

Use a dedicated Python environment. For example, when you already have the
required Python version installed:

```powershell
python -m venv .venv-demo
& .\.venv-demo\Scripts\python.exe -m pip install -r requirements-review.txt
# Install the matching vendor-provided HailoRT wheel into this same environment.
# Install its Windows runtime/driver and place the separately supplied model bundle.
Copy-Item demo_web/demo.config.example.json demo_web/demo.config.json
```

Place authorized JPEG photographs in ignored `demo_web/samples/`:
`military.jpg`, `diplomatic.jpg`, `foreign.jpg` (maximum 20 MiB each). No default
photos or known expected answers are distributed. To use different names, edit
the three `samples` entries in `demo.config.json`; they must stay in `samples/`.
`app_dir` is relative to `demo_web`, and may point to another matching source
checkout whose `HAILO_PLATE` remains beside `車牌辨識核心`. The port is configurable.

```powershell
& .\.venv-demo\Scripts\python.exe -B demo_web/launcher.py check
& .\.venv-demo\Scripts\python.exe -B demo_web/launcher.py start
```

Open **http://127.0.0.1:8766** in current Edge or Chrome. Keep the terminal open.
Wait for the core to load and the device to be ready, choose a photo, then click
**Run on UGen**. Startup does not send any photo. A missing model, stale heartbeat,
missing USB or failed stage is an error rather than successful recognition.

For UI inspection without loading any model/device, run the following instead:

```powershell
& .\.venv-demo\Scripts\python.exe -B demo_web/launcher.py serve
```

This serves the interface with Run disabled unless a separately running demo
core is genuinely ready. It does not emulate recognition. For source-only review,
`serve`, `check` and the smoke tests need only Python's standard library.

## Stop and local data

Press **Ctrl+C** in the launcher terminal, or open another terminal at the same
repository root and run:

```powershell
& .\.venv-demo\Scripts\python.exe -B demo_web/launcher.py stop
```

Only this demo's runtime stop flags are written. The supervisor waits up to 30
seconds for its children to exit and may terminate only child processes it
started. It never finds or kills an unrelated process by a remembered PID.
An exclusive launcher lock prevents two launchers sharing this runtime folder.
Avoid force-closing the launcher terminal; use the normal stop command first.

All new input copies, output images/text/JSON, logs, state and recordings remain
inside ignored `demo_web/runtime/`. Config and source photos are ignored too.
No production input/output directory is reused. Results can consume disk space;
after the demo is fully stopped, review and remove only your own `runtime/` data
when no longer needed. The wrapper retries transient Windows sharing failures
for up to two seconds; it does not alter recognition results or hide elapsed time.

## Scope of the evidence

The original local recording used actual USB execution. This portable packaging
was checked with mocked service tests, not another hardware run. See
[VERIFICATION.md](VERIFICATION.md). Job processing timing on this webpage saves
annotated images and is not directly comparable with the earlier benchmark that
excluded diagnostic output. A successful example is not a measured field accuracy.

The server rejects foreign Host/Origin requests, restricts photos to configured
files, limits recording uploads to 100 MiB and queues at most 12 pending jobs.
It is a local demonstration service, not a production network API. Do not bind it
to `0.0.0.0`, tunnel it, or publish local runtime files as website assets.
