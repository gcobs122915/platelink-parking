# Local packaging verification — 2026-10-12

- 26 copied source, launcher and original requirements files matched their source SHA-256 hashes. The safe configuration was copied as `config.example.json`.
- All 22 Python files compiled in memory without generating bytecode.
- The three included fixture-based test modules passed: **23 tests**, using the existing Windows Python 3.13.5 environment and temporary test directories. No physical device or original application state was used.
- A whitelist inventory check found no image, model binary, raw report, database, log, runtime directory, Python cache, or Git metadata.
- Targeted GitHub-token, private-key and AWS-access-key patterns had no matches. Private endpoint and original workstation-path patterns had no matches. These bounded checks are not a comprehensive security audit.
- `SHA256SUMS.txt` covers every package file except itself, using paths relative to this directory.

During the local packaging verification, no USB/model inference, fresh dependency installation, hardware deployment, ParkingMapApp integration, remote repository creation, upload, or access invitation was performed. Repository creation and source publication are separate subsequent steps. The omitted model/provenance bundle remains necessary for runtime execution. All original files were left unchanged.

## Public source package and portable demo — 2026-10-12

The original source files remain unchanged. Public-facing documentation was updated and an optional `demo_web/` localhost interface was added.

- The portable demo passed **9 tests**, using temporary folders, mocked device results and real loopback HTTP requests. Its **5 Python files** passed compilation checks. See [demo_web/VERIFICATION.md](demo_web/VERIFICATION.md) for commands, covered behavior and limitations.
- A separate read-only review checked sample/result path bounds, Host/Origin restrictions, fresh job and source-hash matching, USB-stage evidence handling, runtime isolation and child-process ownership.
- The package contains source, configuration examples and documentation only. Bounded filename and text scans found no real photographs, recordings, model binaries, runtime state, embedded credentials or private workstation paths. These are bounded packaging checks, not a comprehensive security audit.
- Runtime configuration, sample images, outputs and recordings are excluded by `.gitignore`. The SHA-256 manifest was rebuilt for the final public package and covers every file except itself.

The portable version was **not rerun on USB after packaging**. The earlier real USB recording belongs to the earlier local demo instance; these packaging tests do not establish hardware performance, field accuracy, or completed vehicle/occupancy integration. Model/provenance assets and the matching vendor runtime are still supplied separately.
