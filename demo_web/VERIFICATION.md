# Portable demo packaging verification

Checked on 2026-10-12. This record applies to the portable `demo_web/` source,
not a new accuracy or speed benchmark.

## Checks performed

```powershell
python -B -m unittest discover -s demo_web -p test_demo.py -v
python -B demo_web/launcher.py check
```

**9 smoke tests passed.** They use temporary directories and explicitly mocked
device readiness / submission results. No HailoRT import, USB inference,
production service launch or original photograph is needed by these tests.

The tests cover:

1. Repeating the same sample creates two fresh jobs and two core submissions.
2. A mismatched source hash produces an error, never a recognized result.
3. Events reported by a non-USB backend do not count as USB success.
4. Missing samples or an unready device reject submission.
5. The pending queue is bounded to 12 jobs.
6. Transient Windows file-sharing errors are retried.
7. Real loopback HTTP responses enforce Host/Origin, reject cross-site requests,
   disallow arbitrary paths, handle missing photos, enforce JSON content type,
   and reject an oversized recording before writing it.
8. Configured sample and runtime paths cannot escape their allowed folders.
9. The launcher has an exclusive runtime lock; stop flags are demo-local and
   the lock can be reacquired after its owner releases it.

All **5 Python source files passed `py_compile`**, with temporary bytecode output
kept outside the package. `launcher.py check` found the sibling core source and
correctly reported all three sample photographs absent in the public package.

## Separate hardware evidence

The earlier local version used in the V3 recording executed three supplied
military/diplomatic/foreign examples on the connected UGen300, with recorded
USB stage evidence. That recording is evidence for that local instance.

This portable version changes paths, configuration, supervision, missing-file
handling and UI stage grouping. **It was not rerun on USB after packaging.**
The original recognition core and model primitives are not modified by this
folder. An authorized runtime installation is still required to validate the
portable version on a new machine.

Fresh dependency installation, vendor wheel/driver installation, complete model
bundle provisioning, browser screen-capture permission handling, long-duration
recording and shutdown during a real active inference were not retested here.
The tests are not evidence of field recognition accuracy or occupancy integration.

No models, source photos, runtime state, output files, recordings, credentials or
production configuration are included in this demo package.
