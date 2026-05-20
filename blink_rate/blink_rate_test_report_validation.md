# Blink Rate Test Report Validation

## Reuse Decision

Decision: AIREST should adapt the OpenWillis blink logic rather than use OpenWillis directly or implement a new lightweight blink detector.

Why this is the best direction:
- The current OpenWillis path already implements the right core detector for AIREST: MediaPipe FaceMesh per frame, bilateral EAR averaging, z-score normalization, and trough-based blink detection with `scipy.signal.find_peaks` in [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:131), [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:203), [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:300), and [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:368).
- A small validation run on internal clips produced complete blink outputs with stable landmark coverage, so there is no evidence that AIREST needs a ground-up replacement detector.
- OpenWillis should not be used as-is because it does not apply QC gates itself, uses clip-level FPS for timing, and drops no-face frames before building the returned EAR table. That means direct use would hide face-loss windows and can make downstream blink-rate interpretation look cleaner than it really is.
- A lightweight custom detector would mostly re-implement the same EAR pipeline while adding new tuning and maintenance burden. The current clip set does not show a clear empirical advantage that would justify that extra work.

Recommended AIREST adaptation layer:
- Reuse the OpenWillis core detector.
- Add explicit QC checks for FPS, `face_present_ratio`, `eye_landmark_ratio`, and `missing_ear_ratio`.
- Preserve total frame count and no-face windows for auditability.
- Emit `blink_qc` and set blink outputs to `NaN` when QC fails.
- Derive `blink duration` and `IBI` downstream from the event table, but do not treat `25-30 fps` clips as suitable for fine blink kinematics.

## Small Validation Note

Validation set:
- [sample_data/baseline.mp4](/Users/pelmeshek1706/Desktop/projects/airest-face/sample_data/baseline.mp4)
- [sample_data/not_expressive.mp4](/Users/pelmeshek1706/Desktop/projects/airest-face/sample_data/not_expressive.mp4)
- [sample_data/expressive.mp4](/Users/pelmeshek1706/Desktop/projects/airest-face/sample_data/expressive.mp4)

Method:
- Ran `eye_blink_rate()` from the local OpenWillis checkout.
- Ran a second frame-by-frame MediaPipe pass to estimate `face_present_ratio` and `eye_landmark_ratio` on the same clips.

| Clip | Duration (s) | FPS | Face-present ratio | Eye-landmark ratio | Missing EAR ratio | Blinks | Blink rate (per min) | QC outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline.mp4` | 3.47 | 30.0 | 1.00 | 1.00 | 0.00 | 5 | 86.54 | passed |
| `not_expressive.mp4` | 8.30 | 30.0 | 1.00 | 1.00 | 0.00 | 8 | 57.83 | passed |
| `expressive.mp4` | 21.47 | 30.0 | 1.00 | 1.00 | 0.00 | 18 | 50.31 | passed |

Main observations:
- All three internal clips produced non-empty `ear`, `blinks`, and `summary` outputs.
- On these clean frontal clips, the OpenWillis core detector had complete face coverage and no missing EAR values, which supports reusing the existing detector logic.
- The very short `baseline.mp4` clip produced a high blink-rate estimate (`86.54/min`). That does not indicate a detector failure by itself, but it does show that blink-rate interpretation is sensitive to short clip duration and should be treated as coarse on brief segments.
- This validation set is clean and does not stress occlusion, head turns, glasses glare, face-loss windows, or variable FPS. That is the main reason the recommendation is to adapt OpenWillis with a QC wrapper rather than call it directly without review gates.

Bottom line:
- The validation supports reuse of the OpenWillis blink core.
- The validation does not support using OpenWillis raw outputs without an AIREST QC/adaptation layer.
