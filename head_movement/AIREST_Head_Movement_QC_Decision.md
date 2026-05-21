# AIREST Head Movement QC and Decision Section

## Scope

This section defines the Quality Checks (QC) required before AIREST uses head-movement features in clinical data capture or downstream modelling.

The QC applies to video-derived head movement features produced from the OpenWillis-style `head_movement()` pipeline: `pitch`, `yaw`, `roll`, `xy_disp`, `euclidean_angle`, `euclidean_angle_disp`, and summary means/standard deviations. The thresholds below are operational QC thresholds for clinical data quality; they are not diagnostic thresholds and must be recalibrated after AIREST dry-runs and the first locked pilot dataset.

## Feature and QC assumptions

1. Head movement is a secondary behavioural feature family, not a standalone clinical endpoint.
2. The preferred extraction mode for research comparability is offline post-processing at 15 or 30 fps, with `normalize_by_bb_size=True` for cross-video comparison.
3. Online capture should run lightweight QC only: face present, one face, approximate frontal pose, camera stability, and dropped-frame detection.
4. Offline OpenWillis/py-feat processing should generate the actual modelling features and final QC metrics.
5. Missingness must be computed over sampled frames, not all source frames, because unsampled frames may be intentionally filled with `NaN` when `frames_per_second` is below the source FPS.

## Required QC status labels

| QC status | Meaning | Dataset action |
|---|---|---|
| `PASS_PRIMARY` | Video is suitable for primary head-movement feature use. | Include in primary multimodal modelling. |
| `PASS_SECONDARY_WITH_FLAGS` | Usable but has moderate quality warnings. | Include only with QC covariates/sensitivity analysis. |
| `MANUAL_REVIEW` | Borderline, contradictory, or clinically important but technically noisy. | Review before dataset lock. |
| `FAIL_NO_USE_HEAD_MOVEMENT` | Head movement features are unreliable. | Exclude head-movement features; keep other modalities if valid. |
| `FAIL_RECAPTURE_IF_POSSIBLE` | Capture failed during live session and can be repeated before participant leaves. | Repeat only if allowed by protocol and participant condition. |

## QC checks

### QC-0. Video and metadata integrity

| Check | Pass | Warning | Fail |
|---|---:|---:|---:|
| Video readable | File opens and duration matches session metadata within ±1 s | Metadata missing but file readable | Cannot open video or duration mismatch >5 s |
| Source FPS | ≥25 fps preferred | 15–24 fps | <15 fps |
| Resolution | ≥720p | 480p–719p | <480p |
| Frame drop ratio | <5% | 5–10% | >10% |
| Session alignment | Video timestamps align with task timeline | Minor alignment gap ≤1 s | Missing or inconsistent task timestamps |

### QC-1. Face visibility and identity framing

| Check | Pass | Warning | Fail |
|---|---:|---:|---:|
| Exactly one participant face | ≥95% of sampled frames | 90–95% | <90% or repeated second face |
| Valid pose frame rate | ≥90% of sampled frames have valid face pose | 80–90% | <80% |
| Maximum continuous face-loss gap | ≤1.0 s | 1.0–2.0 s | >2.0 s |
| Repeated face-loss gaps | ≤2 gaps >1.0 s | 3 gaps >1.0 s | >3 gaps >1.0 s |
| Face not clipped by frame | ≥95% of valid frames | 90–95% | <90% |
| BBox size stability | Median bbox width 15–60% of frame width; no major clipping | Outside range but stable | Face too small/large, clipped, or unstable |
| Face centered enough for pose | BBox center x in 25–75% frame width and y in 20–80% frame height for ≥90% frames | 80–90% | <80% |

Implementation note: if `bbox_list` is supplied, do not trust `face_confidence` as a true model confidence score because the current implementation overwrites it with `1`. Use bbox coverage, bbox stability, pose availability, and missingness instead.

### QC-2. Allowable head pose range

These thresholds are intended to preserve reliable facial landmark/head-pose estimation in a webcam-based clinical setup. They use absolute angle values because sign conventions may differ between detector backends.

| Pose metric | Primary usable frame | Warning frame | Invalid frame |
|---|---:|---:|---:|
| `abs(yaw)` | ≤35° | 35–45° | >45° |
| `abs(pitch)` | ≤25° | 25–35° | >35° |
| `abs(roll)` | ≤20° | 20–30° | >30° |
| Extreme hard-fail pose | N/A | N/A | `abs(yaw)>60°` or `abs(pitch)>45°` or `abs(roll)>45°` |

Task-level rule:

| Condition | Status |
|---|---|
| ≥80% of valid sampled frames are inside primary usable pose range | `PASS_PRIMARY` for pose range |
| 60–80% are inside primary range, and <20% are invalid | `PASS_SECONDARY_WITH_FLAGS` |
| >20% invalid frames or >5% extreme hard-fail frames | `FAIL_NO_USE_HEAD_MOVEMENT` |

Calibration/start-of-task rule:

| Check | Threshold |
|---|---:|
| Frontal calibration window | First 5–10 s should have `abs(yaw)≤20°`, `abs(pitch)≤15°`, `abs(roll)≤15°` for ≥90% of frames |
| If failed | Ask participant to re-center before starting the clinical task; do not interpret this as behaviour |

### QC-3. Excessive movement, tracker jumps, and camera artifacts

Head movement can be a valid behavioural signal, so QC must separate real movement from technical artifact. The following checks should be computed only over valid sampled frames after removing frames with invalid pose range.

Preferred derived variables:

```text
xy_disp_norm = xy_disp / bbox_width
angle_disp_rate = euclidean_angle_disp / frame_delta_seconds
bbox_width_change = abs(width_t - width_t-1) / median_bbox_width
bbox_height_change = abs(height_t - height_t-1) / median_bbox_height
```

| Check | Pass | Warning | Fail / artifact likely |
|---|---:|---:|---:|
| `p95(xy_disp_norm)` | ≤0.12 face-width/frame interval | 0.12–0.18 | >0.18 |
| Extreme `xy_disp_norm` fraction | ≤2% frames >0.20 | 2–5% | >5% frames >0.25 or any repeated jumps >0.50 |
| `p95(angle_disp_rate)` | ≤150°/s | 150–240°/s | >240°/s |
| Extreme angular jumps | ≤2% frames >240°/s | 2–5% | >5% frames >360°/s |
| BBox width/height jump | ≤5% frames with >15% change | 5–10% | >10% or repeated >25% jumps |
| Camera/laptop movement | No global shake marker | Single short shake segment | Repeated shake or obvious device repositioning |

Artifact rule:

If high `xy_disp_norm` or high `angle_disp_rate` occurs together with bbox-size jumps, face-confidence drops, missing pose frames, or multi-face switching, classify as `FAIL_NO_USE_HEAD_MOVEMENT` unless manual review confirms valid participant movement.

Behavioural-signal rule:

If movement is high but face tracking is stable, pose remains in range, bbox size is stable, and no second face/camera movement is present, classify as `PASS_SECONDARY_WITH_FLAGS`, not automatic fail. This preserves possible agitation/restlessness signal while preventing technical artifacts from entering primary models.

### QC-4. Task-level completeness

| Task segment | Minimum usable duration | Minimum valid pose rate | Action if failed |
|---|---:|---:|---|
| Calibration / baseline | ≥10 s | ≥95% | Repeat before task starts |
| Reading task | ≥80% of expected duration | ≥90% | Use if pass; otherwise secondary/no-use |
| Image/scenario description | ≥60 s or ≥80% expected duration | ≥85% | Use with speech/task covariates |
| Attention-bias task | ≥90% expected duration | ≥90% | Fail head/gaze features if below threshold |
| Sustained phonation | ≥80% expected duration | ≥80% | Head movement secondary only |
| Full session | ≥85% valid pose coverage across all video tasks | ≥85% | Exclude head movement if below threshold |

### QC-5. Dataset-level monitoring

For each site, device, and task, report:

| Metric | Dataset monitoring rule |
|---|---|
| `valid_pose_rate` | Site/device median should be ≥90%; investigate if <85%. |
| `face_loss_gap_max_sec` | Investigate if site median >1 s. |
| `pose_invalid_fraction` | Investigate if site/device/task median >10%. |
| `p95(xy_disp_norm)` | Flag device/task if above pilot 95th percentile by >2×. |
| `p95(angle_disp_rate)` | Flag device/task if above pilot 95th percentile by >2×. |
| Multi-face rate | Should be near 0%; site retraining needed if >2%. |
| Camera shake/artifact rate | Should be <2%; investigate room/device setup if higher. |

## Required output fields

Each processed task/session should produce the following QC fields in addition to `head_movement_df` and `summary_df`:

| Field | Type | Meaning |
|---|---|---|
| `hm_qc_status` | enum | Final status: `PASS_PRIMARY`, `PASS_SECONDARY_WITH_FLAGS`, `MANUAL_REVIEW`, `FAIL_NO_USE_HEAD_MOVEMENT`, `FAIL_RECAPTURE_IF_POSSIBLE` |
| `hm_qc_reasons` | list[str] | Machine-readable reasons, e.g. `LOW_FACE_VISIBILITY`, `POSE_OUT_OF_RANGE`, `EXCESSIVE_TRACKER_JUMP` |
| `source_fps` | float | Original video FPS |
| `analysis_fps` | float | Head-movement sampling FPS |
| `sampled_frame_count` | int | Number of frames expected to be analysed |
| `valid_pose_frame_count` | int | Sampled frames with valid face pose |
| `valid_pose_rate` | float | Valid pose frames / sampled frames |
| `face_visible_rate` | float | Frames with visible participant face / sampled frames |
| `multi_face_fraction` | float | Fraction of frames with more than one detected face |
| `face_loss_gap_max_sec` | float | Longest continuous missing-face segment |
| `pose_primary_fraction` | float | Fraction of valid frames inside primary pose range |
| `pose_invalid_fraction` | float | Fraction of valid frames outside invalid pose thresholds |
| `p95_xy_disp_norm` | float | 95th percentile normalized bbox-center movement |
| `p95_angle_disp_rate` | float | 95th percentile angular movement rate |
| `bbox_size_jump_fraction` | float | Fraction of frames with unstable bbox size |
| `camera_artifact_flag` | bool | Camera/laptop movement suspected |
| `manual_review_required` | bool | Requires human review before dataset lock |

## Operational capture guidance

During live capture, the application should not continuously coach the participant during clinical tasks because this can alter natural behaviour. Use live prompts only before a task begins or when the face is lost long enough that the recording would otherwise become unusable.

Recommended online prompts:

| Trigger | UI/assistant action |
|---|---|
| No face for >2 s before task start | Ask participant to sit back in frame and restart calibration. |
| Head pose outside calibration range for >3 s before task start | Ask participant to face the screen/camera. |
| More than one face before task start | Ask assistant/clinician to leave the camera frame. |
| Camera/laptop moved before task start | Re-run calibration. |
| Face lost during task for >5 s | Mark QC flag; repeat only if protocol allows and participant is comfortable. |

## Recommendation for AIREST

### Use now

Use head movement in AIREST clinical data capture as a secondary/exploratory multimodal feature family and as a QC signal. Capture the video, run lightweight online QC, and compute final OpenWillis-style head movement features offline. Store both framewise outputs and compact summaries. Use `frames_per_second=15` or `30` and `normalize_by_bb_size=True` for cross-video comparison.

### Do not use now

Do not use head movement as a standalone diagnostic marker, primary endpoint, or automatic clinical decision feature in the MVP. Do not compare raw pixel `xy_disp` across devices or sites without normalization. Do not treat `face_confidence` as meaningful when `bbox_list` is supplied, because it is set to `1` by implementation. Do not include failed-QC head movement features in the primary modelling dataset.

### Future

After internal dry-runs and the first pilot dataset, recalibrate thresholds by site, camera, task type, source FPS, participant position, and clinical subgroup. Add velocity/acceleration, burstiness, movement entropy, speech-aligned movement, task-aligned avoidance/scanning markers, and camera-shake correction. Promote head movement from exploratory to model candidate only after it shows stable test-retest reliability, low site/device bias, and incremental predictive value over speech, facial expressivity, blink, and attention-bias features.

## Final decision

Recommendation: USE WITH QC FOR AIREST MVP.

Head movement should be collected and processed, but only released into the primary modelling table when face visibility, pose range, and excessive-movement QC pass. For MVP clinical data capture, the correct decision is not full no-use; it is controlled use as a secondary feature plus strong QC gating. It should move to a stronger future role only after AIREST-specific validation proves reliability, interpretability, and added value.
