# Technical Red Flags and Compute Bottlenecks Report

## Scope

This report reviews compute and reliability risks across the local OpenWillis face stack used by:

- `head_movement/`
- `emotional_expressivity/`
- `facial_expression/`
- `openwillis-face/src/openwillis/face/`

The goal is to identify what can influence feature computation, where the main bottlenecks are, and how the implementation should be improved or rewritten for an AIREST-style pipeline.

## Executive Summary

The current code is suitable for offline demos and research validation, but it should not be used as a realtime production dependency. The largest compute risks come from repeated full-video decoding, heavy per-frame model inference, broad exception handling, inconsistent sampling semantics, and dataframe-heavy processing over wide landmark tables.

Recommended direction:

1. Keep realtime capture lightweight: video recording, face presence, basic camera/pose QC, timestamps, and optional raw MediaPipe landmarks.
2. Move feature extraction into a structured offline post-processing stage.
3. Decode each video once per post-processing run and share intermediate artifacts across head movement, facial expressivity, blink, mouth, and optional emotion/AU features.
4. Treat py-feat emotion/AU extraction as optional research-only processing until latency, baseline handling, QC metadata, and dependency risks are resolved.

## Highest-Priority Red Flags

| Priority | Area | Current behavior | Compute or reliability impact | Rewrite direction |
| --- | --- | --- | --- | --- |
| P0 | Pipeline architecture | Each feature function opens and reads video independently with `cv2.VideoCapture`. | Repeated video decode and repeated model passes multiply runtime. | Create one offline frame-processing orchestrator that emits shared landmarks, bbox tracks, timestamps, QC, and feature sidecars. |
| P0 | Realtime suitability | Functions are file-based and return only after the whole video is processed. | Not usable for low-latency capture feedback or session gating. | Keep realtime QC separate from offline feature extraction. |
| P0 | Error handling | Broad `except Exception` blocks log errors and often return empty, partial, or `None` outputs. | Downstream code cannot distinguish true low expressivity from failed tracking or processing. | Return structured status, warnings, failed-frame counts, and exception summaries. |
| P0 | QC metadata | Missingness, sampled-frame count, model versions, bbox quality, and baseline status are not public outputs. | Features can be misinterpreted, especially in clinical or research datasets. | Add per-run metadata and QC artifacts for every feature family. |
| P1 | Sampling semantics | `head_movement` uses target fps, `emotional_expressivity` uses `skip_frames`, and `facial_expressivity` ignores `frames_per_second`. | Feature values are not directly comparable across functions or source videos. | Standardize on `analysis_fps`, `sampled_frame_idx`, and rate-normalized displacement features. |
| P1 | py-feat emotion/AU stack | Per sampled frame, code runs face detection, landmarks, AUs, and emotions. | High latency and heavy dependency footprint. | Run only as isolated offline research worker; reuse bbox/crops where possible. |
| P1 | Pandas-heavy landmark math | Landmark displacement loops over 468 columns and concatenates many DataFrames. | Unnecessary CPU and memory overhead for long videos. | Use NumPy arrays shaped `(frames, landmarks, xyz)` and build output DataFrames once. |
| P1 | Baseline logic | Baseline-normalized framewise and summary values can be on different scales. | Baseline outputs can be misleading or internally inconsistent. | Rewrite baseline transforms with explicit mode flags and apply summaries after transformation. |
| P1 | Bbox semantics | Bbox-assisted head movement overwrites confidence with `1`; one bbox helper likely computes y2 as `bb_y + bb_y`. | Confidence is not meaningful and bad boxes can corrupt crops. | Fix bbox conversion and separate `bbox_source`, `bbox_confidence`, and `pose_confidence`. |

## Compute Bottlenecks by Module

### 1. `head_movement`

Relevant code:

- `openwillis-face/src/openwillis/face/head_movement.py`
- `openwillis-face/src/openwillis/face/preprocess_video.py`
- `openwillis-face/src/openwillis/face/util/crop_utils.py`

Main bottlenecks:

| Location | Problem | Why it matters | Improvement |
| --- | --- | --- | --- |
| `extract_landmarks_and_bboxes()` | Creates `feat.Detector()` and runs pose detection on sampled frames. | py-feat pose estimation dominates runtime. | Reuse detector instances consistently or run pose extraction in an isolated worker. |
| `extract_landmarks_and_bboxes()` | Reads every frame even when only sparse frames are analyzed. | Decode cost remains full-video cost. | Use frame seeking or a shared decoder when sampling is sparse. |
| `crop_and_get_facepose()` | Crops around supplied bbox, then overwrites detector bbox/confidence. | `face_confidence` becomes synthetic and not comparable with detector confidence. | Add explicit columns: `bbox_source`, `bbox_confidence`, `pose_detected`. |
| `compute_xy_disp()` | Displacement is per sampled interval but not normalized by time. | Different `frames_per_second` values change metric scale. | Add `xy_disp_rate = xy_disp / frame_delta_seconds`. |
| `euclidean_angle_disp` | Angle delta is not divided by time. | Comparing 3 fps and 30 fps runs is misleading. | Add `angle_disp_rate_deg_s`. |

Rewrite target:

- Keep `head_movement()` as a public compatibility wrapper.
- Internally route through a shared `FacePostprocessContext` containing video metadata, frame timestamps, bbox tracks, and optional cached crops.
- Produce both legacy columns and new QC/rate columns.

Suggested output additions:

- `source_fps`
- `analysis_fps`
- `sampled_frame_count`
- `valid_pose_frame_count`
- `valid_pose_rate`
- `bbox_source`
- `xy_disp_norm`
- `xy_disp_rate`
- `angle_disp_rate_deg_s`
- `bbox_width_jump`
- `bbox_height_jump`

### 2. `emotional_expressivity`

Relevant code:

- `openwillis-face/src/openwillis/face/facial_emotion.py`
- `openwillis-face/src/openwillis/face/util/speaking_utils.py`

Main bottlenecks:

| Location | Problem | Why it matters | Improvement |
| --- | --- | --- | --- |
| `detect_emotions()` | Runs `detect_faces`, `detect_landmarks`, `detect_aus`, and `detect_emotions` per sampled frame. | This is the heaviest pipeline in the repo. | Keep as offline research-only; batch and reuse bbox/crops where possible. |
| `run_pyfeat()` | Builds one small DataFrame per frame, then concatenates. | DataFrame churn adds overhead. | Collect rows as arrays/dicts and create one DataFrame at the end. |
| `baseline()` | Calls `get_emotion()` again for baseline video. | Doubles model inference when baseline is enabled. | Cache baseline features by file hash and parameters. |
| `emotional_expressivity()` | Calls `dropna()` after processing. | Failed sampled frames disappear without public accounting. | Preserve a QC table and report expected vs successful sampled frames. |
| `get_speaking_probabilities()` | Fits a GMM over mouth-motion proxy. | Adds compute and can be unstable on short/low-motion samples. | Prefer audio VAD or transcript alignment for production speaking splits. |
| `bb_dict_to_bb_list()` | Uses `bb_y + bb_y` for y2 instead of likely `bb_y + bb_h`. | Can create invalid boxes if this helper is used. | Fix conversion and add unit tests. |

Rewrite target:

- Rename or wrap this as an optional `offline_emotion_au_extraction` stage.
- Require explicit output metadata that labels these as model-derived scores, not true emotions.
- Disable baseline mode by default until fixed.

Suggested output additions:

- `n_source_frames`
- `skip_frames`
- `analysis_fps_effective`
- `expected_sampled_frames`
- `successful_sampled_frames`
- `failed_sampled_frames`
- `baseline_requested`
- `baseline_used`
- `model_stack`
- `processing_status`

Baseline rewrite requirements:

1. Fail loudly or mark `baseline_used=false` when the baseline path is missing.
2. Compute summary from the same transformed values returned in framewise output.
3. Do not apply emotion/AU baseline shifts to `mouth_openness`.
4. Add stable handling for near-zero baseline means.
5. Store baseline file hash, duration, fps, valid-frame rate, and model parameters.

### 3. `facial_expressivity`

Relevant code:

- `openwillis-face/src/openwillis/face/face_landmark.py`
- `facial_expression/facial_expressivity_feature_inventory.md`

Main bottlenecks:

| Location | Problem | Why it matters | Improvement |
| --- | --- | --- | --- |
| `run_facemesh()` | Processes every native video frame. | Runtime grows with full source fps and duration. | Respect `frames_per_second` or make native-fps behavior explicit. |
| `get_distance()` | Loops over 468 landmarks and concatenates 468 DataFrames. | Avoidable CPU and memory overhead. | Vectorize with NumPy and build one DataFrame. |
| `normalize_face_landmarks()` | Works on very wide DataFrames. | DataFrame operations over 1400+ columns are expensive. | Normalize in array form before DataFrame conversion. |
| `_LANDMARK_CACHE` | Global in-memory cache has no size or invalidation policy. | Long sessions or many videos can grow memory unexpectedly. | Use bounded cache or disk-backed artifacts keyed by file hash and params. |
| `baseline()` | Creates normalized baseline landmarks but computes displacement from raw `base_landmark`. | Baseline normalization may not match main path. | Apply displacement to the same representation used for the main run. |
| `facial_expressivity()` | `frames_per_second` argument is compatibility-only and ignored. | Caller expectations can be wrong. | Implement sampling or remove/rename the parameter. |

Rewrite target:

- Make MediaPipe landmarks the main reusable artifact for all lightweight face features.
- Store raw landmarks once, then derive normalized landmarks, displacement, mouth openness, region summaries, blink, and optional head/gaze summaries from that artifact.
- Convert to DataFrame only at API/export boundaries.

Suggested internal representation:

```text
landmarks_xyz: float32 array with shape (n_frames, 468, 3)
frame_idx: int array with shape (n_frames,)
timestamp_sec: float array with shape (n_frames,)
face_detected: bool array with shape (n_frames,)
visibility/confidence fields where available
```

Suggested output additions:

- `frame_idx`
- `timestamp_sec`
- `face_detected`
- `landmark_valid`
- `landmark_missing_rate`
- `overall_disp_rate`
- `lower_face_disp_rate`
- `upper_face_disp_rate`
- `lips_disp_rate`
- `eyebrows_disp_rate`
- `mouth_openness`

## Proposed Rewrite Structure

### New post-processing layout

```text
openwillis-face/src/openwillis/face/
  pipeline/
    context.py              # video metadata, params, shared artifact handles
    decode.py               # frame decoding, timestamps, frame sampling
    qc.py                   # missingness, bbox stability, pose validity, status labels
    artifacts.py            # parquet/npz/json read/write helpers
  features/
    landmarks.py            # MediaPipe landmark extraction
    landmark_movement.py    # normalization, displacement, region summaries
    head_pose.py            # py-feat or alternative head pose extraction
    emotion_au.py           # optional py-feat emotion/AU extraction
    speaking.py             # audio/VAD-backed or mouth-proxy speaking split
  api/
    compatibility.py        # current public wrappers: facial_expressivity, emotional_expressivity, head_movement
```

### Processing flow

```text
video file
  -> decode metadata and frame timestamps once
  -> lightweight face/landmark extraction
  -> shared artifacts:
       landmarks.npz
       bbox_tracks.parquet
       face_qc.json
  -> derived lightweight features:
       facial_expressivity_summary.csv
       head_movement_summary.csv
       blink_summary.csv
  -> optional heavy features:
       emotional_expressivity_framewise.csv
       emotional_expressivity_summary.csv
       emotional_expressivity_qc.json
```

## Concrete Rewrite Plan

### Phase 1: Low-risk fixes

| Task | Files | Expected benefit |
| --- | --- | --- |
| Fix bbox y2 conversion bug. | `facial_emotion.py` | Prevent invalid bbox conversion. |
| Add structured QC return or sidecar. | `facial_emotion.py`, `face_landmark.py`, `head_movement.py` | Make failures visible. |
| Make baseline mode explicit. | `facial_emotion.py`, `face_landmark.py` | Avoid mislabeled baseline outputs. |
| Add rate-normalized displacement fields. | `head_movement.py`, `face_landmark.py` | Improve comparability across fps settings. |
| Add unit tests for bbox conversion, sampling, first-row displacement, and baseline missing path. | `tests/` | Lock basic behavior. |

### Phase 2: Compute optimization

| Task | Files | Expected benefit |
| --- | --- | --- |
| Vectorize landmark displacement. | `face_landmark.py` | Lower CPU and memory overhead. |
| Replace per-frame DataFrame construction in emotion pipeline. | `facial_emotion.py` | Lower pandas overhead. |
| Add bounded or disk-backed landmark cache. | `face_landmark.py` or new artifact helper | Prevent unbounded memory growth. |
| Reuse video metadata and frame sampling utilities. | New `pipeline/decode.py` | Consistent sampling and fewer duplicate helpers. |

### Phase 3: Architecture rewrite

| Task | Files | Expected benefit |
| --- | --- | --- |
| Add shared post-processing context. | New `pipeline/context.py` | One coherent offline pipeline. |
| Store intermediate landmarks and bbox tracks. | New `pipeline/artifacts.py` | Avoid repeated model inference. |
| Move py-feat emotion/AU into optional worker. | New `features/emotion_au.py` | Isolate slow and fragile dependency stack. |
| Preserve existing public API through wrappers. | New `api/compatibility.py` | Backward compatibility while internals improve. |

## Production Policy Recommendation

| Feature family | Recommended production role | Rationale |
| --- | --- | --- |
| Realtime face QC | Use in production | Lightweight and necessary for capture quality. |
| MediaPipe landmarks | Use as primary raw face artifact | Reusable for many derived features. |
| Facial movement summaries | Use offline | Useful and cheaper than py-feat emotion/AU. |
| Head movement | Use offline with QC | Valuable as secondary feature, but sensitive to bbox jitter and pose failure. |
| Emotional expressivity / py-feat emotions and AUs | Research-only offline | Heavy runtime, fragile baseline behavior, and clinical interpretation risk. |
| Baseline-relative features | Disabled until protocol and code are fixed | Requires neutral baseline definition and robust implementation. |

## Minimum QC Contract

Every offline feature run should emit a metadata or QC artifact with:

- `input_video_path`
- `input_video_hash`
- `source_fps`
- `source_frame_count`
- `duration_sec`
- `analysis_fps`
- `sampled_frame_count`
- `successful_frame_count`
- `failed_frame_count`
- `valid_face_rate`
- `max_face_loss_gap_sec`
- `bbox_source`
- `bbox_stability_metrics`
- `baseline_requested`
- `baseline_used`
- `model_versions`
- `runtime_seconds`
- `processing_status`
- `warnings`
- `errors`

## Recommended Acceptance Criteria

Before this stack is promoted beyond offline research validation:

1. A 20 to 30 second 1080p sample should run with predictable wall time for each feature family.
2. Running all lightweight face features together should not decode the same video more than once.
3. All public outputs should include QC metadata or link to a QC sidecar.
4. Missing baseline files should never silently produce outputs that look baseline-normalized.
5. Sampling should be expressed as effective `analysis_fps` and `frame_delta_seconds`.
6. Displacement metrics used across videos should include rate-normalized and face-size-normalized variants.
7. py-feat emotion/AU output should be optional and clearly marked as model-derived research features.

