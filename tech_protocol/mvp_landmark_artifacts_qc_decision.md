# MVP Landmark Artifacts, Storage Scope, and QC Decision

Reviewed on: 2026-05-27

## Purpose

This draft decision defines the minimum landmark-related artifact set for the AIREST MVP. It assumes the MVP goal is to run one scraped MediaPipe landmark stream and reuse it in both:

- AIREST gaze estimation, which needs refined iris landmarks.
- OpenWillis-face style facial landmark, displacement, and region-summary flows, which use the first 468 Face Mesh landmarks.

Related notes:

- [OpenWillis Landmark Output Schema Note](openwillis_landmark_schema_note.md)
- [AIREST Gaze and OpenWillis Landmark Compatibility Note](airest_gaze_openwillis_landmark_compatibility.md)
- [AIREST Technical Data-Capture Protocol](README.md)

## Draft Decision

The MVP should store both raw landmarks and derived summary features.

Raw landmarks should be the durable source artifact because they are reusable across gaze, blink, face-presence QC, mouth openness, OpenWillis-compatible displacement, and future derived features. Summary features should also be stored because downstream clinical/research consumers should not have to recompute every run, and because the exact feature version used for analysis must be auditable.

Minimum decision:

- Store one raw MediaPipe refined landmark stream per captured video/camera stream: 478 landmarks when `refine_landmarks=True`.
- Treat landmarks `0..467` as the OpenWillis-compatible base face mesh and map them to `lmk001..lmk468` only in derived OpenWillis-compatible artifacts.
- Treat landmarks `468..477` as refined iris/gaze landmarks and keep them out of OpenWillis displacement and region summaries.
- Store an OpenWillis-normalized 468-landmark derivative when facial displacement features are included in a run.
- Store frame-level QC and run-level QC sidecars for every landmark extraction.
- Store displacement summaries when there is enough valid, continuous frame coverage to justify temporal movement features.

Do not store only summary features for the MVP. Summary-only storage would prevent later repair of normalization, coordinate mapping, missingness, gaze/blink derivations, and displacement algorithms without rerunning video. It would also make it harder to minimize raw video retention while keeping useful face geometry.

## Minimum Artifact Set

### Required For MVP

| Artifact | Suggested path | Format | Scope | Keep? | Purpose | Minimal example |
| --- | --- | --- | --- | --- | --- | --- |
| Raw refined landmarks | `features/landmarks_raw_mediapipe.parquet` or `features/landmarks_raw_mediapipe.npz` | Long table or array plus metadata | One row per camera frame and landmark, or array shaped `(frames, 478, 3)` | Yes | Primary reusable source for gaze, iris, blink, and OpenWillis-compatible face features. | `{session_id:"s1", frame_idx:0, landmark_id_mediapipe:468, x:0.51, y:0.42, z:-0.01}` |
| Frame QC table | `qc/landmark_frame_qc.parquet` or `.csv` | Table | One row per expected camera frame | Yes | Face presence, landmark count, processing status, frame timing, and failure reason. | `{frame_idx:0, frame_read_success:true, face_detected:true, landmark_count:478, failure_reason:null}` |
| Landmark extraction metadata | `metadata/landmark_extraction_metadata.json` | JSON | One per extraction run | Yes | Model/library versions, detector settings, coordinate space, frame orientation, schema version, source media, and hashes. | `{mediapipe_version:"0.10.x", refine_landmarks:true, landmark_count:478, coordinate_space:"raw_mediapipe_refined_normalized"}` |
| OpenWillis-compatible landmark view | `features/landmarks_openwillis_468.parquet` or `.npz` | Wide or array | First 468 landmarks only | Yes if OpenWillis-face features are used | Stable `lmk001..lmk468` mapping for OpenWillis-style processing. | `{frame_idx:0, lmk001_x:0.49, lmk001_y:0.31, lmk468_z:-0.02}` |
| Normalized 468 landmarks | `features/landmarks_openwillis_468_normalized.parquet` or `.npz` | Wide or array | First 468 landmarks after OpenWillis transform | Yes if displacement is used | Coordinates centered/scaled consistently before displacement. | `{frame_idx:0, coordinate_space:"openwillis_centered_eye_scaled", lmk001_x:0.0, lmk144_x:-0.5, lmk373_x:0.5}` |
| Displacement frame features | `features/landmark_displacement_framewise.parquet` | Table | One row per valid frame | Yes if displacement is used | Per-frame `overall`, `upper_face`, `lower_face`, `lips`, `eyebrows`, and optional mouth openness. | `{frame_idx:1, overall_displacement:0.012, upper_face_displacement:0.006, lips_displacement:0.018, displacement_valid:true}` |
| Displacement summary | `features/landmark_displacement_summary.json` or `.csv` | Small table/JSON | One row per session/task/segment | Yes if displacement is used | Mean/std/sum or rate-normalized summaries for downstream modeling. | `{session_id:"s1", valid_displacement_frame_count:899, overall_displacement_mean:0.014, overall_displacement_std:0.005}` |
| Landmark QC summary | `qc/landmark_qc_summary.json` | JSON | One per session/task/segment | Yes | Pass/warn/fail, missingness rates, frame-drop rates, and validity denominators. | `{session_id:"s1", status:"pass", face_detection_rate:0.98, iris_landmark_complete_rate:0.94, frame_drop_ratio:0.01}` |

### Optional For MVP

| Artifact | Store when | Reason |
| --- | --- | --- |
| Raw video | Consent/regulatory policy allows, or during short retention/debug window | Useful for audit and reprocessing, but more privacy-sensitive than landmarks. |
| Visualization contact sheet | Dry-run, QA, or failed QC sessions | Helps human reviewers understand face visibility, landmark placement, and failure modes. |
| Per-landmark displacement table for all 468 landmarks | Research runs or debugging | High-dimensional; not needed for minimum downstream MVP if region summaries are sufficient. |
| Iris/gaze landmark sidecar | If raw 478 landmark artifact is stored as OpenWillis 468 plus extras instead of a single 478 array | Keeps iris IDs `468..477` available for gaze without contaminating OpenWillis features. |

## Storage Scope Recommendation

### What To Store

Store raw landmarks and summary features.

| Data class | Recommendation | Rationale |
| --- | --- | --- |
| Raw 478 landmarks | Store by default for MVP sessions when face/gaze features are in scope | Enables one extraction pass and reuse across gaze plus OpenWillis-face; less sensitive than raw video but still biometric. |
| Raw 468 OpenWillis-compatible view | Store as derived view or generate from raw 478 on demand | Makes mapping auditable and prevents off-by-one mistakes. |
| Normalized 468 landmarks | Store when displacement or OpenWillis-style expressivity is used | Displacement depends on coordinate space; storing it prevents silent recomputation drift. |
| Framewise displacement summaries | Store when enough valid frames exist | Needed for temporal expressivity features and QC review. |
| Aggregate summary features | Store by default | Lightweight, stable inputs for analytics and dashboards. |
| Raw video | Do not store by default unless policy permits | Retain landmarks/QC as the privacy-minimized default; raw video can be temporary or consent-gated. |

### Retention And Privacy

Landmarks are derived biometric data and should be treated as sensitive. The MVP should:

- Store landmarks in a session-scoped protected directory.
- Include checksums for landmark and QC files.
- Avoid putting participant identity in filenames.
- Version schemas and feature algorithms.
- Encrypt or access-control landmark artifacts under the same policy as other derived clinical data.
- If raw video is not retained, record whether landmarks are the durable source for reanalysis.

## Coordinate Spaces To Version

Every landmark artifact must declare its coordinate space.

| Coordinate space | Applies to | Definition |
| --- | --- | --- |
| `raw_mediapipe_refined_normalized` | Raw 478 artifact | MediaPipe x/y normalized to processed frame, z relative depth, IDs `0..477`. |
| `raw_mediapipe_base_normalized` | OpenWillis-compatible raw 468 view | Same as raw MediaPipe, filtered to IDs `0..467` and optionally relabeled `lmk001..lmk468`. |
| `openwillis_centered_eye_scaled` | Normalized 468 artifact | Center by `lmk001` and scale by 3D distance between `lmk144` and `lmk373`. |
| `openwillis_centered_aligned_eye_scaled` | Optional aligned 468 artifact | Same as above, plus z-axis rotation so the eye line is horizontal. |
| `frame_pixel_eye_geometry` | Gaze-ratio features | Selected eye/iris points converted to frame pixels for gaze ratio derivation. |
| `screen_pixel_gaze` | Point-of-gaze outputs | Calibrated screen coordinates after polynomial mapping. |

Required metadata:

- `coordinate_space`
- `landmark_count`
- `refine_landmarks`
- `mediapipe_version`
- `model_or_solution_name`
- `processed_frame_orientation`
- `image_transform`, for example `horizontal_flip_before_mediapipe`
- `frame_width`, `frame_height`
- `source_media_id` and checksum
- `schema_version`
- `normalization_method`, if normalized
- `alignment_method`, if aligned

## Proposed Minimum Schemas

### Raw Landmark Long Table

Recommended for CSV/Parquet interoperability:

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `session_id` | string | yes | Session identifier. |
| `camera_stream_id` | string | yes | Video/camera stream identifier. |
| `frame_idx` | integer | yes | Expected camera frame index, not stimulus frame. |
| `frame_time_sec` | number or null | yes | Camera/media time when available. |
| `capture_monotonic_time_sec` | number or null | recommended | Local monotonic timestamp near frame acquisition. |
| `landmark_id_mediapipe` | integer | yes | MediaPipe index, `0..477` for refined FaceMesh. |
| `openwillis_landmark` | string or null | yes | `lmk001..lmk468` for IDs `0..467`; null for IDs `468..477`. |
| `is_refined_iris_landmark` | boolean | yes | True for IDs `468..477`. |
| `x` | number or null | yes | Landmark x in declared coordinate space. |
| `y` | number or null | yes | Landmark y in declared coordinate space. |
| `z` | number or null | yes | Landmark z in declared coordinate space. |
| `coordinate_space` | string | yes | Coordinate-space enum. |
| `face_id` | integer or null | recommended | Selected face index when multi-face detection is possible. |
| `landmark_valid` | boolean | yes | False when the landmark is missing, stale, or invalid. |

### Frame QC Table

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `session_id` | string | yes | Session identifier. |
| `camera_stream_id` | string | yes | Video/camera stream identifier. |
| `frame_idx` | integer | yes | Expected camera frame index. |
| `frame_time_sec` | number or null | yes | Camera/media time when available. |
| `frame_read_success` | boolean | yes | Whether a camera/video frame was available. |
| `frame_duplicate` | boolean | recommended | Whether the frame appears repeated/frozen. |
| `face_detected` | boolean | yes | Whether MediaPipe returned a face for this frame. |
| `selected_face_count` | integer | recommended | Number of faces returned/considered. |
| `landmark_count` | integer | yes | Number of current landmarks returned, expected `478` for refined mode. |
| `base_landmark_count` | integer | yes | Count among IDs `0..467`, expected `468` when face detected. |
| `iris_landmark_count` | integer | yes | Count among IDs `468..477`, expected `10` in refined mode. |
| `landmarks_current` | boolean | yes | False if landmarks are stale or copied from a previous frame. |
| `tracking_confidence` | number or null | recommended | Store if available from the detector/runtime; null when not exposed. |
| `face_bbox_x`, `face_bbox_y`, `face_bbox_w`, `face_bbox_h` | number or null | recommended | Derived bbox from landmark extents or detector output. |
| `failure_reason` | string or null | yes | Examples: `camera_read_failed`, `no_face`, `partial_landmarks`, `stale_landmarks`, `processing_error`. |

MediaPipe FaceMesh solution APIs do not expose a stable per-frame tracking-confidence value in the current AIREST/OpenWillis code paths. For MVP, include the `tracking_confidence` field but allow null. Use face detection status, landmark count, landmark-current status, bbox stability, and frame timing as the practical QC signals unless the detector is changed to a runtime that exposes confidence.

### Displacement Summary

Store framewise and aggregate displacement only for OpenWillis-compatible base landmarks `0..467`.

Framewise fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `frame_idx` | integer | Camera frame index. |
| `frame_time_sec` | number | Camera/media time. |
| `overall_displacement` | number or null | Mean movement across 468 base landmarks. |
| `upper_face_displacement` | number or null | Mean movement for OpenWillis upper-face region. |
| `lower_face_displacement` | number or null | Mean movement for OpenWillis lower-face region. |
| `lips_displacement` | number or null | Mean movement for OpenWillis lips region. |
| `eyebrows_displacement` | number or null | Mean movement for OpenWillis eyebrows region. |
| `mouth_openness` | number or null | Optional ratio feature derived from lip landmarks. |
| `displacement_valid` | boolean | False if current or previous frame lacks valid landmarks. |

Aggregate fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `valid_displacement_frame_count` | integer | Frames with valid current and previous landmarks. |
| `overall_displacement_mean` | number or null | Mean over valid displacement frames. |
| `overall_displacement_std` | number or null | Standard deviation over valid displacement frames. |
| `upper_face_displacement_mean/std` | number or null | Region aggregate. |
| `lower_face_displacement_mean/std` | number or null | Region aggregate. |
| `lips_displacement_mean/std` | number or null | Region aggregate. |
| `eyebrows_displacement_mean/std` | number or null | Region aggregate. |
| `mouth_openness_mean/std` | number or null | Optional mouth openness aggregate. |
| `baseline_used` | boolean | Whether baseline correction was applied. |
| `normalization_method` | string | Coordinate transform used before displacement. |

## Minimum QC Metrics

### Required Run-Level Metrics

| Metric | Formula | MVP interpretation |
| --- | --- | --- |
| `expected_frame_count` | Count from capture/video timeline | Denominator for frame availability. |
| `processed_frame_count` | Frames attempted by landmark extractor | Confirms extraction coverage. |
| `frame_read_success_rate` | `frame_read_success_count / expected_frame_count` | Camera/file availability. |
| `frame_drop_ratio` | `1 - frame_read_success_count / expected_frame_count` or dropped-frame count from capture timestamps | Capture stability; should be distinct from face loss. |
| `face_detection_rate` | `face_detected_frame_count / frame_read_success_count` | Core landmark usability metric. |
| `missing_landmark_ratio` | Missing expected landmark values / expected landmark values on processed frames | Overall landmark completeness. |
| `base_landmark_complete_rate` | Frames with all `0..467` valid / frame_read_success_count | OpenWillis-face readiness. |
| `iris_landmark_complete_rate` | Frames with all `468..477` valid / frame_read_success_count | Gaze/iris readiness. |
| `stale_landmark_frame_ratio` | `stale_landmark_frame_count / processed_frame_count` | Must be zero for valid production exports. |
| `valid_displacement_ratio` | `valid_displacement_frame_count / max(face_detected_frame_count - 1, 1)` | Whether temporal movement summaries are meaningful. |
| `median_observed_fps` | Median inverse timestamp delta | Actual capture cadence. |
| `fps_jitter` | Robust std/IQR of frame timestamp deltas | Detects unstable capture. |
| `frozen_frame_ratio` | Repeated or near-identical frames / expected frames | Detects camera freezes. |
| `bbox_size_median` | Median bbox area / frame area | Face too small/large QC. |
| `bbox_center_jitter` | Robust movement of bbox center | Flags camera shake or tracking instability. |

### Recommended Pass/Warn/Fail Thresholds For Dry-Run MVP

These are starting thresholds for dry-runs, not final clinical validity thresholds.

| Metric | Pass | Warn | Fail |
| --- | ---: | ---: | ---: |
| `frame_read_success_rate` | `>= 0.98` | `0.95 - 0.98` | `< 0.95` |
| `frame_drop_ratio` | `<= 0.02` | `0.02 - 0.05` | `> 0.05` |
| `face_detection_rate` | `>= 0.95` | `0.90 - 0.95` | `< 0.90` |
| `base_landmark_complete_rate` | `>= 0.95` | `0.90 - 0.95` | `< 0.90` |
| `iris_landmark_complete_rate` | `>= 0.90` | `0.80 - 0.90` | `< 0.80` |
| `missing_landmark_ratio` | `<= 0.02` | `0.02 - 0.05` | `> 0.05` |
| `stale_landmark_frame_ratio` | `0.00` | `> 0 and <= 0.01` | `> 0.01` |
| `valid_displacement_ratio` | `>= 0.90` | `0.80 - 0.90` | `< 0.80` |
| `median_observed_fps` for 30 fps target | `>= 28` | `25 - 28` | `< 25` |
| `frozen_frame_ratio` | `<= 0.01` | `0.01 - 0.03` | `> 0.03` |

Task-specific thresholds may differ. For example, gaze calibration should fail on low iris completeness even when base face landmarks are usable for facial displacement.

## QC Decision Logic

Minimum MVP status labels:

| Status | Meaning | Action |
| --- | --- | --- |
| `pass` | Landmark coverage and frame timing are sufficient for intended features | Use raw landmarks and derived features. |
| `warn` | Features may be usable but require review or confidence down-weighting | Store artifacts, mark feature family as `needs_review`. |
| `fail_face_landmarks` | Base landmarks are insufficient | Exclude OpenWillis-face displacement and region summaries. |
| `fail_iris_landmarks` | Refined iris landmarks are insufficient | Exclude or invalidate gaze/iris-derived features. |
| `fail_timing` | Dropped/frozen frames make temporal features unreliable | Exclude displacement/rate features; keep static frame-level landmarks if useful. |
| `fail_stale_landmarks` | Extractor reused old landmarks after face loss | Invalidate affected landmark and displacement artifacts until pipeline is fixed. |

Feature-family gates:

- Gaze requires passing iris landmark completeness, calibration validity, and timing checks.
- OpenWillis displacement requires passing base landmark completeness, continuity, and timing checks.
- Face presence QC can be reported even when displacement or gaze fails.
- Raw landmarks can be retained for audit/reprocessing even when derived features are marked invalid, provided the artifact clearly records failure status.

## Implementation Notes For AIREST

The current AIREST gaze pipeline should be adjusted before MVP storage:

- Run MediaPipe once per camera frame and share that result with gaze and OpenWillis-face feature derivation.
- Clear landmarks on no-face frames; never reuse stale landmarks as current-frame data.
- Emit frame QC rows for failed frames instead of dropping them from landmark files.
- Keep MediaPipe IDs `0..477` in the raw source artifact.
- Add `openwillis_landmark` labels only for IDs `0..467`.
- Derive OpenWillis-compatible normalized landmarks from the raw source, not from a second MediaPipe pass.
- Compute displacement only on normalized 468 base landmarks and only across valid consecutive camera frames.
- Store both camera-frame timing and stimulus/task timing so gaze and displacement can be aligned without overloading one frame counter.

## Final MVP Recommendation

Use raw refined MediaPipe landmarks as the canonical reusable landmark source artifact, and store derived OpenWillis-compatible 468-landmark features as versioned derivatives.

This gives AIREST a single extraction path for gaze and face features without forcing OpenWillis to adopt 478 landmarks or forcing gaze estimation to drop iris points. It also supports privacy-aware raw-video minimization: if raw video is not retained, raw landmarks plus QC sidecars remain sufficient for most face/gaze reanalysis and audit needs.
