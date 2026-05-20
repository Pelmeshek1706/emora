# AIREST vs OpenWillis Facial Expressivity Feature Decision Matrix

## Scope

This matrix compares local OpenWillis face outputs from `facial_expression/` and `demo_openwillis_face.ipynb` with the current AIREST MVP documentation and the referenced `Pelmeshek1706/phonova` `refinement_speech` branch.

Key interpretation:

- AIREST Confluence currently covers local/offline clinical data capture, raw WAV/video recording, MediaPipe-based facial feature extraction, `facial_features.csv`, QC, calibration, live status, and export packaging.
- The `phonova` branch is speech/transcript analytics only. It covers speech structure, pauses, repetition, coherence, sentiment, and first-person language features; it does not implement facial features.
- OpenWillis provides richer facial expressivity outputs than the current AIREST docs specify at column/schema level.

## Decision Matrix

| Feature / output family | OpenWillis output | AIREST current coverage | Decision status | Offline post-processing candidate? | Realtime pipeline impact | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| Raw video capture | Input dependency for all OpenWillis face functions | Covered: full-session raw video recording and required `video.mp4`/`video.webm` artifact | Already covered | No, capture must happen during session | Core realtime/session function | Keep as required artifact; ensure video metadata includes FPS, frame count, codec, dimensions |
| Frame/time index | `frame`, `time` in framewise tables | Implied by `facial_features.csv`, not explicitly columned | Partially covered | No, should be emitted with every feature row | Minimal | Define mandatory `frame_idx`/`timestamp_sec` columns in AIREST face schema |
| Face detected / tracking quality | OpenWillis silently yields missing/NaN rows when detection fails | Covered: system check, live `face_detected`, QC face-detection percentage, face-lost warnings | Already covered for QC, not equivalent to OpenWillis metrics | No, QC needs realtime signal | Required for pausing/warnings | Keep realtime; persist per-frame `face_detected`, confidence, and QC aggregate |
| 468 FaceMesh landmarks | `framewise_loc`: `lmk001_x...lmk468_z` plus frame/time | Covered at feature-family level: MediaPipe facial landmarks, `facial_features.csv`; exact columns not specified | Covered conceptually, missing schema | Yes for normalized derivatives; raw landmarks should be persisted during capture if storage permits | Moderate if writing all columns live; low if buffered | Persist raw MediaPipe landmarks or a compact parquet/npz sidecar; derive OpenWillis-compatible CSV offline |
| Normalized landmarks | Centered/scaled, optional eye alignment | Not specified | Missing | Yes | None if derived after session | Add offline normalization step using stored landmarks; do not block realtime recording |
| Per-landmark displacement | `framewise_disp`: `lmk001...lmk468` frame-to-frame movement | Not specified | Missing | Yes | None if computed from stored landmarks | Add post-session displacement computation; only write summary to QC if needed |
| Composite facial movement | `overall`, `lower_face`, `upper_face`, `lips`, `eyebrows` framewise and summary mean/std | Not specified | Missing | Yes | None | Add as offline summary features; these are high-value, low-risk additions from landmark data |
| Mouth openness | `mouth_openness` framewise and mean/std summary | Possibly adjacent to MediaPipe/emotion-related outputs; no explicit AIREST schema | Missing as explicit feature | Yes; optional realtime if used for speech/face QC | Low if using landmarks already in memory | Add offline feature first; consider realtime only if needed for speaking-state feedback |
| Speaking probability split | Optional `speaking_probability`; summary split into speaking/not-speaking variants | Not specified; AIREST has audio recording and Phonova speech analytics | Missing | Yes | None | Compute offline using mouth-openness and/or audio VAD; align with Phonova transcript turns later |
| Baseline-normalized facial movement | Relative-to-baseline movement ratios when baseline video exists | AIREST has calibration, but no neutral-expression baseline clip contract | Missing | Yes, if protocol captures baseline/calibration video | None | Add only if clinical protocol includes a neutral baseline segment; otherwise keep raw/normalized-within-video features |
| 7 emotion scores | `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutral` | Architecture mentions emotion-related frame-level outputs, but no implementation/schema in docs or `phonova` | Planned/placeholder only | Yes, strongly | High if run realtime; local notebook showed py-feat was slow relative to video duration | Treat as offline post-processing; avoid blocking session recording or live face QC |
| Facial action units | `AU01`, `AU02`, `AU04`, `AU05`, `AU06`, `AU07`, `AU09`, `AU10`, `AU11`, `AU12`, `AU14`, `AU15`, `AU17`, `AU20`, `AU23`, `AU24`, `AU25`, `AU26`, `AU28`, `AU43` | Same as emotion scores: mentioned as emotion-related family only, not schema/code | Planned/placeholder only | Yes, strongly | High if py-feat model stack runs realtime | Add offline AU extraction from recorded video; store framewise and summary mean/std |
| Emotional expressivity summary | Mean/std for 7 emotions, 20 AUs, `mouth_openness`; optional speaking split | Not specified | Missing | Yes | None | Add as derived summary table, separate from realtime `facial_features.csv` |
| Blink / eye aspect ratio | Notebook includes `eye_blink_rate`: EAR framewise, blink events, `blinks`, `blink_rate` | Architecture mentions blink outputs; SRS/QC mentions face detection, not blink schema | Covered conceptually, missing schema | Yes, unless used for liveness/QC | Low to moderate | Define blink event table and session-level blink rate; keep out of blocking realtime path unless clinically required |
| Gaze/head pose | Not a main OpenWillis facial expressivity output reviewed here | AIREST architecture explicitly mentions gaze/head frame-level outputs | AIREST-only, not covered by OpenWillis matrix | Yes for summaries; calibration may need realtime gaze | Moderate if calibration uses gaze live | Keep current AIREST gaze/head plan; do not depend on OpenWillis for this feature family |
| Speech/text features | Not facial; useful for multimodal merge | Covered by `phonova`: structure, pause, repetition, coherence, sentiment, first-person features over transcripts | Already covered by code sample | Yes | None for face pipeline | Keep separate from face pipeline; join offline by session/task/time windows |
| Feature export contract | OpenWillis returns Python dataframes; local docs list shapes and columns | AIREST requires `features/facial_features.csv`, QC, metadata, checksums; schema is TBD | Partially covered | N/A | Low | Define versioned schema: raw realtime face features, derived offline facial expressivity, and summary aggregates |

## Recommendation

Use OpenWillis facial expressivity as a **validation reference**, not as a production runtime dependency.

AIREST should own the production face pipeline, schema, QC behavior, and session lifecycle. OpenWillis is useful as an external reference implementation for landmark-derived movement features and as a benchmark for comparing feature distributions, but it should not sit on the critical path for clinical session recording or realtime face QC.

Use the heavier OpenWillis `emotional_expressivity` / py-feat outputs as a **research-only benchmark** until latency, dependency, baseline-normalization, and clinical-validity risks are resolved.

## Decision Note

| Option | Decision | Rationale |
| --- | --- | --- |
| Production dependency | Do not use | Too much operational risk for the MVP: heavy model stack, narrow Python/Torch compatibility, slow emotion/AU extraction, unclear failure contracts, and fragile baseline behavior. |
| Validation reference | Use | Good fit for validating AIREST-derived landmark movement features: normalized landmarks, per-landmark displacement, region movement summaries, mouth openness, and speaking split logic. |
| Research-only benchmark | Use for py-feat emotion/AU features | Emotion labels and AUs are analytically interesting but should not block recording, QC, export, or clinical-start readiness. |

## MVP Feature Subset

| Feature group | MVP decision | Production source | Notes |
| --- | --- | --- | --- |
| Raw session video | Include | AIREST recording controller | Required source artifact; independent of OpenWillis. |
| Per-frame face QC | Include | AIREST realtime MediaPipe process | Persist `frame_idx`, `timestamp_sec`, `face_detected`, confidence/visibility if available, face-centered status, dropped-frame spans. |
| Raw or compact MediaPipe landmarks | Include if storage/performance is acceptable | AIREST realtime MediaPipe process | Store enough data to reproduce derived features offline. Prefer a versioned sidecar format if CSV is too wide. |
| Normalized landmarks | Add offline | AIREST post-processing, validated against OpenWillis | Center/scale/optional eye alignment after session completion. |
| Landmark displacement | Add offline | AIREST post-processing, validated against OpenWillis | Compute frame-to-frame displacement; do not require realtime. |
| Region movement summaries | Add offline MVP feature | AIREST post-processing, validated against OpenWillis | Include `overall`, `upper_face`, `lower_face`, `lips`, `eyebrows` mean/std. |
| Mouth openness | Add offline MVP feature | AIREST post-processing | Can later become realtime if needed for speaking-state feedback. |
| Blink rate / blink events | Add offline MVP-adjacent feature | AIREST post-processing | Useful and cheap from landmarks, but not required for session gating unless protocol demands it. |
| Speaking vs non-speaking facial summaries | Defer to post-MVP or offline optional | AIREST post-processing + audio/VAD/transcript alignment | Valuable for analysis, but needs a stable alignment contract. |
| Baseline-relative features | Exclude unless protocol adds neutral baseline | AIREST post-processing, OpenWillis reference only | Do not claim baseline-normalized reactivity without an explicit baseline segment. |
| Emotion scores | Research only | Offline benchmark, not production MVP | Keep out of MVP decision features. |
| Action units | Research only | Offline benchmark, not production MVP | Potentially useful later, but too heavy and underspecified for MVP. |

## Production Feature Contract

For MVP, define AIREST-owned outputs rather than adopting OpenWillis dataframe contracts directly:

| Artifact | Required columns / content | Purpose |
| --- | --- | --- |
| `features/facial_features.csv` or equivalent sidecar | `frame_idx`, `timestamp_sec`, face QC fields, optional raw landmarks/head/gaze fields | Realtime capture and QC trace. |
| `features/facial_expressivity_summary.csv` | `overall_mean/std`, `upper_face_mean/std`, `lower_face_mean/std`, `lips_mean/std`, `eyebrows_mean/std`, `mouth_openness_mean/std`, feature version | Offline derived MVP movement summary. |
| `features/facial_qc.json` | face detection percentage, missing spans, camera freeze flags, frame count, FPS, processing status | Session acceptance and export readiness. |
| `features/facial_postprocess_meta.json` | algorithm version, input artifact hashes, parameter values, runtime, success/failure details | Reproducibility and audit trail. |

## Robustness Limitations

| Limitation | Impact | MVP handling |
| --- | --- | --- |
| OpenWillis `frames_per_second` behavior is not a true sampler in local `facial_expressivity` | Output timing may not match user expectation | AIREST should define its own sampling and timestamp policy. |
| First displacement row is `NaN`, not zero | Requires cleanup before summaries or joins | Normalize AIREST output contract explicitly. |
| Missing baseline path silently changes behavior | High risk of mislabeled “baseline-normalized” outputs | Require explicit baseline artifact validation if baseline features are ever enabled. |
| Baseline normalization is methodologically fragile | Relative features can be misleading, especially across raw vs normalized landmark spaces | Exclude from MVP unless fixed and validated. |
| Face loss, occlusion, pose changes, lighting, and multi-face cases can produce missing or unstable landmarks | May bias movement summaries | Keep QC thresholds, missing-span reporting, and manual-review flags. |
| OpenWillis output schema is dataframe-oriented, not an AIREST product contract | Weak backward compatibility for clinical exports | Version AIREST schema independently. |
| Python/runtime constraints are narrow | Operational fragility on clinic laptops | Avoid production dependency; run reference validation in controlled analysis environment. |

## Latency Limitations

| Pipeline | Observed / expected behavior | Decision |
| --- | --- | --- |
| OpenWillis `facial_expressivity` landmark movement | More plausible for offline batch, but still reads/processes full video frame-by-frame | Use as validation reference; AIREST can implement equivalent post-processing. |
| OpenWillis `emotional_expressivity` / py-feat | Local notebook run showed full-frame emotion extraction taking roughly minutes for a short clip; model stack loads RetinaFace, landmarks, AUs, emotions, pose, identity | Research-only/offline; never block realtime recording or QC. |
| Realtime AIREST face QC | Must stay lightweight and deterministic enough for warnings and session gating | Limit realtime work to camera health, face presence, face-centered status, and optional landmark persistence. |

## Implementation Direction

1. Keep AIREST realtime face pipeline minimal and robust: record video, monitor face presence, write QC traces, and never fail a session because optional derived features lag.
2. Add an AIREST-owned offline post-processing step for the MVP movement subset: normalized landmarks, displacement, region summaries, and mouth openness.
3. Use OpenWillis on a validation set to compare summary distributions, missingness, and selected frame-level traces against AIREST outputs.
4. Keep OpenWillis py-feat emotion and AU extraction as a research benchmark until it has stable runtime packaging, acceptable latency, explicit clinical interpretation rules, and validated normalization.
5. Do not expose facial expressivity as a diagnostic or risk-score feature in MVP. Treat all outputs as captured/derived behavioral signals for later analysis.

## Final Decision

OpenWillis should be used as a **validation reference for MVP landmark-movement features** and as a **research-only benchmark for emotion/AU features**. It should **not** be used as a production dependency in the realtime AIREST face pipeline.

## Suggested Feature Layers

| Layer | Include | Blocking realtime? | Rationale |
| --- | --- | --- | --- |
| Realtime minimum | `frame_idx`, `timestamp_sec`, `face_detected`, face confidence/visibility, face-centered status, optional raw MediaPipe landmarks | Yes for QC only | Supports session safety, warnings, and data-quality gating |
| Realtime optional | Raw landmarks, head pose, gaze calibration signals, mouth openness if already cheap | No for clinical completion unless protocol requires it | Useful, but recording must not fail because derived features lag |
| Offline derived face features | Normalized landmarks, displacement, region movement, mouth openness summaries, speaking/not-speaking split, blink event summaries | No | Deterministic from recorded video/landmarks and safer for clinical collection |
| Offline heavy model features | Py-feat emotions and AUs | No | Slow/heavy model stack; better as post-session batch job |
| Multimodal merge | Face summaries + Phonova speech summaries + task events | No | Enables downstream analysis without coupling speech/face extraction to recording |

## Immediate Additions That Do Not Block Realtime Face Capture

| Priority | Addition | Inputs needed | Output |
| --- | --- | --- | --- |
| P0 | Versioned `facial_features.csv` schema | Current MediaPipe process | Stable realtime feature contract |
| P0 | Face QC aggregate | Per-frame face detection | `face_detection_pct`, missing-frame spans, camera-freeze flags |
| P1 | Landmark-derived OpenWillis-compatible summaries | Raw video or stored landmarks | `overall_mean/std`, `lower_face_mean/std`, `upper_face_mean/std`, `lips_mean/std`, `eyebrows_mean/std`, `mouth_openness_mean/std` |
| P1 | Blink summary | Raw video or landmarks | Blink event table, `blinks`, `blink_rate` |
| P2 | Speaking-aware face summaries | Mouth openness plus audio/VAD/transcript timing | Speaking vs non-speaking facial movement summaries |
| P2 | Emotion/AU offline table | Raw video | Framewise 7 emotions, 20 AUs, mean/std summary |
| P3 | Baseline-relative normalization | Protocol-defined neutral baseline clip/segment | Baseline-corrected movement/emotion/AU features |

## Source Notes

- Local OpenWillis review: `facial_expression/facial_expressivity_feature_inventory.md`
- Notebook with outputs: `demo_openwillis_face.ipynb`
- AIREST Confluence hub: <https://pelmeshek.atlassian.net/wiki/spaces/58fb045765d3465da07555dc10f82807/pages/84836354/AIREST+Clinical+Data-Capture+MVP+Documentation+Hub>
- AIREST PRD/SRS/Architecture pages: raw WAV/video recording, MediaPipe facial extraction, required `facial_features.csv`, QC and local/offline MVP boundary
- `phonova` branch: <https://github.com/Pelmeshek1706/phonova/tree/refinement_speech>
