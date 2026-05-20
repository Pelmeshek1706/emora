# AIREST vs OpenWillis Emotional Expressivity Decision Matrix

## Scope

This matrix focuses on OpenWillis `emotional_expressivity` and its fit for AIREST.

It complements:

- `emotional_expressivity.md`
- `emotional_expressivity_feature_inventory.md`
- `Emotional Expressivity Biomarkers for PTSD, Depression, and Anxiety Detection.pdf`
- `facial_expression/airest_openwillis_feature_decision_matrix.md`

Key distinction:

- `facial_expressivity` is a movement pipeline and is a plausible validation reference for AIREST landmark-derived movement features.
- `emotional_expressivity` is a heavy py-feat emotion/AU pipeline and is better treated as an offline research benchmark.

## Decision summary

| Option | Decision | Rationale |
| --- | --- | --- |
| Realtime production dependency | Do not use | Too slow/heavy and too fragile for clinical session gating or capture. |
| MVP facial feature dependency | Do not use | Emotion labels and AUs are not required for a minimum robust face-capture schema. |
| Offline post-processing option | Allow behind a research flag | Can provide useful emotion/AU features after recording completes. |
| Research benchmark | Use | Useful for comparing AIREST-derived facial summaries against a known external implementation. |
| Clinical decision feature | Do not use | Model labels are not disorder diagnoses and baseline behavior is not production-clean. |

## Research evidence driving the decision

The local PDF supports using emotional expressivity as an offline research feature, but not as a stand-alone diagnostic or realtime product dependency.

Key findings:

- depression evidence is strongest for reduced positive expressivity, AU patterns, head movement, and temporal variability rather than a single emotion probability
- PTSD evidence is context-sensitive and strongest in multimodal free-speech or conversation settings where visual markers are joined with audio and language
- anxiety evidence is more tension/arousal-oriented, with anger/fear/neutral differences, head rotation, jawline/eye-region movement, and gaze cues
- AUs are a high-value common denominator across depression and PTSD work
- head pose and gaze are especially important for anxiety-like states and as complementary channels in depression
- temporal dynamics such as entropy, variability, autocorrelation, burstiness, and transitions are missing from the current local output but are central to the PDF's recommendations
- speaking-linked mouth movement is a major confound and should be controlled with audio VAD or active-speaker logic, not only a mouth-motion proxy

Representative study results from the PDF:

| Study | Disorder / setting | Result or takeaway |
| --- | --- | --- |
| Alghowinem et al., 2013 | Depression | Head-pose/movement features reached `71.2%` average recognition. |
| Jiang et al., 2021 | Longitudinal depression | AUC `0.72` for remission and `0.75` for treatment response using emotions, AUs, and temporal statistics. |
| Mahayossanunt et al., 2023 | Depression interviews | Accuracy `91.67%`, F1 `88.89%` from gaze, AU, and expression features. |
| Jin et al., 2025 | E-DAIC depression | Video-only F1 `0.853`, AUC `0.912`; multimodal F1 `0.922`, AUC `0.950`; AU > pose > audio > gaze in contribution analysis. |
| Schultebraucks et al., 2022 | Trauma survivors | PTSD AUC `0.90`, weighted F1 `0.83`; depression AUC `0.86`, weighted F1 `0.82` from visual, acoustic, and semantic markers. |
| Aathreya et al., 2025 | Child PTSD | AU intensities were the best baseline feature family; conversational context mattered. |
| Ren et al., 2025 | GAD | AUC `0.792` for anger, `0.727` for fear, `0.723` for neutral. |
| AnxietyFaceTrack, 2025 preprint | Social anxiety-like state | Multiclass accuracy `0.91`, F1 `0.90`, AUC `0.98`; head rotation and eye/jawline features were important. |

Methodological caution:

- many datasets are small
- labels vary across diagnosis, symptom severity, and state self-report
- temporal units differ widely across studies
- external validation remains rare
- high private-dataset accuracies should be treated as encouraging, not definitive

## Feature decision matrix

| Feature / output | OpenWillis behavior | AIREST fit | Decision | Notes |
| --- | --- | --- | --- | --- |
| Raw video input | Required input artifact | Already required by AIREST | Keep | AIREST should own recording and metadata. |
| Frame/time index | `frame`, `time` returned for sampled frames | Needed for all joins | Include in AIREST schema | Use `frame_idx` and `timestamp_sec`; preserve source-frame coordinates. |
| Face detection | py-feat detection inside sampled frames | Too heavy for realtime QC | Offline only | Realtime QC should stay MediaPipe/camera-health based. |
| 7 emotion scores | `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutral` | Research-useful | Offline research only | Do not treat as true emotion or diagnosis. |
| 20 AU outputs | `AU01`, `AU02`, `AU04`, `AU05`, `AU06`, `AU07`, `AU09`, `AU10`, `AU11`, `AU12`, `AU14`, `AU15`, `AU17`, `AU20`, `AU23`, `AU24`, `AU25`, `AU26`, `AU28`, `AU43` | Research-useful and more granular than labels | Offline research only | Store separately from realtime face QC. |
| Mouth openness | Mean lip-distance from py-feat landmarks | Useful but already derivable from lighter landmarks | Prefer AIREST/MediaPipe version | Use OpenWillis value only with py-feat output family. |
| Speaking probability | GMM over rolling mouth-openness std | Potentially useful | Optional offline | Prefer audio VAD/transcript alignment for production. |
| Head pose and movement | Not returned by current `emotional_expressivity` output | High research value | Add in AIREST offline layer | PDF shows head movement matters for depression and anxiety; this is missing from the current emotional table. |
| Gaze and eye-region behavior | Not returned by current `emotional_expressivity` output | High research value | Add in AIREST offline layer | Needed for anxiety, vigilance/avoidance, and complementary depression features. |
| Temporal dynamics | Only mean/std summaries are returned | High research value | Add windowed dynamics | Add entropy, autocorrelation, burstiness, active-time, onset/offset, and transition statistics. |
| Context labels | Only speaking/not-speaking proxy exists | High research value | Add task/question context | PTSD and trauma work is context-sensitive; summaries should be task-aware. |
| Baseline emotion/AU normalization | Ratio against baseline mean with final shift | Conceptually useful, implementation fragile | Exclude unless fixed | Requires explicit neutral baseline protocol and code cleanup. |
| Summary table | One row, mean/std over features | Useful offline aggregate | Include only with metadata | Must record baseline mode and sampling parameters. |
| Detection/missingness QC | Not explicitly returned | Required for clinical reliability | AIREST must add | Track expected vs successful sampled frames. |
| Model versions | Not returned | Required for reproducibility | AIREST must add | Persist py-feat and model component versions. |

## Recommended AIREST feature layers

| Layer | Include emotional expressivity? | Blocking realtime? | Rationale |
| --- | --- | --- | --- |
| Realtime capture | No | Yes, but only for recording/QC | Keep realtime loop simple: video, audio, face present, frame timestamps. |
| Realtime face QC | No | Yes | py-feat is too heavy for live warnings. |
| Offline MVP movement features | No | No | Use MediaPipe landmarks for movement, blink, mouth, head/gaze as applicable. |
| Offline research feature extraction | Yes, optional | No | Run after the session when runtime cost is acceptable. |
| Clinical reporting | Not as diagnostic content | No | Summaries can be stored as behavioral/model features, not interpreted as diagnoses. |

## Proposed AIREST artifacts

| Artifact | Required? | Contents |
| --- | --- | --- |
| `features/emotional_expressivity_framewise.csv` | Optional research | Sampled frame/time rows, 7 emotions, 20 AUs, `mouth_openness`, optional `speaking_probability`, mode flags. |
| `features/emotional_expressivity_summary.csv` | Optional research | Mean/std summaries for emotions, AUs, and mouth openness; optional speaking/not-speaking splits. |
| `features/emotional_expressivity_qc.json` | Required if feature is run | Source frame count, expected sampled frames, successful rows, failed rows, baseline used, warnings. |
| `features/emotional_expressivity_meta.json` | Required if feature is run | Package versions, model names, parameters, input hashes, baseline hashes, runtime, status. |

## Research-driven roadmap

The PDF recommends treating emotional expressivity as a hierarchy:

| Layer | Feature examples | AIREST decision |
| --- | --- | --- |
| Framewise state | Emotions, AUs, mouth openness, gaze, head pose, blink | Optional offline research extraction. |
| Behavioral episodes | Smile episodes, brow-tension episodes, gaze aversion, blink bursts, lip pressing | Add after stable framewise extraction. |
| Temporal dynamics | Windowed mean/std, coefficient of variation, entropy, autocorrelation, burstiness, transition counts | Add before any modeling claims. |
| Context | Speaking/listening, question type, trauma narrative, neutral baseline, prompt valence | Required for PTSD/anxiety interpretation. |
| Multimodal fusion | Face plus audio VAD/prosody plus ASR/text features | Research tier only, after feature QC. |

Recommended priority list:

| Priority | Addition | Reason |
| --- | --- | --- |
| P0 | QC metadata and missingness accounting | Required to distinguish low expressivity from failed tracking. |
| P1 | AU-centric summaries and composites | Strongest common facial feature family across the PDF. |
| P1 | Temporal dynamics over `2-10` second windows | Mean/std alone is narrower than the biomarker literature. |
| P1 | Audio VAD-backed speaking split | Mouth-only speaking probability confounds speech and affective mouth movement. |
| P2 | Head pose and gaze summaries | High value for anxiety and complementary value for depression/PTSD. |
| P2 | Context-specific summaries | PTSD and trauma findings depend heavily on conversational context. |
| P3 | Multimodal models | Best reported PTSD/depression results combine visual, acoustic, and semantic markers. |

Suggested evaluation standard if AIREST trains models on these features:

- subject-exclusive splits
- nested cross-validation
- AUC, F1, balanced accuracy, sensitivity, specificity
- Brier score or ECE for calibration
- bootstrapped confidence intervals
- subgroup performance by sex, age, and site
- explicit target definition: diagnosis, severity score, or state self-report

## Baseline policy

Do not enable baseline-corrected emotion/AU output unless the protocol defines:

- what the baseline clip is
- when it is recorded
- how long it must be
- minimum face-detection quality
- allowed lighting/camera changes
- whether speech is allowed during baseline
- how missing baseline files are handled

Implementation requirements before enabling baseline:

1. Fail loudly or mark `baseline_used=false` when the baseline path is missing.
2. Compute `summary` from the same transformed values returned in `framewise`.
3. Do not shift `mouth_openness` by the emotion/AU baseline transform.
4. Add an epsilon or a more stable transform for near-zero baseline means.
5. Include baseline metadata in every output artifact.

## Latency and runtime risk

| Risk | Impact | Handling |
| --- | --- | --- |
| Heavy py-feat model stack | Slow extraction on short videos | Run only after recording. |
| Torch/py-feat dependency constraints | Harder deployment on clinic laptops | Keep out of MVP runtime. |
| XGBoost/OpenMP hangs | Possible process stalls | Keep single-thread workaround and isolate as offline worker. |
| Detector failures | Dropped rows without explicit public QC | Add QC metadata wrapper. |
| Multi-face/occlusion/pose sensitivity | Misleading labels or missing rows | Persist face QC and review flags. |

## Clinical interpretation policy

Allowed language:

- "model-derived facial emotion scores"
- "py-feat action unit outputs"
- "offline behavioral features"
- "relative-to-baseline model scores" when baseline mode is explicitly used

Avoid:

- "detected true emotion"
- "diagnosed depression/PTSD/anxiety"
- "clinical risk score"
- "baseline-normalized" when the baseline file was missing
- comparing baselined and raw outputs without a clear flag

## Recommended implementation direction

1. Keep AIREST realtime capture independent of OpenWillis emotional expressivity.
2. Add AIREST-owned metadata and QC around any offline py-feat run.
3. Start with no-baseline emotion/AU outputs for readability and validation.
4. Defer baseline mode until the normalization issues are fixed.
5. Join emotion/AU summaries with speech/task metadata offline, not during recording.
6. Treat emotion labels as model features and AUs as the more granular analysis layer.

## Final decision

Use OpenWillis `emotional_expressivity` as an **offline, research-only benchmark and optional post-processing feature**.

Do not use it as:

- a realtime dependency
- an MVP production feature
- a clinical decision feature
- a substitute for AIREST-owned face QC

## Source notes

- `emotional_expressivity/emotional_expressivity.md`
- `emotional_expressivity/emotional_expressivity_feature_inventory.md`
- `emotional_expressivity/Emotional Expressivity Biomarkers for PTSD, Depression, and Anxiety Detection.pdf`
- `facial_expression/facial_expressivity_feature_inventory.md`
- `facial_expression/airest_openwillis_feature_decision_matrix.md`
- `openwillis-face/src/openwillis/face/facial_emotion.py`
- `openwillis-face/src/openwillis/face/util/speaking_utils.py`
- `demo_openwillis_face.ipynb`
