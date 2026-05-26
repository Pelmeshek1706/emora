# Emotional Expressivity (`emotional_expressivity`)

Upstream documentation:

- https://openwillis.brooklyn.health/Emotional-expressivity-v2-1-15883a8fe04780f987a8cf2bcbd60b8e

Local implementation reviewed here:

- `openwillis-face/src/openwillis/face/facial_emotion.py`
- `openwillis-face/src/openwillis/face/util/speaking_utils.py`
- `openwillis-face/src/openwillis/face/config/facial.json`
- `demo_openwillis_face.ipynb`
- `README.md`
- `emotional_expressivity/Emotional Expressivity Biomarkers for PTSD, Depression, and Anxiety Detection.pdf`

Companion notes in this folder:

- `emotional_expressivity_feature_inventory.md`
- `airest_openwillis_emotional_expressivity_decision_matrix.md`
- `rafdb_landmark_graph_transformer_task.md`
- `model_training_report.md`
- `model_report_review.md`

This note follows the style and depth of the `facial_expression/` documentation. It combines:

- the upstream OpenWillis method description
- the actual local code path
- the local demo notebook behavior
- output schema details
- known implementation caveats
- AIREST-specific product guidance

## What this feature is for

`emotional_expressivity` is the label-based face-analysis signal in OpenWillis.

It tries to answer:

- which emotion-like categories the py-feat model assigns to each sampled face frame
- which facial action units are active
- how mouth opening changes over time
- whether the output should be summarized during speaking vs not speaking
- how a task clip differs from an optional baseline clip

It is different from `facial_expressivity`.

- `facial_expressivity` measures facial movement magnitude from MediaPipe landmarks.
- `emotional_expressivity` runs a heavier py-feat model stack and emits semantic labels and AUs.

Use this function when you need interpretable emotion-category or AU traces. Use `facial_expressivity` when you only need movement intensity and want the more stable local pipeline.

## What this feature is not

This output is not a diagnostic result.

It should not be interpreted as:

- a clinical diagnosis
- a direct measure of true internal emotion
- a disorder-specific risk score
- a cross-person normalized affect score
- a realtime session-gating signal

The labels are model outputs from video frames. They can be useful as research features, exploratory summaries, and offline benchmarks, but they need careful interpretation.

## Current RAF-DB landmark model update

The production `emotional_expressivity` implementation is still the OpenWillis/py-feat framewise pipeline described below. The new notebook work adds a separate landmark-only research branch for the same 7 basic emotion labels.

Main artifact location after the folder move:

- `output/jupyter-notebook/emotional_expressivity/`

Primary local report:

- `emotional_expressivity/model_training_report.md`

### What changed

The new experiments evaluate whether a lightweight model that consumes only MediaPipe facial landmarks can complement or eventually replace the py-feat emotion head for the 7-category emotion task.

This branch does not replace the full py-feat stack yet:

- py-feat still provides the current OpenWillis emotion, AU, and mouth-openness outputs.
- Graphormer-lite currently predicts only 7 emotion probabilities.
- Graphormer-lite does not produce py-feat AUs, head pose, gaze, or speaking split features.
- The current results are RAF-DB static-image results, not clinical-video results.

### Dataset used

Dataset: `Pelmeshek/raf-db-7emotions-mediapipe-768`.

This is a RAF-DB-derived landmark cache with these target labels:

`anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutral`.

The model was trained on MediaPipe landmark rows, not raw pixels:

- `478` MediaPipe landmarks per face.
- `(x, y, z)` per landmark.
- stable eye-normalized coordinate system.
- additional engineered geometry features.

Dataset size before and after MediaPipe filtering:

| stage | anger | disgust | fear | happiness | sadness | surprise | neutral | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| initial RAF-DB cache | 4071 | 877 | 355 | 5957 | 2460 | 1619 | 5132 | 20471 |
| after MediaPipe success filter | 3301 | 756 | 289 | 5078 | 2074 | 1449 | 5131 | 18078 |

Coverage after filtering is `88.31%`. All failures in the current cache are `no_face_detected`.

MediaPipe preprocessing used default Face Landmarker confidence thresholds, not a custom high threshold:

| option | value |
| --- | ---: |
| `min_face_detection_confidence` | 0.5 |
| `min_face_presence_confidence` | 0.5 |
| `min_tracking_confidence` | 0.5 |
| `num_faces` | 1 |

Important bias: filtering is class-dependent. `neutral` almost always survives, while `anger`, `fear`, `happiness`, and `sadness` lose materially more samples. This changes class priors and makes macro-F1, balanced accuracy, and per-class metrics more important than accuracy alone.

Dataset artifacts:

- `output/jupyter-notebook/emotional_expressivity/rafdb_mediapipe_768_publish/cache/processed_rows_full.jsonl`
- `output/jupyter-notebook/emotional_expressivity/model_training_report_assets/rafdb_trainable_distribution.png`
- `output/jupyter-notebook/emotional_expressivity/rafdb_mediapipe_768_publish/failed_mediapipe_contact_sheet.jpg`

### Best current landmark model

Best current model: `gformer_m_ce_sqrtw_geom_seed42`.

Held-out RAF-DB test result:

| model | params | best epoch | test accuracy | test macro-F1 | test weighted-F1 | test balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gformer_m_ce_sqrtw_geom_seed42` | 2,900,030 | 73 | 0.7397 | 0.5817 | 0.7400 | 0.5832 |
| `gformer_s_ce_sqrtw_geom_seed42` | 1,489,978 | 42 | 0.7165 | 0.5501 | 0.7143 | 0.5458 |
| `gformer_m_focal_sqrtw_geom_seed42` | 2,974,142 | 36 | 0.6665 | 0.5178 | 0.6746 | 0.5430 |

The useful scaling result is from `gformer_s` to `gformer_m`: macro-F1 improves from `0.5501` to `0.5817`. The focal-loss medium run did not help in the current setup.

Training artifacts:

- `output/jupyter-notebook/emotional_expressivity/rafdb_graphormer_lite_full_colab_uv_mediapipe_legacy_exp.ipynb`
- `output/jupyter-notebook/emotional_expressivity/rafdb_graphormer_lite_full_run_exp/`
- `output/jupyter-notebook/emotional_expressivity/rafdb_graphormer_lite_full_run_exp/experiments/20260525_140301_gformer_m_ce_sqrtw_geom_seed42/summary.json`

### Comparison with py-feat

The py-feat comparison notebook evaluates both models on the same RAF-DB image sample:

```python
image_for_test, facial_landmarks_for_test = pick_image_frpm_dataset(emotion="anger")
result_gformer = inferance_gformer(facial_landmarks_for_test)
result_pyfeat = inferance_pyfeat(image_for_test)
```

Comparison artifact size: `315` images, `45` per class.

Top-label match against RAF-DB labels:

| comparison | agreement rate | n |
| --- | ---: | ---: |
| `pyfeat_top_vs_dataset_label` | 0.4476 | 315 |
| `gformer_m_top_vs_dataset_label` | 0.6127 | 315 |
| `gformer_m_top_vs_pyfeat_top` | 0.3778 | 315 |

Interpretation:

- `gformer_m` is more aligned with RAF-DB labels on this sample.
- py-feat is not an oracle; it is a different image-based model with a different training source and preprocessing stack.
- Low agreement between py-feat and Graphormer-lite mostly shows representation and calibration mismatch, not that one model is universally correct.

py-feat comparison artifacts:

- `output/jupyter-notebook/emotional_expressivity/mediapipe_pyfeat_test.ipynb`
- `output/jupyter-notebook/emotional_expressivity/mediapipe_pyfeat_test/results/`
- `output/jupyter-notebook/emotional_expressivity/model_training_report_assets/pyfeat_vs_gformer_m_class_match.png`

### Why the MLP baseline is still important

The old `flattened_mlp` baseline remains close to `gformer_s`:

| model | params | best epoch | test accuracy | test macro-F1 | test balanced accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| `flattened_mlp` | 802,567 | 53 | 0.7140 | 0.5470 | 0.5667 |
| `gformer_s_ce_sqrtw_geom` | 1,489,978 | 42 | 0.7165 | 0.5501 | 0.5458 |

This is plausible because MediaPipe has already converted pixels into semantically ordered facial landmarks. A dense MLP over fixed landmark indices can immediately mix all face regions globally. Graphormer-lite adds useful graph bias, but the small version does not have enough advantage to clearly beat a well-regularized dense baseline. The medium model is the first variant where the graph-aware design gives a meaningful gain.

Architecture comparison artifacts:

- `output/jupyter-notebook/emotional_expressivity/architecture_comparison/flattened_mlp_vs_gformer_s_architecture.png`
- `output/jupyter-notebook/emotional_expressivity/architecture_comparison/flattened_mlp_vs_gformer_s_per_class_f1.png`

### Footprint and latency

Local CPU benchmark from `mediapipe_pyfeat_test.ipynb`:

| model | pipeline | params | disk footprint MB | param/buffer memory MB | median latency ms/sample |
| --- | --- | ---: | ---: | ---: | ---: |
| `gformer_s` | precomputed landmarks -> logits | 1,489,978 | 10.955 | 10.928 | 20.983 |
| `gformer_m` | precomputed landmarks -> logits | 2,900,030 | 16.337 | 16.307 | 37.128 |
| `py-feat_default_stack` | raw image -> face/landmark/emotion stack | n/a | 946.464 | 800.026 | 128.901 |

This is not a pure model-only latency comparison. Graphormer-lite assumes MediaPipe landmarks already exist. py-feat includes raw-image face detection, landmark detection, and emotion inference. In a pipeline that already computes MediaPipe landmarks for other features, Graphormer-lite is much cheaper at the emotion-classification step.

Benchmark artifact:

- `output/jupyter-notebook/emotional_expressivity/mediapipe_pyfeat_test/results/model_footprint_latency.csv`

### Practical AIREST conclusion

Use `gformer_m_ce_sqrtw_geom_seed42` as the current best RAF-DB landmark-only emotion baseline. Keep `gformer_s` as the smaller deployment candidate and keep `flattened_mlp` as a required control baseline.

For the current OpenWillis-style `emotional_expressivity` feature, do not treat Graphormer-lite as a full replacement yet:

- it can replace or complement only the 7-emotion classifier branch;
- it does not replace AU extraction;
- it does not solve temporal emotional expressivity;
- it needs calibration checks before probability values are used as stable biomarkers;
- clinical usage still requires subject-exclusive clinical datasets and temporal/video-level evaluation.

## Upstream docs vs local code

The upstream docs describe:

- 7 emotion outputs:
  - happiness
  - sadness
  - anger
  - fear
  - disgust
  - surprise
  - neutral
- raw emotion scores on a `0-100` scale
- action unit values on a `0-1` style scale
- optional baseline correction
- optional speaking split
- a frame-sampling control
- a summary table

The local implementation broadly follows that intent, but several details matter:

1. The local public API uses `skip_frames`, not `frames_per_second`.
2. The function creates one internal row per source frame, fills skipped or failed frames with `NaN`, then drops null rows before returning.
3. The final returned framewise table contains only sampled frames that survived detection.
4. Baseline mode changes emotion and AU values from raw model scores into relative-to-baseline transformed values.
5. In baseline mode, `summary` and returned `framewise` are not on the exact same transformed scale.
6. The docstring mentions an overall expressivity column, but the code does not compute one.

## Public API

```python
import openwillis.face as owf

framewise, summary = owf.emotional_expressivity(
    filepath="video.mov",
    baseline_filepath="sample_data/baseline.mp4",
    bbox_list=[],
    base_bbox_list=[],
    skip_frames=5,
    split_by_speaking=False,
    rolling_std_seconds=3,
)
```

Arguments:

| Argument | Type | Default | Local behavior |
| --- | --- | --- | --- |
| `filepath` | `str` | required | Main video path. |
| `baseline_filepath` | `str` | `""` | Optional baseline video path. Baseline correction happens only if this path exists on disk. |
| `bbox_list` | `list[dict]` | `[]` | Optional per-frame face bounding boxes for the main video. If provided, length must match source frame count. |
| `base_bbox_list` | `list[dict]` | `[]` | Optional per-frame face bounding boxes for the baseline video. |
| `skip_frames` | `int` | `5` | Analyze one frame, skip the next `skip_frames` frames, then repeat. Use nonnegative values. |
| `split_by_speaking` | `bool` | `False` | If true, adds speaking probability and creates speaking/not-speaking summary columns. |
| `rolling_std_seconds` | `int` | `3` | Window size used by the mouth-openness speaking proxy. |

Returns:

| Return value | Type | Meaning |
| --- | --- | --- |
| `framewise` | `pandas.DataFrame` | Per-sampled-frame emotion, AU, and mouth-openness features after `dropna()`. |
| `summary` | `pandas.DataFrame` | One-row mean/std summary over measured feature columns, optionally split by speaking state. |

## Demo notebook call

Notebook file:

- `demo_openwillis_face.ipynb`

Notebook pattern:

```python
import openwillis.face as owf

framewise, summary = owf.emotional_expressivity(
    filepath="video.mov",
    baseline_filepath="/Users/pelmeshek1706/Downloads/baseline (1).mp4",
    bbox_list=[],
    base_bbox_list=[],
    skip_frames=5,
    split_by_speaking=False,
    rolling_std_seconds=3,
)
```

Important workspace note:

- the notebook baseline path points outside this repo
- the path may not exist on another machine
- the local code silently skips baseline correction when the path does not exist
- for validated local documentation, use `sample_data/baseline.mp4`

## Demo result in this workspace

Validated input setup:

- main video: `video.mov`
- baseline video: `sample_data/baseline.mp4`
- main video frame count: `644`
- main video fps: `30`
- `skip_frames=5`
- `split_by_speaking=False`

Returned objects:

- `framewise.shape == (108, 30)`
- `summary.shape == (1, 56)`

Why `108` rows:

- frame `0` is analyzed
- then five frames are skipped
- then frame `6` is analyzed
- this repeats through frame `642`
- `644` source frames at a stride of `6` produce `108` sampled frames

Observed framewise columns in the baselined run:

- `frame`
- `time`
- `mouth_openness`
- `anger`
- `disgust`
- `fear`
- `happiness`
- `sadness`
- `surprise`
- `neutral`
- `AU01`
- `AU02`
- `AU04`
- `AU05`
- `AU06`
- `AU07`
- `AU09`
- `AU10`
- `AU11`
- `AU12`
- `AU14`
- `AU15`
- `AU17`
- `AU20`
- `AU23`
- `AU24`
- `AU25`
- `AU26`
- `AU28`
- `AU43`

Observed summary shape:

- 28 measured features
- 2 statistics per feature
- 56 total summary columns

## Feature inventory

### 1. Frame metadata

Every returned framewise row includes:

- `frame`
- `time`

`frame` is the original source-video frame number, not a compact sampled-frame index.

`time` is computed as:

```text
frame / source_video_fps
```

That means returned rows keep the source-video temporal coordinate even after skipped rows are dropped.

### 2. Emotion outputs

The local runtime emits these 7 emotion columns:

- `anger`
- `disgust`
- `fear`
- `happiness`
- `sadness`
- `surprise`
- `neutral`

Source of names:

- `feat.utils.FEAT_EMOTION_COLUMNS`

How they are produced:

1. `feat.Detector().detect_faces(...)` detects faces.
2. `detector.detect_landmarks(...)` estimates landmarks.
3. `detector.detect_emotions(...)` emits model emotion scores.
4. Local code multiplies the py-feat emotion scores by `100`.

Without baseline:

- these are py-feat model scores on a `0-100` style scale
- they are easiest to inspect and plot in this mode

With baseline:

- these become relative-to-baseline values
- they should not be called raw probabilities or percentages
- values can exceed ordinary `0-100` expectations after normalization artifacts

### 3. Action unit outputs

The local runtime emits these 20 AU columns:

- `AU01`
- `AU02`
- `AU04`
- `AU05`
- `AU06`
- `AU07`
- `AU09`
- `AU10`
- `AU11`
- `AU12`
- `AU14`
- `AU15`
- `AU17`
- `AU20`
- `AU23`
- `AU24`
- `AU25`
- `AU26`
- `AU28`
- `AU43`

Source of names:

- `feat.pretrained.AU_LANDMARK_MAP["Feat"]`

How they are produced:

1. py-feat landmarks are estimated for the detected face.
2. `detector.detect_aus(frame, landmarks)` emits AU values.
3. The values are appended after the emotion columns.

Without baseline:

- the upstream docs describe AU outputs as `0-1` style activations

With baseline:

- the returned AU columns are relative-to-baseline transformed values
- they are no longer direct raw AU activations

### 4. `mouth_openness`

The function adds:

- `mouth_openness`

How it is computed in this module:

- upper lip landmark indices: `[61, 62, 63]`
- lower lip landmark indices: `[65, 66, 67]`
- for each pair, compute 2D Euclidean distance
- return the mean of those three distances

This differs from the `facial_expressivity` mouth-openness ratio. In `emotional_expressivity`, the feature is a simple average lip-distance measurement from py-feat landmarks.

Uses:

- standalone mouth-activity feature
- speaking-probability proxy input

Baseline caveat:

- `mouth_openness` is excluded from the baseline ratio step
- but it is still shifted down by `1` in the final returned framewise table when a baseline file exists
- the summary is computed before that final shift

### 5. Optional `speaking_probability`

If `split_by_speaking=True`, `framewise` also gets:

- `speaking_probability`

The speaking proxy is computed by `get_speaking_probabilities()`:

1. infer fps from differences in the internal `time` column before final `dropna()`
2. compute rolling standard deviation of `mouth_openness`
3. fit a 2-component Gaussian mixture model
4. treat the higher-variance component as speaking-like
5. return probability of belonging to that component

Important caveats:

- this is not audio VAD
- it is a mouth-motion proxy
- because skipped frames still have metadata rows internally, the inferred fps usually reflects source-video fps rather than the final returned sampled-row rate
- it can confuse non-speech mouth movement with speaking
- it depends on enough sampled frames to fit a 2-component model
- after `dropna()`, early rolling-window rows can be removed

## Output schemas

### No-baseline framewise schema

When no baseline file exists, the natural output order is:

1. `frame`
2. `time`
3. 7 emotion columns
4. 20 AU columns
5. `mouth_openness`
6. optional `speaking_probability`

### Baselined framewise schema

When the baseline path exists, the baseline helper reconstructs the dataframe with this order:

1. `frame`
2. `time`
3. `mouth_openness`
4. 7 emotion columns
5. 20 AU columns
6. optional `speaking_probability`

Do not rely on column position alone. Prefer column names.

### Summary schema without speaking split

The summary is a one-row dataframe.

It contains mean and standard-deviation columns for every feature after `frame` and `time`:

- `mouth_openness_mean`
- 7 emotion `_mean` columns
- 20 AU `_mean` columns
- `mouth_openness_std`
- 7 emotion `_std` columns
- 20 AU `_std` columns

Total:

- 28 feature columns x 2 statistics = 56 columns

### Summary schema with speaking split

If `split_by_speaking=True`, summary columns are duplicated as:

- `[feature]_[stat]_speaking`
- `[feature]_[stat]_not_speaking`

Examples:

- `happiness_mean_speaking`
- `happiness_mean_not_speaking`
- `AU12_std_speaking`
- `AU12_std_not_speaking`
- `mouth_openness_mean_speaking`
- `mouth_openness_mean_not_speaking`

The summary does not include a summary of `speaking_probability` itself because the split helper drops the split column before computing statistics.

## Algorithm walkthrough

### Detector setup

The module sets:

```python
os.environ.setdefault("OMP_NUM_THREADS", "1")
```

before importing `feat`. This is a local workaround for intermittent macOS hangs in py-feat's XGBoost AU detector.

The code also uses a shared module-level detector:

```python
_DETECTOR = None

def _get_detector():
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = feat.Detector()
    return _DETECTOR
```

This avoids repeatedly constructing `feat.Detector()` in one Python process.

### Per-video pass

`emotional_expressivity()` calls:

```text
get_emotion() -> run_pyfeat()
```

`run_pyfeat()`:

1. opens the video with `cv2.VideoCapture`
2. reads source frame count and fps
3. builds emotion/AU column names from py-feat
4. initializes or reuses the shared detector
5. walks the video frame by frame
6. analyzes sampled frames
7. creates `NaN` rows for skipped or failed frames
8. returns a list of one-row dataframes

`get_emotion()` concatenates that list into a single dataframe.

### Sampling behavior

For valid nonnegative `skip_frames` values, the effective stride is:

```text
stride = skip_frames + 1
```

Examples:

| `skip_frames` | Behavior |
| ---: | --- |
| `0` | Analyze every frame. |
| `1` | Analyze every other frame. |
| `5` | Analyze frame 0, skip 1-5, analyze 6, and repeat. |
| `29` | Approximate one frame per second for a 30 fps source video. |

This is a frame-stride sampler, not a target-fps sampler. If the source video fps changes, the wall-clock sampling interval changes too.

### Per-sampled-frame processing

For each sampled frame without an external bbox:

```text
detect_emotions()
  -> detector.detect_faces(frame, threshold=0.95)
  -> detector.detect_landmarks(frame, detected_faces=faces)
  -> detector.detect_aus(frame, landmarks)
  -> detector.detect_emotions(frame, faces, landmarks)
  -> mouth_openness(landmarks)
```

Emotion scores are multiplied by `100`.

The emotion and AU arrays are horizontally stacked into one row, and `mouth_openness` is added as a final scalar.

### External bounding boxes

If `bbox_list` is passed:

- it must contain one bbox dictionary per original video frame
- expected keys are `bb_x`, `bb_y`, `bb_w`, and `bb_h`
- if `bb_x` is `NaN`, the frame is treated as undetected
- otherwise the frame is cropped before py-feat detection

Important bug:

- `bb_dict_to_bb_list()` returns `bb_y + bb_y` for the bottom coordinate
- the intended value is almost certainly `bb_y + bb_h`
- this helper is not used in the default notebook call, which passes empty bbox lists

### Baseline behavior

Baseline correction happens only when:

```python
os.path.exists(baseline_filepath)
```

If the baseline file does not exist, no warning is raised and the function silently returns no-baseline output.

When the baseline exists:

1. copy the main dataframe
2. preserve `frame`, `time`, and `mouth_openness`
3. remove `frame`, `time`, and `mouth_openness` from the normalization columns
4. run `get_emotion()` on the baseline video
5. compute baseline means for emotion and AU columns
6. add `1` to the baseline means
7. add `1` to the main emotion/AU values
8. divide main values by baseline means
9. reattach `frame`, `time`, and `mouth_openness`
10. compute summary
11. subtract `1` from the whole returned framewise dataframe
12. add `1` back only to `frame`, `time`, and optionally `speaking_probability`
13. drop rows containing `NaN`
14. reset the index

For emotion and AU columns, the intended final formula is:

```text
((main_value + 1) / (baseline_mean + 1)) - 1
```

Interpretation:

- `0` means approximately equal to baseline
- positive values mean above baseline
- negative values mean below baseline

However, because `summary` is computed before the final subtract-`1` step, the summary and returned framewise values are offset from each other in baseline mode.

## Null and row-retention behavior

The internal table has one row per original source frame.

Rows can contain `NaN` because:

- the frame was intentionally skipped by `skip_frames`
- face detection failed on a sampled frame
- the bbox was invalid
- the frame could not be decoded
- speaking probability could not be computed for early rolling-window rows

The public function ends with:

```python
df_norm_emo.dropna(inplace=True)
df_norm_emo.reset_index(drop=True, inplace=True)
```

So the returned `framewise` table contains only complete rows.

For the validated `644`-frame demo with `skip_frames=5`:

- internal raw rows: `644`
- sampled rows: `108`
- expected skipped rows: `536`
- final returned rows: `108`
- final returned null count: `0` in the validated run

If the final returned row count is smaller than expected, likely causes are:

- py-feat face detection failed on some sampled frames
- the face was occluded or off-camera
- the crop was wrong
- the detector threw an exception and the row was replaced with `NaN`
- `speaking_probability` introduced additional null rows

## Interpreting outputs

### Raw no-baseline mode

Raw mode is best when the goal is:

- plotting model emotion traces
- reviewing expression-rich moments
- inspecting AU activations
- generating interpretable exploratory features
- benchmarking against another emotion/AU extractor

In this mode:

- emotion columns are model scores on a `0-100` style scale
- AU columns are py-feat AU outputs
- `mouth_openness` is the mean lip-distance feature

### Baseline mode

Baseline mode is best when the goal is:

- within-subject comparison
- current task vs neutral/rest clip
- current task vs calibration clip
- relative expression change

In this mode:

- emotion and AU values are relative to baseline
- `0` is the conceptual no-change point after final framewise shift
- raw probability interpretation no longer applies
- near-zero baseline means can create unstable ratios

Baseline mode should only be used when:

- the baseline clip exists
- the baseline clip is protocol-defined
- the baseline clip is similar in lighting, camera, pose, and duration
- output consumers know the values are transformed

### Speaking split mode

Speaking split mode is useful when:

- expression is expected to differ during speech
- the clip contains clear speech and non-speech segments
- downstream analysis needs separate summaries for mouth-active intervals

It is weaker when:

- the clip has little silence
- mouth movement is unrelated to speech
- frame sampling is too sparse
- the recording is too short for a stable rolling-window estimate

## What the demo suggests

The validated baselined demo returned:

- high `surprise_mean` relative to other emotion summary columns
- high mouth-related AU means, especially around `AU25`, `AU20`, `AU26`, and `AU10`
- strong mouth activity in sampled frames

Practical read:

- the clip appears expression-rich and mouth-active
- the output is analytically useful for exploration
- the absolute baselined magnitudes should be treated cautiously

What not to over-interpret:

- exact baselined emotion magnitudes
- exact baselined AU magnitudes
- emotion labels as true affective state
- any disorder inference

## Research context from the PDF

Source:

- `Emotional Expressivity Biomarkers for PTSD, Depression, and Anxiety Detection.pdf`

The PDF's central conclusion is that emotional expressivity features are promising behavioral biomarkers for psychiatric screening and monitoring, but the evidence is stronger for adjunctive assessment than for stand-alone diagnosis.

The strongest cross-study signal is not a generic "emotion score." The more reproducible signals are:

- facial action units
- head-motion dynamics
- gaze and eye-region behavior
- temporal variability over time
- context-aware summaries
- multimodal fusion with speech, language, and sometimes physiology

That matters for this repo because the current `emotional_expressivity` path has the right starting point, but it is still narrow. It returns py-feat emotions, AUs, and `mouth_openness`, then reduces them mostly to mean/std summaries. The PDF argues for moving from a framewise summary toolbox toward a windowed, sequenced, multimodal biomarker pipeline.

### Disorder-level findings

| Disorder / state | Main findings from the PDF | How it maps to this repo |
| --- | --- | --- |
| Depression | Most consistent signals involve reduced positive expressivity, altered brow and mouth activity, reduced movement variability, and lower facial flexibility. | Current emotion/AU output is relevant, but the repo should add temporal dynamics, smile/mouth composites, brow features, and head-pose summaries. |
| PTSD | Evidence is more context-sensitive and multimodal. Visual arousal markers, facial movement parameters, speech prosody, and language cues jointly outperform single channels. Recent child-PTSD work emphasizes de-identified AU, landmark, gaze, and head-pose sequences. | Current py-feat output should be joined with task context, speech/audio features, and sequence models. AUs are useful, but not enough alone. |
| Anxiety / anxious states | Signals are less about blunting and more about tension, arousal, vigilance, anger/fear/neutral expression differences, head rotation, jawline/eye-region motion, and gaze cues. | Current emotion labels capture some anger/fear/neutral signal, but head pose, gaze, and eye-region features are missing from the emotional summary path. |

### Study evidence summarized in the PDF

| Study | Population / setting | Feature families | Results or main finding |
| --- | --- | --- | --- |
| Alghowinem et al., 2013 | Clinically validated depression videos; exact N not recovered in the inspected source | Head pose and movement | Head-pose/movement features reached `71.2%` average recognition with SVM, showing nonverbal kinematics can carry depression signal. |
| Harati et al., 2020 | 12 severely depressed patients in repeated DBS recovery interviews | Dynamic facial variability, multiscale entropy | Higher facial expressivity/variability tracked lower depression severity; useful because dynamics, not only average intensity, were treated as biomarkers. |
| Jiang et al., 2021 | 365 video interviews, 88 hours, 12 depressed patients | 7 emotions, AUs, temporal statistics | AUC `0.72` for remission classification and `0.75` for treatment response under leave-one-subject-out validation. |
| Mahayossanunt et al., 2023 | 474 clinical interview videos; 134 depressed and 340 non-depressed | Gaze angles, AU intensity, expression features | LSTM with attention-based fusion reported accuracy `91.67%`, F1 `88.89%`, precision `91.40%`, recall `87.03%`. |
| Kim et al., 2023 | 59 older adults | 17 AUs across posed and spontaneous emotion tasks | Higher depressive symptoms were linked to mouth-corner downward/inward pull in posed expressions and raised/narrowed inner brows in spontaneous expressions. |
| Jin et al., 2025 | E-DAIC, 219 participants | OpenFace facial features, gaze, head pose, AUs, audio MFCC | Video-only F1 `0.853`, AUC `0.912`; multimodal fused model F1 `0.922`, AUC `0.950`, MAE `3.51`; feature contribution ranked AU > pose > audio > gaze. |
| Schultebraucks et al., 2022 | 81 trauma survivors, one month after ED visit | Facial emotion/intensity, movement, speech prosody, language | PTSD AUC `0.90`, weighted F1 `0.83`; depression AUC `0.86`, weighted F1 `0.82`, supporting multimodal post-trauma markers. |
| Aathreya et al., 2025 | 18 children, seven conversational PTSD contexts | AU intensities, landmarks, eye gaze, head pose | AU intensities were the best baseline feature family and context mattered; full metric table was not exposed in the inspected abstract. |
| Fujiwara et al., 2015 | 23 preschool children exposed to the Great East Japan Earthquake | Human-rated facial expression response to comedy clip | Supports the idea that altered emotional reactivity is measurable after trauma exposure, even without automated CV features. |
| Ren et al., 2025 | 60 GAD patients and 60 matched controls | Seven expression categories, especially neutral, anger, fear | Expression-symptom correlations ranged from about `-0.35` to `0.34`; AUC was `0.792` for anger, `0.727` for fear, and `0.723` for neutral. |
| Zhou et al., 2023 | 319 older adults with mild cognitive impairment | Speech, facial expression, and text features | Accessible sources indicated multiclass accuracies above `85%` for depression/anxiety/apathy presentations; population-specific to MCI. |
| AnxietyFaceTrack, 2025 preprint | 91 students, 1,173 ten-second smartphone samples | 669 OpenFace-derived landmark, gaze, pose, and AU features | Random Forest reached multiclass accuracy `0.91`, F1 `0.90`, AUC `0.98`; head rotation, jawline/face-edge, and eye landmarks were important. Not a clinical anxiety-disorder study. |

### Biomarker families emphasized by the PDF

| Feature family | Depression | PTSD | Anxiety / anxious states | Repo implication |
| --- | --- | --- | --- | --- |
| Reduced positive expressivity / smile attenuation | Strong and recurring depression signal. | Can appear, but evidence is context-sensitive. | Less central than tension, fear, vigilance, or social discomfort. | Add positive-AU and smile-amplitude composites instead of relying only on `happiness`. |
| Brow tension / inner-brow elevation | Appears in older-adult depression and subthreshold-depression work. | Likely relevant through arousal and threat processing. | Fits GAD/threat-tension paradigms. | Track AU01/AU02/AU04 patterns and brow asymmetry over time. |
| AU intensity patterns | One of the strongest facial feature families. | Recent child-PTSD work found AU intensities to be the best baseline feature set. | Useful, but often needs pose and eye-region cues too. | Keep py-feat AU output as the primary emotional-expressivity feature family. |
| Temporal variability / entropy / flexibility | Clinically meaningful in depression; reduced variability often tracks higher severity. | Emerging evidence suggests dynamics may reveal PTSD maintenance mechanisms. | May show rapid context-linked tension fluctuations rather than tonic flattening. | Add windowed dynamics beyond mean/std. |
| Head pose / head movement | Long-running depression cue; head-only models can perform reasonably well. | Included in trauma and child-PTSD datasets. | Especially salient for anxiety-like states. | Current emotional summary path is missing this high-value family. |
| Eye gaze / blink / eye-region cues | Helpful, though often weaker than AUs plus audio in depression models. | Plausible for avoidance/vigilance patterns; still sparse. | Clinically plausible; eye-gaze-only models can be modest but eye-region features still contribute. | Join with blink-rate docs and add gaze/fixation summaries when available. |
| Speech-linked mouth movement | Major confound in interview videos. | Crucial in trauma narratives, where arousal is expressed during recall. | Can mimic or mask anxiety-related tension. | The current speaking split is directionally right but should be upgraded with audio VAD. |

Clinical caution:

- depression, PTSD, and anxiety studies often report group-level patterns, not individual diagnostic certainty
- demographic, lighting, pose, occlusion, camera quality, and task design can affect model output
- emotion models trained on general facial-expression datasets may not transfer cleanly to clinical interviews
- reported accuracies from private or small datasets are encouraging but not definitive
- diagnosis, severity scales, and state self-report are different prediction targets

## AIREST guidance

Recommended AIREST role:

- offline research benchmark
- optional post-session derived table
- not a realtime dependency
- not a clinical gating feature
- not an MVP diagnostic feature

Why:

- py-feat loads a heavy model stack
- runtime is slower and more fragile than MediaPipe landmark extraction
- dependency constraints are narrow
- baseline behavior is not production-clean
- face-detection failures are collapsed into dropped rows without quality metadata
- the PDF emphasizes head pose, gaze, temporal dynamics, and multimodal fusion, which are not fully represented in the current local output

Suggested production boundary:

1. Use AIREST-owned realtime capture for video, face QC, frame timestamps, and optional raw landmarks.
2. Use lightweight MediaPipe-derived features for MVP movement summaries.
3. Run `emotional_expressivity` offline only when research protocols request emotion/AU features.
4. Store emotion/AU outputs separately from realtime `facial_features.csv`.
5. Include processing metadata, model versions, sampling parameters, and missingness metrics.

Recommended artifact names if AIREST adopts this offline feature family:

- `features/emotional_expressivity_framewise.csv`
- `features/emotional_expressivity_summary.csv`
- `features/emotional_expressivity_qc.json`
- `features/emotional_expressivity_meta.json`

### Research-driven feature roadmap

The PDF recommends treating emotional expressivity as a hierarchy rather than one score:

1. State: framewise emotions and AUs.
2. Behavioral episodes: smile episodes, brow tension, gaze aversion, blink bursts.
3. Temporal dynamics: variability, entropy, autocorrelation, onset/offset, burstiness.
4. Context: speaking/listening state, question type, prompt valence, trauma narrative vs neutral baseline.

Recommended next features:

| Feature group | Concrete variables | Priority |
| --- | --- | --- |
| Core AUs | AU01, AU02, AU04, AU05, AU06, AU07, AU10, AU12, AU14, AU15, AU17, AU20, AU23, AU24, AU25, AU26, eye-closure AU; intensity and presence | Very high |
| Valence/arousal proxies | Positive-AU composite, negative-AU composite, neutral proportion, anger/fear/sadness intensity | Very high |
| Mouth and speech separation | Mouth openness, lip compression, smile amplitude, speaking probability, non-speaking expressivity | Very high |
| Head dynamics | Mean/std/range/velocity of yaw, pitch, roll; nod and shake episode counts | High |
| Gaze and eyes | Horizontal/vertical gaze, fixation stability, gaze aversion proportion, blink count, blink duration | High |
| Temporal dynamics | Windowed mean/std, coefficient of variation, sample entropy, spectral entropy, autocorrelation, burstiness, transition counts | Very high |
| Symmetry and laterality | Left-right AU asymmetry, unilateral smile/brow activity, head-turn bias | Medium |
| Quality and confounds | Detection confidence, missingness ratio, occlusion ratio, illumination flag, speaking ratio | Very high |

The PDF's proposed experimental pipeline is:

1. extract framewise features at native frame rate
2. resample to about `10` fps after extraction for efficiency
3. build `2-10` second windows with `50%` overlap
4. compute both raw sequences and window summaries
5. train an interpretable baseline such as elastic net or XGBoost
6. add a sequence model such as BiLSTM or a small transformer encoder
7. add multimodal fusion across face, audio, and text
8. evaluate with subject-exclusive splits, nested cross-validation, AUC, F1, balanced accuracy, sensitivity/specificity, calibration metrics, bootstrapped confidence intervals, and subgroup performance by sex, age, and site

## Technical strengths

What is good:

- returns both framewise and summary outputs
- includes emotion labels and AU-level detail
- supports optional external bounding boxes
- supports optional speaking-state summaries
- reuses the py-feat detector to reduce repeated initialization risk
- adds progress logging for long video passes
- emits source frame and time coordinates

## Technical weaknesses

Important issues:

1. `skip_frames` is a stride, not a target fps.
2. skipped frames are represented internally as `NaN` rows and later dropped.
3. final row count can be smaller than expected without explicit QC output.
4. baseline paths that do not exist silently disable baseline mode.
5. baseline ratio normalization is unstable when baseline means are near zero.
6. summary is computed before final baseline framewise shift.
7. `mouth_openness` is not baseline-normalized but still shifted in returned baseline framewise output.
8. `speaking_probability` can introduce additional null rows before final `dropna()`.
9. the docstring describes an overall expressivity column that does not exist.
10. `bb_dict_to_bb_list()` appears to compute the bbox bottom coordinate incorrectly.
11. broad `try/except` blocks log errors but do not expose structured failure state to callers.
12. no model-version metadata is returned with outputs.

## Improvements worth adding

Highest-value code improvements:

1. Replace `skip_frames` with a real target-fps sampler, or document it everywhere as stride sampling.
2. Return a QC table or metadata object with:
   - source frame count
   - expected sampled frame count
   - successful sampled frame count
   - failed sampled frame count
   - dropped row count
   - baseline used yes/no
   - model stack and versions
3. Warn or raise when a nonempty `baseline_filepath` does not exist.
4. Compute summary from the final returned framewise table.
5. Keep `mouth_openness` out of the global subtract-`1` baseline postprocessing.
6. Add a stable epsilon or alternative transform for near-zero baseline means.
7. Add an explicit `overall_emotional_expressivity` column only if its formula is defined and validated.
8. Fix `bb_dict_to_bb_list()` to use `bb_y + bb_h`.
9. Avoid generating internal rows for skipped frames.
10. Add tests for schema, sampling, baseline consistency, speaking split, and bbox behavior.

## Practical guidance

Use `emotional_expressivity` when:

- you need emotion-category traces
- you need AU-level features
- the run is offline or research-oriented
- you can tolerate model-stack latency
- you can inspect missingness and normalization behavior

Prefer no-baseline mode when:

- you want easy-to-read model scores
- you are plotting raw emotion/AU traces
- no protocol-defined neutral baseline exists

Use baseline mode only when:

- the baseline file is guaranteed to exist
- the protocol defines what the baseline means
- you explicitly want within-subject relative values
- downstream consumers understand the transformed scale

Prefer `facial_expressivity` when:

- you mainly need movement intensity
- you want a simpler, lighter, more stable pipeline
- emotion labels are not required
- outputs may feed an MVP or production-like pipeline

## Source references

Local code:

- `openwillis-face/src/openwillis/face/facial_emotion.py`
- `openwillis-face/src/openwillis/face/util/speaking_utils.py`
- `openwillis-face/src/openwillis/face/util/crop_utils.py`
- `openwillis-face/src/openwillis/face/config/facial.json`

Local docs:

- `README.md`
- `README_openwillis_upstream.md`
- `demo_openwillis_face.ipynb`
- `emotional_expressivity/Emotional Expressivity Biomarkers for PTSD, Depression, and Anxiety Detection.pdf`
- `facial_expression/facial_expression.md`
- `facial_expression/facial_expressivity_feature_inventory.md`
- `facial_expression/airest_openwillis_feature_decision_matrix.md`

External model references already captured by the local docs:

- https://huggingface.co/py-feat/resmasknet
- https://huggingface.co/py-feat/retinaface
- https://huggingface.co/py-feat/mobilefacenet
- https://huggingface.co/py-feat/xgb_au
- https://huggingface.co/py-feat/img2pose
- https://huggingface.co/py-feat/facenet
