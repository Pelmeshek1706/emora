# Facial Expressivity Review: Feature Inventory

This note documents how the local `openwillis-face` implementation produces frame-level and summary-level outputs for:

- `facial_expressivity`
- `emotional_expressivity`

It is based on:

- local source code under `openwillis-face/src/openwillis/face/`
- local repo notes in `facial_expression.md`, `emotional_expressivity.md`, and `README.md`
- notebook cells from `demo_openwillis_face.ipynb`
- live runs in this workspace on `video.mov` with `sample_data/baseline.mp4`

## Notebook Entry Points

Relevant notebook cells:

Cell 2:

```python
import openwillis.face as ef

framewise_loc, framewise_disp, summary = ef.facial_expressivity(
	filepath = 'video.mov', 
	baseline_filepath = '/Users/pelmeshek1706/Downloads/baseline (1).mp4', 
	bbox_list = [], 
	base_bbox_list = [], 
	frames_per_second = 30, 
	normalize = True, 
	align = False, 
	rolling_std_seconds = 3, 
	split_by_speaking = False)
```

Cell 13:

```python
import openwillis.face as owf

framewise, summary = owf.emotional_expressivity(
	filepath = 'video.mov', 
	baseline_filepath = '/Users/pelmeshek1706/Downloads/baseline (1).mp4', 
	bbox_list = [], 
	base_bbox_list = [], 
	skip_frames = 5, 
	split_by_speaking = False, 
	rolling_std_seconds = 3) 
```

For runtime validation here, the missing notebook baseline path was replaced with:

- `sample_data/baseline.mp4`

## 1. `facial_expressivity`

### What the implementation does

Source: `openwillis-face/src/openwillis/face/face_landmark.py`

Pipeline:

1. `get_landmarks()` calls `run_facemesh()` to read the whole video frame by frame.
2. MediaPipe FaceMesh produces 468 landmarks per frame.
3. `normalize_face_landmarks()` optionally:
   - centers landmarks on `lmk001`
   - scales by 3D eye distance between `lmk144` and `lmk373`
   - rotates to level the eyes if `align=True`
4. `get_displacement()` computes per-landmark frame-to-frame Euclidean displacement with `get_distance()`.
5. If a baseline file exists, displacement is normalized as:
   `((main_displacement + 1) / (baseline_mean_displacement + 1)) - 1`
6. `calculate_areas_displacement()` adds composite region means.
7. `get_mouth_openness()` computes a mouth openness ratio from landmark geometry.
8. `get_summary()` summarizes selected composite columns with mean/std suffixes.

### Frame-level outputs

The function returns:

1. `framewise_loc`
2. `framewise_disp`
3. `summary`

#### `framewise_loc`

Per-frame landmark locations.

Columns:

- 468 `x` columns: `lmk001_x` ... `lmk468_x`
- 468 `y` columns: `lmk001_y` ... `lmk468_y`
- 468 `z` columns: `lmk001_z` ... `lmk468_z`
- `frame`
- `time`

Important implementation detail:

- with `normalize=True`, `frame` and `time` are appended at the end of the dataframe
- with `normalize=False`, `get_landmarks()` returns `frame` and `time` first

#### `framewise_disp`

Per-frame displacement table.

Columns:

- `frame`
- `time`
- 468 per-landmark displacement columns: `lmk001` ... `lmk468`
- `overall`
- `lower_face`
- `upper_face`
- `lips`
- `eyebrows`
- `mouth_openness`

How the composite columns are produced:

- `overall`: row mean across all 468 landmark displacement columns
- `lower_face`: mean displacement across indices from `config/facial.json["lower_face_landmarks"]`
- `upper_face`: mean displacement across all landmarks not in `lower_face_landmarks`
- `lips`: mean displacement across `config/facial.json["lips_landmarks"]`
- `eyebrows`: mean displacement across `config/facial.json["eyebrows_landmarks"]`
- `mouth_openness`: `mouth_height / min(upper_lip_height, lower_lip_height)`

How `mouth_openness` is produced:

- `mouth_height` sums three upper/lower lip pair distances
- `upper_lip_height` and `lower_lip_height` each sum three within-lip distances
- ratios are computed from the normalized or raw landmark table used for the run

If `split_by_speaking=True`, `framewise_disp` also gets:

- `speaking_probability`

Speaking probability is estimated by:

1. computing rolling std of `mouth_openness`
2. fitting a 2-component Gaussian mixture model
3. assigning the higher-variance cluster as the speaking-like cluster

### Summary output

Actual implementation summary columns are only composite features, not all 468 landmarks.

With `split_by_speaking=False`, `summary` contains one row with:

- `overall_mean`
- `lower_face_mean`
- `upper_face_mean`
- `lips_mean`
- `eyebrows_mean`
- `mouth_openness_mean`
- `overall_std`
- `lower_face_std`
- `upper_face_std`
- `lips_std`
- `eyebrows_std`
- `mouth_openness_std`

With `split_by_speaking=True`, the same metrics are duplicated as:

- `[metric]_speaking`
- `[metric]_not_speaking`

Example:

- `overall_mean_speaking`
- `overall_mean_not_speaking`

### Live run in this workspace

Inputs used:

- `filepath='video.mov'`
- `baseline_filepath='sample_data/baseline.mp4'`
- `frames_per_second=30`
- `normalize=True`
- `align=False`
- `split_by_speaking=False`

Returned shapes:

- `framewise_loc.shape == (644, 1406)`
- `framewise_disp.shape == (644, 476)`
- `summary.shape == (1, 12)`

Observed key columns:

- `framewise_loc` begins with `lmk001_x ...`
- `framewise_disp` ends with `overall`, `lower_face`, `upper_face`, `lips`, `eyebrows`, `mouth_openness`

Observed summary columns:

- `overall_mean`, `lower_face_mean`, `upper_face_mean`, `lips_mean`, `eyebrows_mean`, `mouth_openness_mean`
- `overall_std`, `lower_face_std`, `upper_face_std`, `lips_std`, `eyebrows_std`, `mouth_openness_std`

## 2. `emotional_expressivity`

### What the implementation does

Source: `openwillis-face/src/openwillis/face/facial_emotion.py`

Pipeline:

1. `get_emotion()` calls `run_pyfeat()`.
2. `run_pyfeat()` opens the video and samples frames using `skip_frames`.
3. For sampled frames, `detect_emotions()` runs:
   - `detect_faces`
   - `detect_landmarks`
   - `detect_aus`
   - `detect_emotions`
4. Emotion outputs are multiplied by 100.
5. AU outputs are appended.
6. `mouth_openness()` is computed from three upper-lip vs lower-lip landmark pairs.
7. If a baseline file exists, emotion and AU columns are normalized against baseline means.
8. `get_summary()` summarizes all non-metadata columns with mean/std suffixes.

### Available emotion probability outputs

Source of names:

- `feat.utils.FEAT_EMOTION_COLUMNS`

Installed runtime values:

- `anger`
- `disgust`
- `fear`
- `happiness`
- `sadness`
- `surprise`
- `neutral`

How they are produced:

- `py-feat` returns emotion scores
- the local code converts them from 0-1 to 0-100 with `emotions = emotions[0][0] * 100`

Important interpretation note:

- without baseline, these are raw py-feat emotion scores on a 0-100 scale
- with baseline, the returned framewise table is no longer a probability table; it becomes a relative-to-baseline table after normalization and subtraction

### Default py-feat model cards

The local implementation instantiates `feat.Detector()` with default arguments. In the installed py-feat runtime here, that resolves to:

- `emotion_model='resmasknet'` -> [py-feat/resmasknet](https://huggingface.co/py-feat/resmasknet)
- `face_model='retinaface'` -> [py-feat/retinaface](https://huggingface.co/py-feat/retinaface)
- `landmark_model='mobilefacenet'` -> [py-feat/mobilefacenet](https://huggingface.co/py-feat/mobilefacenet)
- `au_model='xgb'` -> [py-feat/xgb_au](https://huggingface.co/py-feat/xgb_au)
- `facepose_model='img2pose'` -> [py-feat/img2pose](https://huggingface.co/py-feat/img2pose)
- `identity_model='facenet'` -> [py-feat/facenet](https://huggingface.co/py-feat/facenet)

For `emotional_expressivity`, the emotion labels above come from the `resmasknet` model card, which documents the 7-class output used by py-feat.

#### `py-feat/resmasknet` model capabilities

The upstream RMN model is a 7-class facial emotion classifier with outputs aligned to:

- angry
- disgust
- fear
- happy
- sad
- surprise
- neutral

Upstream benchmark data and reported results:

- Dataset: [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
- Reported single-model accuracy on FER2013: 74.14%
- Reported 7-model ensemble accuracy on FER2013: 76.82%

Source: [phamquiluan/ResidualMaskingNetwork](https://github.com/phamquiluan/ResidualMaskingNetwork)

### Available action-unit-related outputs

Source of names:

- `feat.pretrained.AU_LANDMARK_MAP['Feat']`

Installed runtime values:

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

These are emitted once per analyzed frame and are also summarized with mean/std columns.

### Mouth openness

`emotional_expressivity` also returns:

- `mouth_openness`

How it is produced in this module:

- the code selects upper lip landmarks `[61, 62, 63]`
- the code selects lower lip landmarks `[65, 66, 67]`
- it averages the Euclidean distances between those matched landmark pairs

This value is used both as a standalone feature and as the input signal for the optional speaking split.

### Frame-level outputs

The function returns:

1. `framewise`
2. `summary`

#### `framewise`

Columns observed in the live run:

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

If `split_by_speaking=True`, `framewise` also gets:

- `speaking_probability`

Frame retention behavior:

- `run_pyfeat()` creates one row per original frame
- skipped frames are filled with NaNs
- `emotional_expressivity()` calls `dropna()`
- the returned framewise table therefore only keeps analyzed frames that survived detection

With `skip_frames=5` on a 644-frame video, the live run returned 108 rows.

### Summary output

With `split_by_speaking=False`, the summary contains one row with mean/std columns for all 28 measured features:

- `mouth_openness_mean`
- 7 emotion means
- 20 AU means
- `mouth_openness_std`
- 7 emotion std columns
- 20 AU std columns

Total:

- 56 summary columns

With `split_by_speaking=True`, the summary duplicates those statistics into:

- `[metric]_speaking`
- `[metric]_not_speaking`

### Live run in this workspace

Inputs used:

- `filepath='video.mov'`
- `baseline_filepath='sample_data/baseline.mp4'`
- `skip_frames=5`
- `split_by_speaking=False`

Returned shapes:

- `framewise.shape == (108, 30)`
- `summary.shape == (1, 56)`

Observed framewise columns:

- metadata: `frame`, `time`
- scalar behavior feature: `mouth_openness`
- 7 emotions
- 20 AUs

Observed summary columns:

- `mouth_openness_mean`
- emotion means: `anger_mean` ... `neutral_mean`
- AU means: `AU01_mean` ... `AU43_mean`
- matching `_std` columns for the same features

## 3. Result Table Inventory

### Landmark movement pipeline

| Table | Granularity | Features present |
| --- | --- | --- |
| `framewise_loc` | per frame | 468 x/y/z landmark positions plus `frame`, `time` |
| `framewise_disp` | per frame | 468 landmark displacements plus `overall`, `lower_face`, `upper_face`, `lips`, `eyebrows`, `mouth_openness`, optional `speaking_probability` |
| `summary` | one row per run | means/stds for `overall`, `lower_face`, `upper_face`, `lips`, `eyebrows`, `mouth_openness`, optionally split by speaking |

### Emotion/AU pipeline

| Table | Granularity | Features present |
| --- | --- | --- |
| `framewise` | sampled frames only after `dropna()` | `mouth_openness`, 7 emotion outputs, 20 AU outputs, optional `speaking_probability`, plus `frame`, `time` |
| `summary` | one row per run | means/stds for `mouth_openness`, all 7 emotions, all 20 AUs, optionally split by speaking |

## 4. Important Review Findings

These affect how outputs should be interpreted.

1. The summary tables are not multi-row statistical tables.
   The shared `get_summary()` helper returns a single row with suffixed columns such as `_mean` and `_std`.

2. `facial_expressivity` summary excludes individual landmark summaries.
   Despite docstrings that describe landmark-level summary output, the actual call `get_summary(df_disp, 470)` starts summarization at `overall`.

3. `facial_expressivity` first-frame displacement is not zero in the dataframe; it is NaN.
   `get_distance()` uses `.shift()` and does not fill the first row.

4. `facial_expressivity` ignores `frames_per_second` for sampling.
   The parameter is present for compatibility, but the current implementation processes native frames and uses native timestamps.

5. `emotional_expressivity` does not create an "overall expressivity" or "composite" output column.
   The docstring and `facial.json["comp_exp"]` suggest such a concept, but the function does not compute or return it.

6. In `emotional_expressivity`, baseline normalization changes the semantic meaning of emotion outputs.
   After baseline correction, returned framewise values are relative-to-baseline ratios shifted by `-1`, not raw probabilities.

7. In `emotional_expressivity`, `mouth_openness` is not baseline-normalized, but it is still shifted by `-1` in the final returned framewise table when a baseline file exists.
   This happens because the code subtracts `1` from the whole dataframe and only restores `frame` and `time`.

8. In `facial_expressivity`, the baseline helper appears to ignore normalized baseline landmarks when calculating baseline displacement.
   The `normalize=True` branch creates a normalized baseline dataframe, but `get_distance()` is still applied to `base_landmark`, not the normalized landmark dataframe.

9. The notebook baseline path is not portable in this workspace.
   The notebook points to `/Users/pelmeshek1706/Downloads/baseline (1).mp4`; the local sample baseline is `sample_data/baseline.mp4`.

## 5. Source References

Local code:

- `openwillis-face/src/openwillis/face/face_landmark.py`
- `openwillis-face/src/openwillis/face/facial_emotion.py`
- `openwillis-face/src/openwillis/face/util/speaking_utils.py`
- `openwillis-face/src/openwillis/face/config/facial.json`

Local docs:

- `README.md`
- `facial_expression.md`
- `emotional_expressivity.md`
- `demo_openwillis_face.ipynb`
