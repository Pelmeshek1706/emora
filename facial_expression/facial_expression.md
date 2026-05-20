# Facial Expression (`facial_expressivity`)

Upstream documentation:

- https://openwillis.brooklyn.health/Facial-Expressivity-v2-2-1b483a8fe04780ae9c63d9e4034a8463

Local implementation reviewed here:

- `openwillis-face/src/openwillis/face/face_landmark.py`
- `openwillis-face/src/openwillis/face/config/facial.json`
- `openwillis-face/src/openwillis/face/util/speaking_utils.py`

This note combines:

- the upstream OpenWillis method description
- the actual code in this repo
- results from the local demo notebook run

## What this feature is for

`facial_expressivity` is the movement-based face signal in OpenWillis.

It does not try to answer "what emotion is this?"
It tries to answer:

- how much did the face move
- where in the face did the movement happen
- did movement increase relative to a baseline clip
- did movement differ during speaking vs not speaking

This makes it useful when you care more about facial animation intensity than about emotion labels.

## Upstream docs vs local code

The upstream docs describe the function as sampling at `frames_per_second`.

The local repo keeps the `frames_per_second` argument for compatibility, but the current implementation does not resample the video from that parameter. It reads native frames and uses native frame timing from the file metadata.

The upstream docs also describe:

- 468 landmark positions in `framewise_loc`
- framewise landmark displacement in `framewise_disp`
- composite measures for `overall`, `upper_face`, `lower_face`, `lips`, `eyebrows`
- `mouth_openness`
- optional split by speaking

Those parts broadly match the local implementation.

## Demo notebook call

Notebook file:

- `demo_openwillis_face.ipynb`

Relevant cell:

```python
import openwillis.face as ef

framewise_loc, framewise_disp, summary = ef.facial_expressivity(
    filepath='video.mov',
    baseline_filepath='/Users/pelmeshek1706/Downloads/baseline (1).mp4',
    bbox_list=[],
    base_bbox_list=[],
    frames_per_second=30,
    normalize=True,
    align=False,
    rolling_std_seconds=3,
    split_by_speaking=False,
)
```

Important repo-specific note:

- the notebook baseline path points outside the repo
- in this workspace that file is not present
- if you run the notebook exactly as written here, the function silently behaves as if no baseline was supplied

For the validated run below, the same call was executed with:

- `baseline_filepath='sample_data/baseline.mp4'`

## Demo result in this workspace

Input media:

- main video: `video.mov`
- baseline video: `sample_data/baseline.mp4`
- main video fps: `30`
- main video frame count: `644`
- baseline frame count: `104`

Returned objects:

- `framewise_loc.shape == (644, 1406)`
- `framewise_disp.shape == (644, 476)`
- `summary.shape == (1, 12)`

Observed summary values:

| Metric | Value |
| --- | ---: |
| `overall_mean` | `0.015443` |
| `lower_face_mean` | `0.014805` |
| `upper_face_mean` | `0.015817` |
| `lips_mean` | `0.013397` |
| `eyebrows_mean` | `0.022610` |
| `mouth_openness_mean` | `0.878355` |
| `overall_std` | `0.008900` |
| `lower_face_std` | `0.010404` |
| `upper_face_std` | `0.009567` |
| `lips_std` | `0.011449` |
| `eyebrows_std` | `0.015660` |
| `mouth_openness_std` | `0.662627` |

What that means at a high level:

- eyebrow movement was the strongest of the region-level displacement features in this clip
- lip displacement was somewhat lower than eyebrow displacement
- `mouth_openness` is on a different geometric ratio scale and should not be compared directly to the displacement columns

## Feature inventory

### 1. `framewise_loc`

This is the frame-level landmark table.

Features:

- 468 `x` columns: `lmk001_x` ... `lmk468_x`
- 468 `y` columns: `lmk001_y` ... `lmk468_y`
- 468 `z` columns: `lmk001_z` ... `lmk468_z`
- `frame`
- `time`

How it is produced:

1. every video frame is read with OpenCV
2. MediaPipe FaceMesh estimates 468 landmarks
3. if `normalize=True`, the landmarks are:
   - centered on `lmk001`
   - scaled by eye distance between `lmk144` and `lmk373`
   - optionally rotated to level the eyes if `align=True`

Important local detail:

- with `normalize=True`, the returned dataframe starts with landmark columns and places `frame` and `time` at the end
- that is why the local demo returned `1406` columns instead of the `1405` columns shown in the upstream example screenshots

### 2. `framewise_disp`

This is the frame-level movement table.

Features:

- `frame`
- `time`
- 468 per-landmark displacement columns: `lmk001` ... `lmk468`
- `overall`
- `lower_face`
- `upper_face`
- `lips`
- `eyebrows`
- `mouth_openness`

Optional:

- `speaking_probability` if `split_by_speaking=True`

How the main features are produced:

- each `lmk###` column is the Euclidean distance between the landmark position in the current frame and the previous frame
- `overall` is the mean of all 468 landmark displacement columns
- `lower_face` is the mean displacement across landmark indices listed in `facial.json["lower_face_landmarks"]`
- `upper_face` is the mean displacement across the remaining landmarks
- `lips` is the mean displacement across `facial.json["lips_landmarks"]`
- `eyebrows` is the mean displacement across `facial.json["eyebrows_landmarks"]`

### 3. `mouth_openness`

This is not a displacement feature.

It is computed as:

`mouth_height / min(upper_lip_height, lower_lip_height)`

where:

- `mouth_height` is derived from three upper/lower lip point pairs
- `upper_lip_height` and `lower_lip_height` are derived from simple lip landmark subsets from `facial.json`

This feature is useful on its own and also drives the speaking split.

### 4. `summary`

The local implementation returns a one-row summary table with these columns:

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

If `split_by_speaking=True`, each feature is duplicated as:

- `[feature]_[stat]_speaking`
- `[feature]_[stat]_not_speaking`

## How the algorithm produces the outputs

### Without baseline

This is the intended method described by the upstream docs and mostly implemented locally:

1. detect 468 landmarks per frame
2. optionally normalize the face into a common face-centered coordinate space
3. compute frame-to-frame Euclidean displacement for each landmark
4. average subsets of landmarks into composite region measures
5. summarize the composite measures

### With baseline

The upstream docs recommend baseline clips for within-subject interpretation.

The local code applies baseline normalization to the displacement table as:

`((main_displacement + 1) / (baseline_mean_displacement + 1)) - 1`

Interpretation:

- `0` means approximately equal to baseline
- positive values mean more movement than baseline
- negative values mean less movement than baseline

## Why nulls appear in the demo

In the validated local run:

- `framewise_loc` had `0` nulls
- `framewise_disp` had `473` nulls total
- all `473` nulls were in the first row only
- rows `1:` had `0` nulls

Why:

- displacement is computed with `.shift()`
- the first frame has no previous frame to compare against
- so the first row of the 468 landmark displacement columns plus `overall`, `lower_face`, `upper_face`, `lips`, and `eyebrows` becomes `NaN`
- `mouth_openness` is still available on the first row because it is computed from geometry within that frame, not between frames

This is expected from the current implementation.

If you see additional nulls beyond the first row, the likely causes are:

- MediaPipe failed to detect the face in a frame
- a provided bounding box had missing coordinates
- the video could not be decoded cleanly

## How to use this feature

Good use cases:

- overall facial movement intensity during an interview or task
- region-specific movement analysis:
  - upper face
  - lower face
  - lips
  - eyebrows
- comparing task video vs subject baseline video
- deriving custom movement scores from selected landmarks
- separating speaking vs non-speaking movement when the clip clearly contains speech

How to interpret the main outputs:

- start with `overall` for a single movement summary
- use `upper_face` and `eyebrows` when you care about brow/forehead activity
- use `lower_face`, `lips`, and `mouth_openness` when you care about speech-like or mouth-centered movement
- use baseline only when the baseline clip is meaningful and stable

## What the demo suggests

In this sample clip:

- `eyebrows_mean` was higher than `lips_mean` and slightly higher than `overall_mean`
- that suggests the clip contains noticeable upper-face motion rather than only mouth-centered movement
- `mouth_openness` varied substantially over time, so speaking-based partitioning could be reasonable here if needed

This is still a movement signal, not an affect label.

## Research context from the PDF

Source:

- `Facial Expressivity Features and Biomarkers for PTSD, Depression, and Anxiety Disorders.pdf`

The paper frames facial expressivity as a candidate digital biomarker for screening and monitoring, not as a standalone diagnostic test. Its main cross-disorder conclusion is that temporal facial dynamics are more useful than static frames.

Key points that map onto this implementation:

- depression-related facial markers are usually described as reduced positive expressivity, lower overall facial variability, smoother temporal dynamics, reduced head movement, and altered AU patterns such as lower AU12/AU15 and higher AU14
- anxiety-disorder signals are thinner but promising, especially gaze behavior, head pose, facial landmarks, and selected AUs during social interaction
- PTSD facial-video evidence is the sparsest of the three groups, and the field is still building datasets and benchmarks
- across the reviewed papers, AUs, landmarks, head motion/pose, gaze/eye measures, and temporal modeling are more common and more actionable than microexpression-only pipelines

Benchmark results highlighted in the paper:

- `AnxietyFaceTrack` on 91 participants reported `91.0%` multiclass accuracy and `92.33%` average binary accuracy using head position, landmarks, eye movements, and AUs
- `PTSD in the Wild` used `634` total videos inferred from the published split counts (`317` PTSD / `317` non-PTSD) and reported a visual baseline test accuracy, precision, recall, and F1 of `0.82` with `ResNet50v2 + LSTM`
- the paper also notes that depression work is the most mature area, with repeated findings around reduced positive expressivity and reduced facial reactivity in challenge and naturalistic datasets

Practical read-through for this repo:

- `facial_expressivity` fits the paper’s recommended direction because it measures temporal movement, region-level dynamics, and within-person baseline change
- the outputs here are better interpreted as movement biomarkers than disorder labels
- if you use the paper as background, the strongest fit is the baseline-vs-current comparison and the speaking/non-speaking split, not a direct claim of psychiatric diagnosis

## Is the current implementation technically bad?

Not fundamentally bad, but not production-tight.

What is good:

- the core idea is sensible
- the pipeline is readable
- the output tables are useful for downstream analysis
- region-level composites are practical and easy to interpret
- the function can be used with or without baseline clips

What is weak or inconsistent:

1. `frames_per_second` is documented as meaningful but is ignored by the local code.
2. The docstring says the first displacement row is zero, but the implementation returns `NaN`.
3. The docstring implies richer summary output than what is actually returned.
4. `framewise_loc` column ordering changes after normalization.
5. The baseline helper appears to compute baseline displacement from raw baseline landmarks even when `normalize=True`, which is probably not what was intended.
6. The function does not expose detection-quality metadata, so downstream users cannot easily distinguish "low signal" from "failed tracking."

So the implementation is usable for local research work, but it still has code-quality and method-reporting gaps.

## Improvements worth adding

1. Honor `frames_per_second` or remove it from the local API and docs.
2. Fill first-row displacement values with `0` explicitly instead of leaving them as `NaN`.
3. Keep `frame` and `time` in a stable position in `framewise_loc`.
4. Return a detection-quality column such as:
   - face detected yes/no
   - bounding box used yes/no
   - landmark completeness
5. Fix the baseline path behavior so missing baseline files raise a clear warning instead of silently changing analysis mode.
6. Revisit baseline normalization to ensure the baseline displacement is computed from the same normalized landmark space as the main run.
7. Add tests for:
   - shape contracts
   - first-row behavior
   - missing baseline behavior
   - speaking split columns

## Practical guidance

- Use `facial_expressivity` when you want a stable movement measure and do not need emotion labels.
- Prefer this function over `emotional_expressivity` when model interpretability is less important than movement magnitude.
- For research interpretation, always record whether the run was:
  - raw vs baselined
  - normalized vs not normalized
  - split by speaking vs not split by speaking
