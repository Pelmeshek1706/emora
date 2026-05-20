# Emotional Expressivity (`emotional_expressivity`)

Upstream documentation:

- https://openwillis.brooklyn.health/Emotional-expressivity-v2-1-15883a8fe04780f987a8cf2bcbd60b8e

Local implementation reviewed here:

- `openwillis-face/src/openwillis/face/facial_emotion.py`
- `openwillis-face/src/openwillis/face/util/speaking_utils.py`

This note combines:

- the upstream OpenWillis method description
- the actual code in this repo
- results from the local demo notebook run

## What this feature is for

`emotional_expressivity` is the label-based face signal in OpenWillis.

Unlike `facial_expressivity`, which measures movement magnitude, this function tries to characterize:

- emotion-like facial judgments
- action unit activations
- mouth openness
- optional speaking vs not-speaking summaries

This is the function to use when you want interpretable expression categories or AU traces, not just movement intensity.

## Upstream docs vs local code

The upstream docs describe:

- emotion judgments for:
  - happiness
  - sadness
  - anger
  - fear
  - disgust
  - surprise
  - neutral
- raw emotion scores on a `0-100` scale
- raw action unit values on a `0-1` scale
- optional speaking split
- a `frames_per_second` style sampling control

The local code matches the general method, but with important implementation differences:

- the local function signature uses `skip_frames`, not `frames_per_second`
- the final output only contains frames that survive detection and `dropna()`
- baseline normalization behaves differently from what a user would expect from the docs

## Demo notebook call

Notebook file:

- `demo_openwillis_face.ipynb`

Relevant cell:

```python
import openwillis.face as owf

framewise, summary = owf.emotional_expressivity(
    filepath='video.mov',
    baseline_filepath='/Users/pelmeshek1706/Downloads/baseline (1).mp4',
    bbox_list=[],
    base_bbox_list=[],
    skip_frames=5,
    split_by_speaking=False,
    rolling_std_seconds=3,
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
- `skip_frames=5`

Returned objects:

- `framewise.shape == (108, 30)`
- `summary.shape == (1, 56)`

Returned framewise columns:

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

Observed summary values from the baselined run:

### Emotion summary columns

| Metric | Value |
| --- | ---: |
| `surprise_mean` | `24.191500` |
| `anger_mean` | `7.465008` |
| `happiness_mean` | `4.830577` |
| `fear_mean` | `3.815408` |
| `disgust_mean` | `1.101504` |
| `sadness_mean` | `0.820626` |
| `neutral_mean` | `0.307490` |

### Highest AU summary columns

| Metric | Value |
| --- | ---: |
| `AU25_mean` | `1.682072` |
| `AU20_mean` | `1.527778` |
| `AU26_mean` | `1.325430` |
| `AU10_mean` | `1.311244` |
| `AU12_mean` | `1.217659` |
| `AU14_mean` | `1.188605` |
| `AU06_mean` | `1.115396` |
| `AU28_mean` | `0.977249` |

Interpretation caveat:

- these are the summary values returned by the current implementation
- in a baselined run they are not clean probabilities
- some of these values become very large because the local baseline normalization divides by baseline means that can be close to zero

So the pattern is informative, but the absolute magnitudes should be treated cautiously.

## Feature inventory

### 1. Emotion outputs

The local runtime exposes these emotion columns:

- `anger`
- `disgust`
- `fear`
- `happiness`
- `sadness`
- `surprise`
- `neutral`

How they are produced:

- `py-feat` estimates face-level emotion scores
- the local code multiplies them by `100`

Without baseline:

- they are best read as model scores on a `0-100` style scale

With baseline:

- they become relative-to-baseline transformed values
- they are no longer plain probabilities or percentages

### 2. Action-unit-related outputs

The local runtime exposes these AU columns:

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

These are taken from `feat.pretrained.AU_LANDMARK_MAP['Feat']`.

Without baseline:

- the upstream docs describe them as `0-1` style activations

With baseline:

- they become relative-to-baseline transformed values

### 3. `mouth_openness`

The function also adds:

- `mouth_openness`

How it is produced in the local code:

- upper lip landmarks: `[61, 62, 63]`
- lower lip landmarks: `[65, 66, 67]`
- average Euclidean distance across those matched lip point pairs

This feature is both:

- a useful mouth-related signal
- the basis for the optional speaking split

### 4. `summary`

The local summary contains:

- `mouth_openness_mean`
- 7 emotion means
- 20 AU means
- `mouth_openness_std`
- 7 emotion std columns
- 20 AU std columns

That is:

- `28` measured features
- `56` summary columns total

If `split_by_speaking=True`, each feature is duplicated into:

- `[feature]_[stat]_speaking`
- `[feature]_[stat]_not_speaking`

## How the algorithm produces the outputs

### Frame sampling

The local code samples frames with `skip_frames`.

For `skip_frames=5`:

- analyze one frame
- skip the next five
- analyze the next one

On a `644`-frame video, that produces about:

- `108` analyzed frames

This matches the validated run exactly.

### Per-sampled-frame processing

For each sampled frame:

1. detect face
2. detect landmarks
3. detect AUs
4. detect emotions
5. compute `mouth_openness`

The implementation uses `py-feat` for the face, landmark, AU, and emotion passes.

### Summary generation

The function computes a one-row summary table using:

- mean of each feature
- standard deviation of each feature

## Why nulls happen in the demo

This is the main reason users get confused by this function.

### Internal raw behavior

Inside `run_pyfeat()` and `get_emotion()`:

- one row is created for every original video frame
- skipped frames are filled with `NaN` features
- sampled frames contain emotion/AU data if detection succeeds

For this demo:

- total original frames: `644`
- sampled frames with data: `108`
- intentionally skipped frames: `536`

So the internal raw table contains `536` rows of expected null features.

Those null rows are not failures.
They are how the sampling strategy is implemented.

### Final public behavior

`emotional_expressivity()` then runs:

- `dropna()`
- `reset_index(drop=True)`

So the final returned `framewise` table:

- keeps only rows that contain data
- had `108` rows in this demo
- had `0` null values in the final validated run

### When nulls would indicate a real problem

If you modify the function or inspect internal raw outputs, additional null rows beyond expected skipped frames would usually mean:

- face detection failed on a sampled frame
- the frame crop was bad
- the video frame was unreadable
- a provided bounding box was invalid

## How to use this feature

Good use cases:

- frame-level emotion-like facial traces
- AU-based analysis when you need more granular facial behavior than emotion labels provide
- comparing within-subject expression relative to a baseline clip
- splitting outputs into speaking vs not speaking segments
- exploratory inspection of expression-rich moments in a video

Recommended interpretation workflow:

1. inspect raw no-baseline output first if you want readable model outputs
2. use baseline mode only for relative within-subject comparisons
3. look at AUs together with `mouth_openness`, not in isolation
4. use speaking split only when the clip genuinely contains speaking and silence

## What the demo suggests

Qualitatively, this demo clip looks like a strongly expressive, mouth-active clip.

Why that conclusion is reasonable:

- `surprise_mean` is much larger than the other emotion summary columns
- `AU25`, `AU26`, and `AU10` are elevated, which is consistent with mouth activity
- `mouth_openness` is also high in many returned frames

What not to over-interpret:

- the exact size of the baselined emotion means
- the exact size of the baselined AU means

The direction of the signal is useful.
The absolute magnitude is less trustworthy in the current implementation.

## Is the current implementation technically bad?

It is usable, but it has more technical issues than `facial_expressivity`.

What is good:

- useful combination of emotion judgments and AUs
- framewise output is practical for downstream plots and event analysis
- summary output is compact
- speaking split is available without needing a separate voice pipeline

What is weak or inconsistent:

1. The upstream docs describe a `frames_per_second` control, while the local code uses `skip_frames`.
2. Baseline normalization is ratio-based, which becomes unstable when baseline emotion means are near zero.
3. The summary is computed before the final `-1` shift applied to the returned baselined framewise table, so `summary` and `framewise` are not on the same transformed scale.
4. `mouth_openness` is not baseline-normalized, but in a baselined run it still gets shifted by `-1` because the code subtracts `1` from the whole dataframe and only restores `frame` and `time`.
5. If `split_by_speaking=True` in a baselined run, `speaking_probability` is also shifted by the same global subtraction/addition pattern.
6. The docstring mentions overall expressivity, but the function does not actually create a composite or overall emotion column.
7. The implementation is inefficient because it writes null rows for skipped frames and then drops them later.
8. There is a bounding-box helper bug in `bb_dict_to_bb_list()`:
   - it uses `bb_y + bb_y`
   - it should almost certainly use `bb_y + bb_h`
   - this does not affect the default notebook path because the notebook passes empty bbox lists

So the implementation is not unusable, but it is methodologically fragile in baseline mode and needs cleanup.

## Improvements worth adding

1. Replace `skip_frames` with a real `frames_per_second` sampler or make the docs match the code exactly.
2. Avoid creating null rows for skipped frames; return only sampled frames from the start.
3. Separate raw outputs from baseline-corrected outputs explicitly.
4. Compute summary from the final returned framewise table so the two outputs stay on the same scale.
5. Do not subtract `1` from `mouth_openness` or `speaking_probability`.
6. Stabilize baseline normalization with an epsilon floor or a different normalization strategy for near-zero baseline means.
7. Add an explicit composite emotional expressivity measure if the docs are going to reference one.
8. Warn clearly when the baseline file path does not exist instead of silently switching to no-baseline mode.
9. Add tests for:
   - sampling behavior
   - null row handling
   - baseline summary consistency
   - bbox helper correctness

## Practical guidance

- Use `emotional_expressivity` when you need emotion categories or AU-level features.
- Use it without baseline when you want outputs that are easiest to interpret.
- Use it with baseline only when you are intentionally doing relative within-subject comparison and are prepared to inspect normalization artifacts.
- If you mainly care about overall facial movement rather than emotion labels, `facial_expressivity` is technically cleaner in this repo.
