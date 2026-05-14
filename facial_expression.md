# Facial Expression (`facial_expressivity`)

This note explains the facial expressivity function in plain language.

Source document:
- https://openwillis.brooklyn.health/Facial-Expressivity-v2-2-1b483a8fe04780ae9c63d9e4034a8463

## What it measures

`facial_expressivity` measures how much the face moves over time in a video.

It is important to separate this from emotion classification:

- `facial_expressivity` measures movement
- it does **not** directly decide whether someone is happy, sad, angry, or surprised

In other words, this function answers "how much did the face move?" more than "what emotion was shown?"

## Basic use

```python
import openwillis.face as owf

framewise_loc, framewise_disp, summary = owf.facial_expressivity(
    filepath="video.mov",
    baseline_filepath="baseline_video.mov",
    bbox_list=[],
    base_bbox_list=[],
    frames_per_second=10,
    normalize=True,
    align=False,
    rolling_std_seconds=3,
    split_by_speaking=False,
)
```

## What comes back

The function returns three tables:

1. `framewise_loc`
   The face landmark positions for each frame.
2. `framewise_disp`
   How much those landmarks moved.
3. `summary`
   A compact summary of the main movement measures.

## How it works

### 1. It finds the face landmarks

For each frame, MediaPipe FaceMesh detects 468 facial landmarks.

If you already have face bounding boxes, you can pass them in with `bbox_list` so the function can crop to the face before landmark detection.

### 2. It can normalize the face

If `normalize=True`, the coordinates are moved into a common face-centered space so results are less affected by where the face appears in the image.

In the local code, this normalization does three things:

1. centers the face around the nose-tip landmark
2. scales the face by eye distance
3. optionally rotates the face so the eyes are more level when `align=True`

This makes movement easier to compare across frames and clips.

### 3. It saves landmark locations

The raw landmark coordinates are returned in `framewise_loc`.

Each landmark has:

- an `x` coordinate
- a `y` coordinate
- a `z` coordinate

So the output contains columns like:

- `lmk001_x`
- `lmk001_y`
- `lmk001_z`

along with `frame` and `time`.

### 4. It measures movement between frames

For each landmark, the function calculates how far that point moved from one frame to the next.

This movement is stored in `framewise_disp`.

- The first frame is effectively the starting point, so its displacement values are `0`.
- Larger values mean more facial movement.

### 5. It creates easier-to-read group measures

Instead of only returning 468 separate landmark movement columns, the function also averages movement across larger face regions:

- `overall`
- `upper_face`
- `lower_face`
- `lips`
- `eyebrows`

These are usually the easiest columns to read first.

### 6. It calculates `mouth_openness`

The function also calculates `mouth_openness`.

In the local implementation, this is:

`mouth height / min(upper lip height, lower lip height)`

This is useful on its own and is also used for the optional speaking split.

### 7. It can compare the clip to a baseline

If you provide `baseline_filepath`, the function also analyzes that baseline clip and uses it as a reference point.

That means the main clip is no longer interpreted only in absolute terms. Instead, it is interpreted relative to the person's own baseline movement.

In the local code, the baseline correction is:

`((main + 1) / (baseline + 1)) - 1`

That means:

- positive values mean more movement than baseline
- negative values mean less movement than baseline
- `0` means roughly the same as baseline

Without a baseline clip, displacement values are movement values from the clip itself and usually fall in a `0` to `1` style range.

### 8. It can split results by speaking vs not speaking

If `split_by_speaking=True`, the function estimates when the person is speaking.

It does this by looking at how much `mouth_openness` varies over time:

1. it calculates a rolling standard deviation of `mouth_openness`
2. it fits a 2-group model to that signal
3. it treats the more variable group as more likely to be speaking

This adds:

- `speaking_probability` to `framewise_disp`

and changes the summary so you get separate results for:

- speaking frames
- non-speaking frames

## Inputs

### `filepath`

Path to the main video.

### `baseline_filepath`

Optional path to a baseline video. Useful when you want movement relative to a calmer or reference clip.

### `bbox_list`

Optional face boxes for the main video.

### `base_bbox_list`

Optional face boxes for the baseline video.

### `frames_per_second`

The upstream documentation describes this as the analysis rate.

In this local package copy, the argument is kept for compatibility, but the implementation uses the video's own timing metadata and does not currently resample based on this parameter.

### `normalize`

Whether to place faces into a shared face-centered coordinate space.

### `align`

Whether to rotate the face so the eyes are more level.

### `rolling_std_seconds`

Window size used when estimating speaking probability.

### `split_by_speaking`

Whether to produce speaking and non-speaking summaries separately.

## Outputs

### 1. `framewise_loc`

A frame-by-frame table of landmark positions.

Typical columns include:

- `frame`
- `time`
- `lmk001_x ... lmk468_x`
- `lmk001_y ... lmk468_y`
- `lmk001_z ... lmk468_z`

### 2. `framewise_disp`

A frame-by-frame table of landmark movement.

Typical columns include:

- `frame`
- `time`
- `lmk001 ... lmk468`
- `overall`
- `lower_face`
- `upper_face`
- `lips`
- `eyebrows`
- `mouth_openness`
- `speaking_probability` when `split_by_speaking=True`

### 3. `summary`

When `split_by_speaking=False`, the summary contains means and standard deviations for the main composite measures, such as:

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

When `split_by_speaking=True`, the same style of metrics is returned separately for speaking and not-speaking segments.

## How to interpret the main columns

- `overall`: a general measure of how much the face moved
- `upper_face`: movement in the upper part of the face
- `lower_face`: movement in the lower part of the face
- `lips`: movement centered on the lips
- `eyebrows`: movement centered on the eyebrows
- `mouth_openness`: how open the mouth is in each frame

If you want one simple starting point, `overall` is usually the most useful column to inspect first.

## Practical advice

- Use a clip where the face is visible for most of the recording.
- Keep lighting and camera angle as stable as possible.
- Use a baseline clip only if it is a meaningful reference for your use case.
- Use `split_by_speaking=True` only when actual speaking is likely present in the clip.

The speaking split can be useful, but the upstream documentation also notes that it may behave unpredictably when there is no real speaking and mouth movement varies for other reasons.

## Important distinction

This repo also contains [`facial_emotion.py`](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/facial_emotion.py), which is a different idea.

- `facial_expressivity` = facial movement over time
- `emotional_expressivity` = estimated emotion-related outputs from `py-feat`

They are related, but they are not the same measure.

## Repo note

This explanation is based on the OpenWillis documentation and checked against the local implementation in [`face_landmark.py`](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/face_landmark.py).
