# Emotional Expressivity (`emotional_expressivity`)

This note explains `emotional_expressivity` in a clearer, more practical way for this repo.

Source document:
- https://openwillis.brooklyn.health/Emotional-expressivity-v2-1-15883a8fe04780f987a8cf2bcbd60b8e?pvs=25

## What it measures

`emotional_expressivity` estimates emotion-related facial signals from a video.

It does two kinds of measurement on each analyzed frame:

- emotion scores such as happiness, sadness, anger, fear, disgust, surprise, and neutral
- action unit values, which describe smaller facial muscle patterns

So this function is not only asking "how much did the face move?" It is also asking "what kind of expression does this frame look like?"

## Basic use in this repo

```python
import openwillis.face as owf

framewise, summary = owf.emotional_expressivity(
    filepath="video.mov",
    baseline_filepath="baseline_video.mov",
    bbox_list=[],
    base_bbox_list=[],
    skip_frames=5,
    split_by_speaking=False,
    rolling_std_seconds=3,
)
```

## Important repo note

The upstream OpenWillis page describes this function with a `frames_per_second` argument.

In this local package copy, the actual function signature uses `skip_frames` instead:

- upstream docs: `frames_per_second`
- local repo code: `skip_frames`

That means the local implementation samples one frame, then skips the next `skip_frames` frames before analyzing again.

## What comes back

The function returns two tables:

1. `framewise`
   The frame-by-frame emotion and action-unit output.
2. `summary`
   Mean and standard deviation values for those outputs.

## How it works

### 1. It reads the video frame by frame

The function opens the video and processes selected frames.

In this repo, sampling is controlled by `skip_frames`:

- `skip_frames=5` means it analyzes one frame, then skips five
- lower values give more detail
- higher values run faster but keep less timing detail

Only frames that actually get analyzed are returned in the final framewise output.

### 2. It finds the face

By default, `py-feat` detects the face directly from each processed frame.

If you already have face bounding boxes, you can pass them in with `bbox_list`. In that case, the frame is cropped to the face area first, then analyzed.

### 3. It estimates emotion scores

For each processed frame, `py-feat` returns scores for:

- `happiness`
- `sadness`
- `anger` or `angry` depending on the underlying package version
- `fear`
- `disgust`
- `surprise`
- `neutral`

In the upstream method description, these emotion values are treated as a set of competing emotion judgments for the frame, so they add up to 100.

This makes them easier to read as "which emotion looks most dominant here?"

### 4. It estimates action units

The function also returns facial action units, usually named like:

- `AU_01`
- `AU_02`
- `AU_12`
- `AU_43`

Action units are smaller building blocks of facial behavior, such as brow movement, lip movement, or eye closure.

Unlike the emotion scores, multiple action units can be active at the same time.

In the upstream documentation, these values are described as ranging from `0` to `1`.

### 5. It calculates `mouth_openness`

This function also adds a `mouth_openness` column.

In the local implementation, this is calculated from distances between upper-lip and lower-lip landmarks detected by `py-feat`.

This is later used for the optional speaking split.

### 6. It can compare the video to a baseline

If you provide `baseline_filepath`, the function also analyzes that baseline video and uses it as a reference.

In the local code, baseline correction works like this:

1. take the mean value of each emotion and action-unit column from the baseline video
2. add `1`
3. divide the main video values by those baseline means
4. subtract `1` from the final output

That makes the framewise output easier to interpret relative to the person's own baseline:

- positive values mean higher than baseline
- negative values mean lower than baseline
- `0` means roughly similar to baseline

Without a baseline clip:

- emotion columns stay on their original `0-100` style scale
- action unit columns stay on their original `0-1` style scale

With a baseline clip:

- values become relative-to-baseline values and typically fall in a `-1` to `1` style range

One repo-specific detail matters here:

- the local code computes `summary` before subtracting `1` back out of the `framewise` values
- so `framewise` is centered around zero in the final returned table
- but `summary` is based on the pre-shift normalized values

If you compare `framewise` and `summary` in a baselined run, keep that difference in mind.

### 7. It can split results by speaking vs not speaking

If `split_by_speaking=True`, the function estimates when the person is speaking.

It does this by:

1. measuring how much `mouth_openness` changes over time
2. calculating a rolling standard deviation
3. fitting a 2-group model
4. treating the more variable group as more likely to be speaking

This adds:

- `speaking_probability` to `framewise`

and the summary is split into:

- speaking segments
- non-speaking segments

The upstream documentation warns that this works best when the clip really does contain speaking. If there is no true speaking, the model may still force the video into two groups and label one of them as speaking.

## Inputs

### `filepath`

Path to the main video.

### `baseline_filepath`

Optional path to a baseline video.

### `bbox_list`

Optional list of face bounding boxes for the main video.

### `base_bbox_list`

Optional list of face bounding boxes for the baseline video.

### `skip_frames`

How many frames to skip between analyzed frames in this local implementation.

### `split_by_speaking`

Whether to return separate speaking and non-speaking summaries.

### `rolling_std_seconds`

The window size used when estimating speaking probability.

## Outputs

### 1. `framewise`

A frame-by-frame table that typically includes:

- `frame`
- `time`
- emotion columns such as `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutral`
- depending on package version, the anger column may appear as `anger` or `angry`
- many action unit columns such as `AU_01 ... AU_43`
- `mouth_openness`
- `speaking_probability` when `split_by_speaking=True`

How to read it:

- each row is one analyzed frame
- emotion columns describe likely visible emotional expression in that frame
- action units describe smaller facial movement patterns

### 2. `summary`

A compact table containing:

- mean values for each framewise measure
- standard deviation values for each framewise measure

Typical columns look like:

- `happiness_mean`
- `sadness_mean`
- `anger_mean` or `angry_mean`
- `AU_01_mean`
- `happiness_std`
- `sadness_std`
- `anger_std` or `angry_std`
- `AU_01_std`

When `split_by_speaking=True`, the summary columns are split into:

- `[column]_speaking`
- `[column]_not_speaking`

In the local implementation this usually means names such as:

- `happiness_mean_speaking`
- `happiness_mean_not_speaking`

## How to interpret the results

### Emotion columns

These tell you which visible emotion the model thinks is most strongly expressed in a frame.

- higher `happiness` means the frame looks more happy
- higher `sadness` means the frame looks more sad
- higher `neutral` means the face looks less emotionally expressive

If you use a baseline clip, read these values as change from that person's own baseline rather than as raw scores.

### Action unit columns

These tell you about more specific facial patterns.

They are often useful when:

- you want more detail than broad emotion labels
- you care about specific facial regions or movements
- you want to build your own downstream features later

### `mouth_openness`

This is useful both as a face-behavior measure on its own and as the input to the speaking split.

## Practical advice

- Use clips where the face is clearly visible.
- Good lighting and a stable camera angle improve reliability.
- Pass bounding boxes if you already have them and trust them.
- Use a baseline clip only if it is a meaningful comparison point.
- Use `split_by_speaking=True` only when speaking is genuinely expected in the clip.

If face detection or landmark detection fails on a frame, the local code fills that frame with `NaN` values and later drops missing rows from the final output.

## Difference from `facial_expressivity`

This repo has both:

- [`facial_expression.md`](/Users/pelmeshek1706/Desktop/projects/airest-voice/facial_expression.md), which explains `facial_expressivity`
- this file, which explains `emotional_expressivity`

The difference is:

- `facial_expressivity` focuses on facial movement
- `emotional_expressivity` focuses on emotion-related expression scores and action units

They answer different questions and can be useful together.

## Repo note

This explanation is based on the upstream OpenWillis documentation and checked against the local implementation in [`facial_emotion.py`](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/facial_emotion.py).
