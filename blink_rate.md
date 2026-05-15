# Blink Feature Specification Notes

Reference implementation reviewed:
- OpenWillis `eye_blink.py`: [raw source](https://raw.githubusercontent.com/bklynhlth/openwillis/main/openwillis-face/src/openwillis/face/eye_blink.py)
- Local copy: [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/eye_blink.py)
- Config: [eye.json](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/config/eye.json)

## Scope

OpenWillis exposes blink features through `eye_blink_rate(video)`, which returns:

```python
ear, blinks, summary = owf.eye_blink_rate(video="video.mov")
```

The function produces three outputs only:
- framewise EAR signal
- blink event table
- one-row summary with blink count and blink rate

## What EAR means

EAR stands for `eye aspect ratio`.

It is a compact numeric measure of how open the eye is in a given frame:
- higher EAR generally means the eye is more open
- lower EAR generally means the eye is more closed
- a blink appears as a short dip in EAR

In OpenWillis, EAR is computed from six eye landmarks using the standard formula from the Soukupová and Čech blink paper:

```python
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Where:
- `p1` and `p4` are the horizontal eye corner points
- `p2`, `p3`, `p5`, and `p6` are the vertical eyelid points

The actual implementation is in [eye_aspect_ratio() in `eye_blink.py`](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/eye_blink.py:83).
That function computes:
- `A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])`
- `B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])`
- `C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])`
- `ear = (A + B) / (2.0 * C)`

OpenWillis then averages left-eye and right-eye EAR for each frame and z-score normalizes the resulting series across the clip.

## How OpenWillis computes blink features

Implementation details confirmed from [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/eye_blink.py:203):

1. MediaPipe FaceMesh is run frame by frame.
2. Six landmarks are used per eye.
3. Left-eye and right-eye EAR are computed, then averaged into one EAR value per frame.
4. The framewise EAR series is z-score normalized across the video.
5. Blinks are detected as troughs in the normalized EAR signal using `scipy.signal.find_peaks` on `-EAR`.
6. Blink start and end are taken from `find_peaks` `left_ips` and `right_ips`, rounded to integer frame indices and shifted to 1-based frame numbering.

Configured detection parameters from [eye.json](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/config/eye.json):
- `prominence = 2`
- `width = 0.01`

Eye landmark sets:
- left eye: `362, 385, 387, 263, 373, 380`
- right eye: `33, 160, 158, 133, 153, 144`

## Available output: framewise EAR

Output object: `ear`

Columns:
- `frame`
- `ear`

Meaning:
- `frame` is the 1-based video frame index.
- `ear` is not raw EAR. It is the video-level z-scored average EAR after combining left and right eyes.

Important interpretation note for AIREST:
- This signal is suitable for blink detection and framewise eye-opening dynamics.
- It is not a calibrated physical eyelid-opening measure.
- Because it is z-scored within each video, values are relative to that recording, not directly comparable as absolute EAR across videos without additional normalization policy.

Implementation reference:
- framewise EAR calculation and z-scoring occur in [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/eye_blink.py:300)

## Available output: blink event table

Output object: `blinks`

Columns:
- `blink_peak_frame`
- `blink_starting_frame`
- `blink_ending_frame`
- `blink_peak_time`
- `blink_starting_time`
- `blink_ending_time`

Meaning:
- `blink_peak_frame`: frame of maximum eye closure, implemented as the EAR trough.
- `blink_starting_frame`: estimated blink onset from `left_ips`.
- `blink_ending_frame`: estimated blink offset from `right_ips`.
- `blink_peak_time`, `blink_starting_time`, `blink_ending_time`: those same frame indices divided by video FPS.

Timing interpretation:
- OpenWillis does provide event timing for start, peak, and end.
- These timings are derived from the trough and peak-width boundaries in the normalized EAR trace, not from a separate state-machine or threshold-crossing model.

Implementation reference:
- blink detection in [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/eye_blink.py:203)
- frame-to-time conversion in [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/eye_blink.py:249)

## Available output: blink count

Output object: `summary`

Column:
- `blinks`

Meaning:
- Total number of detected blink events in the clip.
- Implemented as `len(troughs)`.

Implementation reference:
- summary creation in [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/eye_blink.py:417)

## Available output: blink rate

Output object: `summary`

Column:
- `blink_rate`

Meaning:
- Blinks per minute over the analyzed clip.

Formula used by OpenWillis:

```python
blink_rate = len(troughs) / (frame_n / fps) * 60
```

Notes:
- Denominator is total processed clip duration based on frame count and FPS.
- This is a clip-level aggregate only. No windowed or rolling blink-rate output is provided.

Implementation reference:
- [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/eye_blink.py:417)

## Not currently provided by OpenWillis

The reviewed module does not directly output:
- inter-blink interval (IBI)
- blink duration as a separate explicit feature column
- closing duration or opening duration
- dwell/closed-eye hold time
- rolling blink rate by time window
- per-eye EAR outputs
- raw, non-z-scored EAR output

Some of these can be derived from the current event table:
- Blink duration can be computed as `blink_ending_time - blink_starting_time`.
- Time-to-peak from onset can be computed as `blink_peak_time - blink_starting_time`.
- Reopening time can be computed as `blink_ending_time - blink_peak_time`.

## AIREST recommendation on inter-blink intervals

Inter-blink intervals should be added separately for AIREST.

Reason:
- OpenWillis does not emit IBI directly.
- The current outputs are sufficient to derive IBI downstream without modifying the detector.

Recommended AIREST derivations:
- Peak-to-peak IBI: difference between consecutive `blink_peak_time` values.
- End-to-start IBI: next `blink_starting_time - previous blink_ending_time`.

Preferred default for reporting:
- Use peak-to-peak IBI as the primary interval metric because peak timing is the most direct event anchor produced by OpenWillis.
- Optionally retain end-to-start gap as a secondary measure if AIREST wants a stricter "time with eyes open between blinks" feature.

## Practical spec summary

OpenWillis currently gives AIREST:
- Framewise normalized EAR trace: yes
- Blink count: yes
- Blink rate in blinks/minute: yes
- Blink event timing with start, peak, end in both frames and seconds: yes
- Inter-blink intervals: no, derive separately in AIREST
