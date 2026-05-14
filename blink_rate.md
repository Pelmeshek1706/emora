# Blink Rate (`eye_blink_rate`)

This note explains what the OpenWillis blink-rate function does, what it returns, and how to read the results in this repo.

Source document:
- https://openwillis.brooklyn.health/Eye-blink-rate-v1-0-15883a8fe047809f82fcef89964973e3

## What it measures

`eye_blink_rate` estimates how often a person blinks in a video.

It does this by tracking how open the eyes are in each frame. When the eye-opening signal drops and rises again in the pattern of a blink, the function counts that as one blink.

## Basic use

```python
import openwillis.face as owf

ear, blinks, summary = owf.eye_blink_rate(video="video.mov")
```

## How it works

1. The video is opened frame by frame.
2. MediaPipe FaceMesh finds facial landmarks in each frame.
3. The function takes 6 landmarks around the left eye and 6 around the right eye.
4. It calculates the eye aspect ratio, or `EAR`, for each eye.
5. It averages the left-eye and right-eye EAR values into one value per frame.
6. It z-score normalizes the EAR signal across the video so the method is less sensitive to camera distance and face size.
7. It looks for clear downward dips in that signal. Those dips are treated as blinks.
8. For each detected blink, it records:
   - the frame where the blink reaches its lowest point
   - the frame where the blink starts
   - the frame where the blink ends
9. It converts those frame numbers into seconds using the video FPS.
10. It returns frame-level data, blink-level data, and a short summary.

## The key idea: EAR

EAR is a simple number that describes how open the eye is.

- Higher EAR usually means the eye is more open.
- Lower EAR usually means the eye is more closed.
- A blink shows up as a short drop in EAR.

In this local package copy, the blink detector uses:

- left eye landmarks: `362, 385, 387, 263, 373, 380`
- right eye landmarks: `33, 160, 158, 133, 153, 144`
- peak detection settings from [`eye.json`](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/config/eye.json): `prominence=2`, `width=0.01`

## Outputs

### 1. `ear`

A frame-by-frame table with:

- `frame`
- `ear`

This is the eye-opening signal used for blink detection.

### 2. `blinks`

One row per detected blink with:

- `blink_peak_frame`
- `blink_starting_frame`
- `blink_ending_frame`
- `blink_peak_time`
- `blink_starting_time`
- `blink_ending_time`

This table tells you when each blink happened.

### 3. `summary`

A one-row summary with:

- `blinks`: total blink count
- `blink_rate`: blinks per minute

## How to read the results

- If `blink_rate` is higher, the person blinked more often in that clip.
- Each row in `blinks` is one detected blink event.
- The `ear` table is useful if you want to inspect the raw signal or debug missed detections.

## Things that matter for quality

- The face should be visible for most of the clip.
- The eye area should not be heavily blocked by hands, hair, blur, or extreme head pose.
- Lighting should be good enough for stable landmark detection.
- The code checks for common video file types such as `.mp4`, `.avi`, and `.mov`.

If the face cannot be tracked well enough, the outputs may contain `NaN` values.

## Repo note

This explanation follows the OpenWillis documentation, but it also reflects what the local implementation in [`eye_blink.py`](/Users/pelmeshek1706/Desktop/projects/airest-voice/openwillis-face/src/openwillis/face/eye_blink.py) actually does.
