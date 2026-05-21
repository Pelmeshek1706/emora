# Head Movement Online vs Offline Feasibility

This note summarizes whether the current AIREST/OpenWillis face stack can estimate head movement features during live recording, after a session completes, or only on a validation subset.

It is based on:

- [`head_movement_feature_inventory.md`](/Users/pelmeshek1706/Desktop/projects/airest-face/head_movement/head_movement_feature_inventory.md)
- [`openwillis-face/src/openwillis/face/head_movement.py`](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/head_movement.py)
- [`openwillis-face/src/openwillis/face/preprocess_video.py`](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/preprocess_video.py)

## Feasibility Table

| Mode | Can the current stack do it? | Practical interpretation |
| --- | --- | --- |
| Live recording / online | No, not natively | The current API is file-based: it opens a saved video with `cv2.VideoCapture`, iterates frame by frame, and returns outputs only after processing finishes. |
| After session completion / offline | Yes | This is the supported mode. Head movement features are computed from a finished video file and summarized into per-frame and aggregate outputs. |
| Validation subset / smoke test | Yes | The stack can intentionally downsample analysis with `frames_per_second`, but that is a sampling choice, not true online inference. Unsampled frames are filled with `NaN`. |

## Technical Constraints

### Latency

- The implementation is not low-latency streaming code. It processes a video file and waits until the file is fully analyzed before returning `out_df` and `summary_df`.
- Runtime is dominated by pose estimation on sampled frames using `py-feat` / `feat.Detector`.
- The local benchmark note shows processing is measured in seconds to minutes for a short 21-second video, depending on the analysis rate and whether bbox preprocessing is used.
- Sparse settings like `frames_per_second=3` are suitable for smoke testing, but they reduce temporal resolution and change the meaning of displacement features.

### Model and Pipeline Dependencies

- Head pose estimation depends on `py-feat` through `feat.Detector()`.
- If face tracks are precomputed, `preprocess_face_video()` adds a DeepFace-based preprocessing path and KMeans clustering before head movement is computed.
- `bbox_list` must match the full video frame count when supplied.
- `frames_per_second` is a sampling parameter, not a real-time clock.

### Robustness Constraints

- `face_confidence` is not directly comparable between modes because supplying `bbox_list` forces confidence to `1`.
- `xy_disp` is sensitive to camera motion, body lean, and bounding-box jitter because it is based on bbox-center displacement.
- `normalize_by_bb_size=True` improves cross-video comparability, but unstable bbox widths can amplify noise.
- Stability is better when bbox tracking is reliable, and worse in small-face, off-center, or multi-person scenes if the crop is wrong.
- The summary features are first-order aggregates only; they do not capture burst structure, rhythm, or temporal synchrony without additional feature engineering.

## Bottom Line

- **Offline after the session:** supported and is the intended usage pattern.
- **Live recording:** not supported out of the box.
- **Validation subset:** supported as a downsampled analysis mode, but it is not a separate streaming capability.

If the goal is true online computation, the current stack would need a new buffered or chunked video-processing wrapper around the existing pose estimator and bbox tracking logic.
