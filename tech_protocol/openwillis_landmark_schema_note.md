# OpenWillis Landmark Output Schema Note

Reviewed on: 2026-05-27

## Purpose

This note records the OpenWillis facial landmark schema for AIREST feature mapping. It is based on the local `airest-face` repository, the referenced upstream OpenWillis face landmark module, and MediaPipe Face Mesh / Face Landmarker documentation.

Primary sources reviewed:

- Local implementation: `openwillis-face/src/openwillis/face/face_landmark.py`
- Local region config: `openwillis-face/src/openwillis/face/config/facial.json`
- Upstream reference module: [OpenWillis `face_landmark.py`](https://raw.githubusercontent.com/bklynhlth/openwillis/main/openwillis-face/src/openwillis/face/face_landmark.py)
- MediaPipe Face Mesh solution docs: [Face Mesh output](https://chuoling.github.io/mediapipe/solutions/face_mesh.html#multi_face_landmarks)
- MediaPipe Face Landmarker task docs: [Face landmark detection guide](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)

## Short Answer

OpenWillis uses MediaPipe Face Mesh landmarks, not the newer MediaPipe Tasks Face Landmarker output schema. The OpenWillis schema is 468 landmarks per face, named `lmk001` through `lmk468`. MediaPipe indexes the same base mesh as `0` through `467`; OpenWillis adds `1` to create the column label.

Each landmark has `x`, `y`, and `z` coordinate columns in the framewise location output. Raw MediaPipe `x` and `y` are normalized by image width and height. Raw `z` is relative depth, where smaller values are closer to the camera and the scale is roughly comparable to `x`. In the default local OpenWillis `facial_expressivity()` path, coordinates are then normalized by centering on `lmk001` and scaling by the 3D distance between `lmk144` and `lmk373`, so exported coordinates are no longer raw `[0, 1]` image-normalized values.

OpenWillis displacement is frame-to-frame Euclidean movement per landmark in the current coordinate space:

```text
displacement_t,i =
  sqrt((x_t,i - x_t-1,i)^2 + (y_t,i - y_t-1,i)^2 + (z_t,i - z_t-1,i)^2)
```

The framewise displacement output includes separate aggregate columns for `overall`, `lower_face`, `upper_face`, `lips`, and `eyebrows`. The local summary output summarizes those aggregate columns separately with mean and standard deviation. It does not summarize all 468 per-landmark displacement columns in the current local code path.

## Landmark Count and Indexing

| Item | Value |
| --- | --- |
| OpenWillis landmark count | 468 |
| OpenWillis labels | `lmk001` through `lmk468` |
| MediaPipe base Face Mesh index | `0` through `467` |
| Mapping rule | `openwillis_lmk_number = mediapipe_index + 1` |
| New MediaPipe Face Landmarker task count | 478 in current task docs, but not used by this OpenWillis module |
| Iris / refined landmarks | Not part of the OpenWillis schema documented here |

The distinction matters: MediaPipe's newer Face Landmarker task docs describe 478 landmarks, while this OpenWillis module uses `mp.solutions.face_mesh.FaceMesh` and fixes `NUM_LANDMARKS = 468`. The local implementation also slices detected landmarks to the first 468.

## Framewise Location Output

Expected dataframe role: `framewise_loc`, returned as the first object from `facial_expressivity()`.

| Column group | Columns | Count | Meaning |
| --- | --- | ---: | --- |
| Frame metadata | `frame`, `time` | 2 | `frame` is a zero-based decoded video frame counter. `time` is computed as `frame / fps` from OpenCV-reported FPS. |
| X coordinates | `lmk001_x` through `lmk468_x` | 468 | Landmark x-coordinate. Raw MediaPipe value is normalized by image width. |
| Y coordinates | `lmk001_y` through `lmk468_y` | 468 | Landmark y-coordinate. Raw MediaPipe value is normalized by image height. |
| Z coordinates | `lmk001_z` through `lmk468_z` | 468 | Landmark relative depth. Smaller values are closer to camera in MediaPipe convention. |

Total normal raw-location column count: `2 + (468 * 3) = 1406`.

Implementation details to preserve in mapping:

- `run_facemesh()` / `get_landmarks()` process every decoded frame in the current local repo and compute `time = frame / fps`.
- The referenced upstream module includes a `frames_per_second` sampling path; the current local module treats `frames_per_second` as a compatibility argument in `facial_expressivity()` and uses native frame metadata instead.
- Missing or failed face detection produces the `frame` and `time` values with all 1,404 coordinate fields set to `NaN`.
- The default `facial_expressivity(normalize=True)` path returns normalized landmarks. In the current local implementation, `normalize_face_landmarks()` appends `frame` and `time` after the coordinate columns, so downstream mapping should select by column name rather than positional order.

## Coordinate Spaces

| Mode | How produced | Coordinate interpretation |
| --- | --- | --- |
| Raw MediaPipe Face Mesh | `get_landmarks()` before normalization | `x` and `y` are image-normalized; `z` is relative depth with roughly `x` scale. |
| OpenWillis normalized | `facial_expressivity(normalize=True)` default | Coordinates are centered by subtracting `lmk001` and scaled by the 3D eye distance between `lmk144` and `lmk373`. |
| OpenWillis aligned | `align=True` | After centering, coordinates are rotated around the z-axis so the eye line is horizontal, then scaled by eye distance. Top-level default is `align=False`. |

For AIREST schema mapping, store a coordinate-space flag such as `raw_mediapipe`, `openwillis_centered_eye_scaled`, or `openwillis_centered_aligned_eye_scaled`. Do not treat the normalized OpenWillis coordinates as raw MediaPipe `[0, 1]` image coordinates.

## Framewise Displacement Output

Expected dataframe role: `framewise_disp`, returned as the second object from `facial_expressivity()`.

| Column group | Columns | Count | Meaning |
| --- | --- | ---: | --- |
| Frame metadata | `frame`, `time` | 2 | Copied from the landmark dataframe. |
| Per-landmark displacement | `lmk001` through `lmk468` | 468 | Euclidean frame-to-frame movement for each landmark. |
| Overall displacement | `overall` | 1 | Per-frame mean of the 468 landmark displacement columns. |
| Region displacement | `lower_face`, `upper_face`, `lips`, `eyebrows` | 4 | Per-frame mean of configured landmark displacement subsets. |
| Mouth openness | `mouth_openness` | 1 | Ratio derived from lip distances; not a displacement measure. |
| Speaking probability | `speaking_probability` | 0 or 1 | Present only when `split_by_speaking=True`. |

Normal column count without speaking split: `2 + 468 + 1 + 4 + 1 = 476`.

Displacement details:

- `get_distance()` uses `.shift()`, so the first row is `NaN` in the current local implementation, despite the public docstring saying first-row displacement is always zero.
- Displacement is not rate-normalized by elapsed time or FPS.
- If `normalize=True`, displacement is calculated on OpenWillis normalized coordinates.
- If a baseline file exists, OpenWillis computes mean baseline displacement per landmark and applies `(actual + 1) / (baseline + 1) - 1`.
- Current local caveat: `baseline()` computes a normalized baseline dataframe but then derives baseline displacement from `base_landmark`, not the normalized variable. Treat baseline-corrected values as implementation-specific until this is fixed or explicitly versioned.

## Region Summaries

OpenWillis creates separate framewise aggregate displacement columns and separately summarizes them in the returned summary dataframe.

Region provenance:

- Numbered landmark identity comes from the MediaPipe Face Mesh ordered landmark array. Official MediaPipe code records `FACEMESH_NUM_LANDMARKS = 468` and `FACEMESH_NUM_LANDMARKS_WITH_IRISES = 478` in [`face_mesh.py`](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/python/solutions/face_mesh.py#L55-L56).
- A numbered landmark map is available as [`mesh_map.jpg`](https://raw.githubusercontent.com/tensorflow/tfjs-models/master/face-landmarks-detection/mesh_map.jpg) in TensorFlow's `tfjs-models` repository. This is useful for visually locating MediaPipe dot IDs, but it does not define OpenWillis' upper/lower/lips/eyebrows groupings.
- OpenWillis region lists for `lower_face_landmarks`, `lips_landmarks`, `eyebrows_landmarks`, `upper_lip_simple_landmarks`, and `lower_lip_simple_landmarks` come from [`facial.json`](../openwillis-face/src/openwillis/face/config/facial.json) in this repo and the upstream OpenWillis [`facial.json`](https://raw.githubusercontent.com/bklynhlth/openwillis/main/openwillis-face/src/openwillis/face/config/facial.json).
- `upper_face` is not a separate source list. OpenWillis derives it in `calculate_areas_displacement()` as every MediaPipe index from `0` to `467` that is not in `lower_face_landmarks`; see upstream [`face_landmark.py`](https://github.com/bklynhlth/openwillis/blob/main/openwillis-face/src/openwillis/face/face_landmark.py#L704-L717).

| Output | Source indices | Unique dots | Summary behavior | Notes |
| --- | --- | ---: | --- | --- |
| `overall` | All MediaPipe indices `0..467` | 468 | Yes: `overall_mean`, `overall_std` | Mean of all per-landmark displacement columns. |
| `lower_face` | `lower_face_landmarks` in `facial.json` | 169 unique | Yes: `lower_face_mean`, `lower_face_std` | Config has 170 entries because MediaPipe index `140` appears twice; current mean weights `lmk141` twice. |
| `upper_face` | Complement of `lower_face_landmarks` | 299 | Yes: `upper_face_mean`, `upper_face_std` | Includes eyebrows. |
| `lips` | `lips_landmarks` in `facial.json` | 80 | Yes: `lips_mean`, `lips_std` | Subset of lower face. |
| `eyebrows` | `eyebrows_landmarks` in `facial.json` | 28 | Yes: `eyebrows_mean`, `eyebrows_std` | Subset of upper face. |
| `mouth_openness` | Upper/lower lip simple landmarks | N/A | Yes: `mouth_openness_mean`, `mouth_openness_std` | Ratio feature, not displacement. |

The local `get_summary(df_disp, 470)` call starts summary calculations at the `overall` column. Therefore the current summary dataframe includes aggregate means/stds, not all `lmk001_mean` through `lmk468_std` per-landmark summaries. If `split_by_speaking=True`, the same aggregate summaries are split into speaking and not-speaking columns.

## Visual Region Mapping

Generated artifacts live in `tech_protocol/assets/openwillis_landmarks/`.

Source links for the numbered views:

- The OpenWillis region membership used by the figures is generated from this repo's [`facial.json`](../openwillis-face/src/openwillis/face/config/facial.json) and the same derivation logic used in [`calculate_areas_displacement()`](../openwillis-face/src/openwillis/face/face_landmark.py).
- The external source that shows MediaPipe dots with their numeric IDs is TensorFlow's [`mesh_map.jpg`](https://raw.githubusercontent.com/tensorflow/tfjs-models/master/face-landmarks-detection/mesh_map.jpg). There is no separate OpenWillis source image naming every upper-face dot; the upper-face visualization is computed from the OpenWillis lower-face complement.

| View | PNG | SVG |
| --- | --- | --- |
| Overall, all 468 numbered landmarks | [PNG](assets/openwillis_landmarks/overall_all_landmarks_numbered.png) | [SVG](assets/openwillis_landmarks/overall_all_landmarks_numbered.svg) |
| Region overlay with exclusive display colors | [PNG](assets/openwillis_landmarks/openwillis_region_overlay_numbered.png) | [SVG](assets/openwillis_landmarks/openwillis_region_overlay_numbered.svg) |
| Upper face numbered landmarks | [PNG](assets/openwillis_landmarks/upper_face_landmarks_numbered.png) | [SVG](assets/openwillis_landmarks/upper_face_landmarks_numbered.svg) |
| Lower face numbered landmarks | [PNG](assets/openwillis_landmarks/lower_face_landmarks_numbered.png) | [SVG](assets/openwillis_landmarks/lower_face_landmarks_numbered.svg) |
| Lips numbered landmarks | [PNG](assets/openwillis_landmarks/lips_landmarks_numbered.png) | [SVG](assets/openwillis_landmarks/lips_landmarks_numbered.svg) |
| Eyebrows numbered landmarks | [PNG](assets/openwillis_landmarks/eyebrows_landmarks_numbered.png) | [SVG](assets/openwillis_landmarks/eyebrows_landmarks_numbered.svg) |
| Region membership table | [CSV](assets/openwillis_landmarks/openwillis_landmark_region_membership.csv) | N/A |

The overlay uses exclusive colors for readability: lips override lower face, eyebrows override upper face, and the remaining points are shown as upper or lower face. This is only a visualization choice. In OpenWillis feature computation, `lips` remain a subset of `lower_face`, and `eyebrows` remain a subset of `upper_face`.

To regenerate the figures:

```bash
python3 tech_protocol/scripts/visualize_openwillis_landmarks.py
```

## Mapping Recommendations

For AIREST exports, store landmark fields by name and include metadata for:

- OpenWillis package/source version and MediaPipe version.
- Whether coordinates are raw or OpenWillis-normalized.
- `normalize`, `align`, `baseline_filepath`, and whether baseline correction was applied.
- Video FPS source, `frame` counter semantics, and missing-landmark rate.
- Region-index version derived from `facial.json`.
- Whether summary values are overall/region-only or include per-landmark summaries.
