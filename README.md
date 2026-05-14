# airest-voice (OpenWillis Face Demo)

This folder contains a local `openwillis-face` package copy and a runnable demo notebook.

## 1) Create environment

`openwillis-face` requires Python `>=3.9,<3.11`.

This setup uses the available Python `3.10.12` interpreter from:

`/Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/.venv/bin/python`

Run:

```bash
cd /Users/pelmeshek1706/Desktop/projects/airest-voice

/Users/pelmeshek1706/Desktop/projects/final_airest_voice/airest/.venv/bin/python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r openwillis-face/requirements.txt
python -m pip install -e openwillis-face
```

## 2) Pin compatible Torch stack

`py-feat` inside `openwillis-face` needs a torchvision API that still includes `torchvision.io.read_video`.

Run:

```bash
cd /Users/pelmeshek1706/Desktop/projects/airest-voice
source .venv/bin/activate

python -m pip install --force-reinstall torch==2.2.2 torchvision==0.17.2 numpy==1.23.5
```

## 3) (Optional) Install Jupyter kernel

```bash
cd /Users/pelmeshek1706/Desktop/projects/airest-voice
source .venv/bin/activate

python -m pip install jupyter ipykernel
python -m ipykernel install --user --name airest-voice --display-name "Python (airest-voice)"
```

## 4) Demo notebook

Notebook file:

`/Users/pelmeshek1706/Desktop/projects/airest-voice/demo_openwillis_face.ipynb`

Open it with:

```bash
cd /Users/pelmeshek1706/Desktop/projects/airest-voice
source .venv/bin/activate
jupyter notebook demo_openwillis_face.ipynb
```

## 5) Files used by demo

- `filepath='video.mov'` (already present in this folder)
- `baseline_filepath='/Users/pelmeshek1706/Downloads/baseline (1).mp4'`

If `video.mov` is missing, create it from sample data:

```bash
cp sample_data/expressive.mp4 video.mov
```

## 6) Quick CLI smoke test

```bash
cd /Users/pelmeshek1706/Desktop/projects/airest-voice
source .venv/bin/activate

python - <<'PY'
import openwillis.face as owf

framewise_loc, framewise_disp, summary = owf.facial_expressivity(
    filepath='video.mov',
    baseline_filepath='/Users/pelmeshek1706/Downloads/baseline (1).mp4',
    bbox_list=[],
    base_bbox_list=[],
    frames_per_second=10,
    normalize=True,
    align=False,
    rolling_std_seconds=3,
    split_by_speaking=False,
)

print(framewise_loc.shape, framewise_disp.shape, summary.shape)
PY
```

## 7) Function Dependency Diagrams

The package exports five main entry points from `openwillis.face`:

- `facial_expressivity`
- `emotional_expressivity`
- `eye_blink_rate`
- `preprocess_face_video`
- `create_cropped_video`

### `facial_expressivity`

```mermaid
flowchart TD
  FE["facial_expressivity"] --> CFG["get_config"]
  FE --> GL["get_landmarks"]
  GL --> RF["run_facemesh"]
  RF --> CV["cv2.VideoCapture / cvtColor"]
  RF --> IFM["init_facemesh"]
  RF --> PFM["process_and_format_face_mesh"]
  RF --> CPM["crop_and_process_face_mesh"]
  RF --> UDM["get_undected_markers"]
  CPM --> CROP["crop_with_padding_and_center"]
  CPM --> PFM
  PFM --> FPROC["face_mesh.process(...)"]
  PFM --> FC["filter_coord"]
  FC --> GCOL["get_column"]
  FC --> FL["filter_landmarks('x'/'y'/'z')"]

  FE -- "normalize=True" --> NFL["normalize_face_landmarks"]
  NFL --> GV["get_vertices_for_col"]
  NFL --> CL["center_landmarks"]
  NFL --> CRM["calculate_rotation_matrix_for_all_frames"]
  NFL --> AR["apply_rotation_per_frame"]

  FE --> GD["get_displacement"]
  GD --> GED["get_empty_dataframe"]
  GD --> DIST["get_distance"]
  GD -- "baseline exists" --> BASE["baseline"]
  BASE --> GL
  BASE --> NFL
  BASE --> DIST
  GD --> CAD["calculate_areas_displacement"]

  FE --> MO["get_mouth_openness"]
  MO --> LH["get_lip_height"]
  MO --> MH["get_mouth_height"]

  FE -- "split_by_speaking=True" --> SP["get_speaking_probabilities"]
  SP --> FPS["get_fps"]
  SP --> GMM["GaussianMixture"]
  FE -- "split_by_speaking=True" --> SPLIT["split_speaking_df"]
  SPLIT --> SUM["get_summary"]
  FE -- "split_by_speaking=False" --> SUM
```

### `emotional_expressivity`

```mermaid
flowchart TD
  EE["emotional_expressivity"] --> GE["get_emotion"]
  GE --> RPF["run_pyfeat"]
  RPF --> DET["feat.Detector()"]
  RPF --> CV["cv2.VideoCapture"]
  RPF --> DE["detect_emotions"]
  RPF --> CDE["crop_and_detect_emotions"]
  RPF --> UDE["get_undected_emotion"]
  CDE --> CCF["create_cropped_frame"]
  CDE --> DE
  DE --> DF["detector.detect_faces"]
  DE --> DL["detector.detect_landmarks"]
  DE --> DAU["detector.detect_aus"]
  DE --> DEMO["detector.detect_emotions"]
  DE --> MO["mouth_openness"]

  EE --> BASE["baseline"]
  BASE -- "baseline exists" --> GE
  EE -- "baseline exists" --> POST["post-baseline recenter / subtract 1"]

  EE -- "split_by_speaking=True" --> SP["get_speaking_probabilities"]
  SP --> FPS["get_fps"]
  SP --> GMM["GaussianMixture"]
  EE -- "split_by_speaking=True" --> SPLIT["split_speaking_df"]
  SPLIT --> SUM["get_summary"]
  EE -- "split_by_speaking=False" --> SUM
```

### `eye_blink_rate`

```mermaid
flowchart TD
  EBR["eye_blink_rate"] --> CFG["get_config"]
  EBR --> CED["create_empty_dataframes"]
  EBR --> IFM["initialize_facemesh"]
  EBR --> GVC["get_video_capture"]
  EBR --> CFW["calculate_framewise"]
  CFW --> PF["process_frame"]
  PF --> MP["face_mesh.process(...)"]
  CFW --> EAR["eye_aspect_ratio"]
  EBR --> DB["detect_blinks"]
  DB --> FP["scipy.signal.find_peaks"]
  EBR --> CFT["convert_frame_to_time"]
  EBR --> SUM["populate summary dataframe"]
```

### `preprocess_face_video`

```mermaid
flowchart TD
  PFV["preprocess_face_video"] --> CFG["get_config"]
  PFV --> LFV["load_facedata_from_video"]
  LFV --> CV["cv2.VideoCapture"]
  LFV --> EEF["extract_embed_faces_from_frame"]

  EEF --> EF["extract_faces"]
  EF --> ERGB["extract_face_rgb"]
  EF --> EBGR["extract_face_bgr"]
  ERGB --> DFE["DeepFace.extract_faces"]
  EBGR --> DFE
  EF --> D2F["deepface_dict_to_facedata"]

  EEF --> PREP["prep_face_data_for_embed"]
  PREP --> SFFI["FaceData.set_face_from_image"]
  EEF --> EMB["embed_faces"]
  EMB --> DFR["DeepFace.represent"]

  PFV --> CF["cluster_facedata"]
  CF --> FL2D["facedata_list_to_df"]
  FL2D --> TD["FaceData.to_dict"]
  CF --> CE["cluster_embeddings"]
  CE --> KM["KMeans.fit_predict"]

  PFV --> PCO["prep_face_clusters_for_output"]
  PCO --> CSFO["create_single_face_output"]
```

### `create_cropped_video`

```mermaid
flowchart TD
  CCV["create_cropped_video"] --> IO["cv2.VideoCapture / VideoWriter"]
  CCV --> CFF["create_face_frame"]

  CFF -- "bbox present" --> CCF["create_cropped_frame"]
  CFF -- "debug and no bbox" --> ORIG["return original frame"]
  CFF -- "no bbox and trim=False" --> BLACK["return black frame"]
  CFF -- "no bbox and trim=True" --> EMPTY["return empty array"]

  CCF -- "crop=True" --> CPAC["crop_with_padding_and_center"]
  CCF -- "crop=False" --> BOBB["blacken_outside_bounding_box"]

  CPAC --> PAD["calculate_padding"]
  CPAC --> CIMG["crop_img"]
  CPAC --> CIF["center_in_frame"]
  CIF -- "oversized crop" --> RTF["resize_to_fit"]

  CCV --> WRITE["write frame when size != 0"]
```

## 8) Additional Documentation

- [Blink rate notes](/Users/pelmeshek1706/Desktop/projects/airest-voice/blink_rate.md)
- [Facial expression / facial expressivity notes](/Users/pelmeshek1706/Desktop/projects/airest-voice/facial_expression.md)
- [Emotional expressivity notes](/Users/pelmeshek1706/Desktop/projects/airest-voice/emotional_expressivity.md)
