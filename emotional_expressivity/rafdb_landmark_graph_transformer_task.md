# Task Description: RAF-DB Landmark-Based Facial Expression Recognition with GCN / Graph Transformer

## 1. Objective

Build a security-compatible facial expression classification pipeline that uses **only normalized MediaPipe facial landmarks** as model input. The current implementation must focus on **GCN / graph-based architectures**, with the primary target model being a **Graph Transformer / Graphormer-lite** architecture.

The model must classify facial expressions into RAF-DB basic emotion categories, aligned with common FER2013-style outputs:

```text
anger, disgust, fear, happiness, sadness, surprise, neutral
```

The final production-oriented pipeline should support the following flow:

```text
480x480 image/frame
→ MediaPipe Face Landmarker
→ 478 3D facial landmarks
→ landmark normalization
→ graph construction
→ Graph Transformer / GCN classifier
→ 7-class emotion probabilities
```

The system must not store or train directly on raw face images after landmark extraction. Images are used only as temporary inputs to the landmark extraction stage.

---

## 2. Background and Source Grounding

### 2.1 MediaPipe Face Landmarker

MediaPipe Face Landmarker is used as the extraction component. The official Google AI Edge documentation describes the task as detecting face landmarks and facial expressions in images and videos. It can output facial landmarks, blendshape scores, and facial transformation matrices. The Face Landmarker model bundle includes a face landmarks detection model that predicts **478 3D face landmarks**, and a blendshape prediction model that predicts **52 blendshape scores**.

Source: https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker

### 2.2 RAF-DB Dataset

RAF-DB is the selected dataset for the current implementation. The official RAF-DB page describes it as a large-scale facial expression database with around **30K diverse facial images** downloaded from the Internet. It contains basic and compound expression annotations. For this implementation, use the **basic expression subset**.

Source: https://www.whdeng.cn/RAF/model1.html

### 2.3 Graph-Based Facial Landmark FER

Graph-based facial expression recognition is appropriate because facial landmarks naturally form a graph: landmarks are nodes, and facial mesh relations are edges. Existing landmark-based FER research has used graph neural networks for this reason. For example, the DGNN paper proposes a directed graph neural network using facial landmark features for FER.

Source: https://www.mdpi.com/2079-9292/9/5/764

### 2.4 Graph Transformer / Graphormer Inspiration

Graphormer shows that Transformers can be adapted to graph data by adding graph structural encodings such as centrality encoding, spatial encoding, and edge encoding. This is directly relevant to facial landmark graphs: degree/region centrality, shortest-path distance, and edge type can be used as structural biases in attention.

Source: https://arxiv.org/pdf/2106.05234

---

## 3. Scope

### 3.1 In Scope

- RAF-DB basic expression dataset preparation.
- MediaPipe landmark extraction from RAF-DB images.
- Landmark-only storage format.
- Landmark normalization pipeline.
- Facial mesh graph construction.
- Training/evaluation of graph-based models:
  - simple GCN baseline;
  - GAT baseline;
  - Graph Transformer / Graphormer-lite primary model.
- Experiment tracking and reproducible training configs.
- Metrics, confusion matrix, per-class analysis, and production-readiness report.
- Inference module for 480x480 production-style image/frame input.

### 3.2 Out of Scope for Current Implementation

- Raw RGB image CNN training.
- SigLIP2 image fine-tuning.
- Video temporal modeling.
- Multimodal fusion with audio/gaze/speech.
- Clinical diagnosis prediction.
- Storing original face images after landmark extraction.
- Production API/server deployment.

### 3.3 Optional / Future Extensions

- SigLIP2 text-prototype alignment for emotion labels.
- Spatio-temporal GCN / Transformer for video sequences.
- Distillation from image-based teacher models into a landmark-only student.
- AffectNet and FER2013 experiments for cross-dataset validation.
- MediaPipe blendshape feature fusion, if allowed by project security policy.

---

## 4. Security and Privacy Constraints

The implementation must satisfy a landmark-only security design.

### Required Behavior

1. Raw image files may be read only during preprocessing.
2. MediaPipe must extract facial landmarks locally.
3. After extraction, only normalized landmarks, label, and non-identifying metadata should be stored.
4. Do not store raw images, cropped faces, rendered landmark images, or face textures in the processed training dataset.
5. Do not train models directly on raw image pixels.
6. Treat landmark coordinates as biometric-derived data and store them securely.

### Allowed Stored Data

```json
{
  "sample_id": "train_000001",
  "dataset": "RAF-DB",
  "split": "train",
  "label": "happy",
  "label_id": 3,
  "landmarks_norm": [[0.12, -0.31, 0.02], [0.11, -0.29, 0.02]],
  "quality": {
    "landmark_success": true,
    "face_detection_confidence": 0.98,
    "face_presence_confidence": 0.97
  }
}
```

### Forbidden Stored Data

```text
raw image
face crop
thumbnail
rendered face
rendered landmark image if it can visually reconstruct the face
full video frame
```

---

## 5. Dataset Requirements

### 5.1 Dataset

Use **RAF-DB basic expression subset**.

Expected classes:

```text
surprise, fear, disgust, happiness, sadness, anger, neutral
```

Map labels to a stable internal order:

```python
LABEL_MAP = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "happiness": 3,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
}
```

If RAF-DB annotation files use a different numeric order, implement an explicit mapping and document it in `configs/label_map.yaml`.

### 5.2 Dataset Split

Use official RAF-DB train/test split if available in the local dataset package. Do not create a random split unless the official split is missing.

Required split metadata:

```text
sample_id
original_file_name
split: train/test/val
label_name
label_id
landmark_success
```

If RAF-DB does not provide a validation split, create a stratified validation split from training data:

```text
train: 85–90% of official train
val: 10–15% of official train
final test: official RAF-DB test only
```

### 5.3 Dataset Quality Report

Before model training, generate a dataset quality report:

- total images;
- total successfully landmarked samples;
- landmark extraction success rate;
- success rate per class;
- class distribution before extraction;
- class distribution after failed extraction removal;
- number of images with multiple faces;
- number of skipped samples;
- mean/median confidence scores.

Output file:

```text
reports/data_quality/rafdb_landmark_extraction_report.md
```

---

## 6. Landmark Extraction Pipeline

### 6.1 Input

RAF-DB image file.

### 6.2 Processing Steps

```text
1. Load image.
2. Resize/prepare image for MediaPipe if needed.
3. Run MediaPipe Face Landmarker in image mode.
4. Validate exactly one primary face or select highest-confidence face.
5. Extract 478 landmarks: x, y, z.
6. Extract confidence/quality metadata where available.
7. Normalize landmarks.
8. Store normalized landmark data and label.
9. Do not store the image.
```

### 6.3 Failure Handling

If no face is detected:

```json
{
  "sample_id": "train_000123",
  "landmark_success": false,
  "failure_reason": "no_face_detected"
}
```

If multiple faces are detected:

- choose the highest-confidence face if it is clearly dominant;
- otherwise mark sample as ambiguous and skip it.

Possible failure reasons:

```text
no_face_detected
multiple_faces_ambiguous
low_detection_confidence
invalid_landmark_shape
normalization_failed
corrupt_image
```

---

## 7. Landmark Normalization

Raw MediaPipe coordinates must not be used directly. Normalize before graph construction.

### 7.1 Goals

Normalization should reduce variance caused by:

- face position in image;
- face scale;
- image crop differences;
- in-plane rotation;
- partial head pose effects.

### 7.2 Required Normalization Procedure

Implement at least this normalization:

```text
1. Center landmarks by face center or selected stable anchor point.
2. Scale by inter-ocular distance or face bounding-box diagonal.
3. Rotate-align the eye line horizontally.
4. Normalize z by the same scale factor as x/y.
5. Store normalized landmarks as float32.
```

Recommended formula:

```python
center = mean(selected_stable_landmarks)
scale = distance(left_eye_center, right_eye_center)

landmarks_centered = landmarks - center
landmarks_scaled = landmarks_centered / scale
landmarks_aligned = rotate_to_horizontal_eye_axis(landmarks_scaled)
```

### 7.3 Required Validation

After normalization, validate:

- no NaN / Inf values;
- expected shape is `[478, 3]`;
- coordinate range is plausible;
- scale is non-zero;
- left/right eye anchors are valid.

### 7.4 Optional Derived Features

Add optional geometric features as node or graph-level features:

```text
mouth opening
mouth width
eye openness
eyebrow-eye distance
jaw opening
lip corner displacement
left-right asymmetry
face aspect ratio
```

These should not replace normalized landmarks in the graph model. They can be concatenated to the pooled graph embedding before the classifier head.

---

## 8. Graph Construction

### 8.1 Node Definition

Each MediaPipe landmark is one graph node.

```text
N = 478 nodes
```

Node feature baseline:

```text
[x_norm, y_norm, z_norm]
```

Recommended node feature vector:

```text
[x_norm, y_norm, z_norm, region_id, landmark_id]
```

Implementation detail:

- `x_norm, y_norm, z_norm` go through a coordinate MLP.
- `landmark_id` should be represented as learned embedding.
- `region_id` should be represented as learned embedding.

### 8.2 Region IDs

Create region labels for landmarks:

```text
left_eye
right_eye
left_eyebrow
right_eyebrow
nose
upper_lip
lower_lip
mouth_corner
jaw
cheek
face_oval
other
```

The engineer must create `configs/mediapipe_regions.yaml` containing landmark index groups. If exact grouping is uncertain, start with a conservative grouping and document all assumptions.

### 8.3 Edge Definition

Use MediaPipe face mesh topology as the primary edge set.

Graph edge types:

```text
mesh_edge
same_region_edge
symmetry_edge
optional_knn_edge
```

Minimum graph:

```text
nodes = 478 landmarks
edges = MediaPipe facial mesh connections
```

Recommended graph:

```text
edges = mesh_edges
      + same_region_dense_edges for important regions
      + symmetry_edges between left/right corresponding regions
```

### 8.4 Edge Attributes

For every edge `(i, j)`, compute:

```text
edge_type_id
Euclidean distance between normalized landmarks i and j
same_region flag
left_right_symmetric flag
```

For Graph Transformer attention bias, additionally precompute:

```text
shortest_path_distance(i, j)
spatial_distance_bucket(i, j)
region_relation(i, j)
```

---

## 9. Model Architectures

The implementation must include three graph model levels:

1. **GCN baseline**
2. **GAT baseline**
3. **Graph Transformer / Graphormer-lite primary model**

---

## 10. Model 1: GCN Baseline

### Purpose

Establish a simple topology-aware baseline.

### Architecture

```text
Input graph:
  478 nodes
  node features: [x_norm, y_norm, z_norm]

GCN:
  GCNConv(input_dim, 128)
  ReLU
  Dropout(0.2)
  GCNConv(128, 128)
  ReLU
  Dropout(0.2)
  GCNConv(128, 256)
  ReLU

Pooling:
  global mean pooling
  global max pooling
  concat

Classifier:
  Linear(512, 256)
  ReLU
  Dropout(0.3)
  Linear(256, 7)
```

### Expected Use

Fast baseline, not expected to be final model.

---

## 11. Model 2: GAT Baseline

### Purpose

Test whether attention over neighboring landmarks improves over standard GCN.

### Architecture

```text
Input graph:
  478 nodes
  node features: coordinate embedding + landmark_id_embedding + region_embedding

GAT:
  GATConv(d_model, 128, heads=4)
  ELU
  Dropout(0.2)
  GATConv(512, 128, heads=4)
  ELU
  Dropout(0.2)
  GATConv(512, 256, heads=2)

Pooling:
  attention pooling or mean+max pooling

Classifier:
  MLP → 7 classes
```

### Expected Use

Intermediate baseline between GCN and Graph Transformer.

---

## 12. Model 3: Graph Transformer / Graphormer-Lite

### Purpose

Primary architecture for this implementation.

The model should combine:

- landmark coordinate tokens;
- landmark identity embeddings;
- face-region embeddings;
- graph structural attention bias;
- global self-attention over all facial landmarks.

### Core Idea

```text
normalized landmark graph
→ node token embeddings
→ Graph Transformer with structural attention bias
→ pooled graph embedding
→ 7-class classifier
```

### Input

```text
landmarks_norm: [478, 3]
edge_index: [2, num_edges]
edge_attr: [num_edges, edge_attr_dim]
shortest_path_matrix: [478, 478]
region_relation_matrix: [478, 478]
```

### Token Embedding

```python
coord_emb = CoordMLP(xyz)                 # [N, d_model]
landmark_emb = LandmarkIDEmbedding(ids)   # [N, d_model]
region_emb = RegionEmbedding(region_ids)  # [N, d_model]
node_tokens = coord_emb + landmark_emb + region_emb
```

Recommended defaults:

```yaml
d_model: 192
num_layers: 4
num_heads: 6
dropout: 0.2
ffn_dim: 384
num_classes: 7
```

### Structural Attention Bias

For attention score between node `i` and node `j`:

```text
attention_score(i, j) = Q_i K_j^T / sqrt(d)
                      + spatial_bias[shortest_path_distance(i, j)]
                      + region_bias[region_relation(i, j)]
                      + edge_bias[edge_type(i, j)]
```

Required encodings:

1. **Shortest-path spatial bias**
   - Use graph shortest path distance buckets.
   - Clamp distances above a max value.

2. **Region relation bias**
   - same region;
   - symmetric region;
   - cross region;
   - unrelated.

3. **Centrality/degree encoding**
   - Add learned degree embeddings to node tokens.

Graphormer uses centrality, spatial, and edge encodings to inject graph structure into a Transformer. This implementation should adapt those principles to the facial landmark graph.

### Pooling

Implement two options:

```text
Option A: CLS token pooling
Option B: mean pooling + max pooling concat
```

Default:

```text
CLS token + mean pooling concat
```

### Classifier Head

```text
pooled_embedding
→ LayerNorm
→ Linear(hidden, 256)
→ GELU
→ Dropout(0.3)
→ Linear(256, 7)
```

### Regularization

Use:

```text
dropout: 0.2–0.4
weight_decay: 0.01
label_smoothing: 0.05
coordinate noise augmentation
random landmark masking
region dropout
```

---

## 13. Training Strategy

### 13.1 Loss Functions

Required baseline loss:

```text
CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
```

Optional experiments:

```text
FocalLoss(gamma=1.5 or 2.0)
Class-balanced loss
Supervised contrastive loss on graph embedding
```

Recommended combined loss for Graph Transformer experiment:

```text
L = CE(logits, label) + 0.1 * SupCon(embedding, label)
```

Only add supervised contrastive loss after the CE-only model works.

### 13.2 Optimizer

Default:

```yaml
optimizer: AdamW
learning_rate: 0.0003
weight_decay: 0.01
scheduler: cosine_with_warmup
warmup_epochs: 3
max_epochs: 80
early_stopping_patience: 12
batch_size: 32
```

### 13.3 Data Augmentation on Landmarks

Use landmark-space augmentation only:

```text
small Gaussian coordinate noise
small scale jitter
small rotation jitter
random landmark dropout
random region dropout
z-coordinate noise
```

Do not use image augmentations after landmark extraction because model input is landmarks only.

Suggested defaults:

```yaml
coord_noise_std: 0.005
scale_jitter: 0.03
rotation_jitter_degrees: 5
landmark_dropout_prob: 0.02
region_dropout_prob: 0.05
```

### 13.4 Class Imbalance

Compute class weights from the training split after landmark extraction.

Required outputs:

```text
class_counts_before_extraction.json
class_counts_after_extraction.json
class_weights.json
```

---

## 14. Evaluation Requirements

Do not report accuracy only.

Required metrics:

```text
accuracy
macro-F1
weighted-F1
balanced accuracy
per-class precision
per-class recall
per-class F1
confusion matrix
ROC-AUC optional if implemented one-vs-rest
ECE/calibration optional
```

Required reports:

```text
reports/evaluation/gcn_baseline.md
reports/evaluation/gat_baseline.md
reports/evaluation/graphormer_lite.md
reports/evaluation/model_comparison.md
```

### 14.1 Minimum Comparison Table

The final comparison report must include:

| Model | Accuracy | Macro-F1 | Balanced Accuracy | Anger F1 | Disgust F1 | Fear F1 | Happy F1 | Sad F1 | Surprise F1 | Neutral F1 | Latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GCN | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GAT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Graphormer-lite | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 14.2 Production-Focused Validation

Because production input is expected to be **480x480**, evaluate separately on a production-style validation set if available.

If no internal 480x480 validation set exists, create a small controlled validation protocol:

```text
1. Sample RAF-DB images.
2. Run MediaPipe at original resolution.
3. Run MediaPipe after resizing to 480x480.
4. Compare landmark stability and model prediction stability.
```

Report:

```text
landmark coordinate drift
prediction agreement
confidence drift
per-class instability
```

---

## 15. Inference Pipeline

### 15.1 Function Signature

Implement a production-style function:

```python
def predict_expression_from_image(image: np.ndarray) -> dict:
    """
    Input:
        image: RGB image/frame, expected production size 480x480 or compatible.

    Output:
        {
            "label": "happiness",
            "label_id": 3,
            "probabilities": {
                "anger": 0.01,
                "disgust": 0.00,
                "fear": 0.02,
                "happiness": 0.91,
                "sadness": 0.02,
                "surprise": 0.03,
                "neutral": 0.01
            },
            "quality": {
                "landmark_success": true,
                "face_detection_confidence": 0.98
            }
        }
    """
```

### 15.2 Failure Behavior

If no valid landmarks are detected:

```json
{
  "label": null,
  "label_id": null,
  "probabilities": null,
  "quality": {
    "landmark_success": false,
    "failure_reason": "no_face_detected"
  }
}
```

Do not return a forced emotion prediction when landmark quality is below threshold.

---

## 16. Repository Structure

Recommended structure:

```text
project_root/
  configs/
    data/rafdb.yaml
    models/gcn.yaml
    models/gat.yaml
    models/graphormer_lite.yaml
    label_map.yaml
    mediapipe_regions.yaml
    graph_edges.yaml

  data/
    raw/rafdb/                  # local only, not committed
    processed/landmarks/         # normalized landmark JSONL/parquet
    splits/

  src/
    data/
      extract_landmarks.py
      build_dataset.py
      dataset.py
      validation.py

    preprocessing/
      normalize_landmarks.py
      region_features.py
      quality_checks.py

    graph/
      build_edges.py
      graph_features.py
      shortest_paths.py
      structural_bias.py

    models/
      gcn.py
      gat.py
      graphormer_lite.py
      heads.py

    training/
      train.py
      evaluate.py
      losses.py
      metrics.py
      callbacks.py

    inference/
      predictor.py
      mediapipe_runner.py

    utils/
      config.py
      logging.py
      seed.py

  notebooks/
    landmark_quality_analysis.ipynb
    graph_model_error_analysis.ipynb

  reports/
    data_quality/
    evaluation/
    figures/

  tests/
    test_landmark_normalization.py
    test_graph_construction.py
    test_dataset_shapes.py
    test_model_forward.py
    test_inference_failure_modes.py
```

---

## 17. Implementation Tasks

### Phase 1 — Dataset and Landmark Extraction

- [ ] Obtain RAF-DB dataset according to its license/access rules.
- [ ] Implement RAF-DB annotation parser.
- [ ] Implement label mapping for 7 basic expressions.
- [ ] Implement MediaPipe landmark extraction script.
- [ ] Save landmark-only processed dataset.
- [ ] Store extraction failures separately.
- [ ] Generate data quality report.

Deliverables:

```text
data/processed/landmarks/rafdb_train.jsonl
data/processed/landmarks/rafdb_val.jsonl
data/processed/landmarks/rafdb_test.jsonl
reports/data_quality/rafdb_landmark_extraction_report.md
```

### Phase 2 — Normalization and Graph Construction

- [ ] Implement landmark centering.
- [ ] Implement inter-ocular scaling.
- [ ] Implement eye-line rotation alignment.
- [ ] Validate output shape and numerical stability.
- [ ] Define MediaPipe graph edges.
- [ ] Define region IDs.
- [ ] Compute shortest-path distance matrix.
- [ ] Compute graph attention bias matrices.

Deliverables:

```text
src/preprocessing/normalize_landmarks.py
src/graph/build_edges.py
src/graph/structural_bias.py
configs/mediapipe_regions.yaml
configs/graph_edges.yaml
```

### Phase 3 — Baseline Models

- [ ] Implement GCN baseline.
- [ ] Implement GAT baseline.
- [ ] Implement training script.
- [ ] Implement metric reporting.
- [ ] Train and evaluate GCN.
- [ ] Train and evaluate GAT.

Deliverables:

```text
src/models/gcn.py
src/models/gat.py
reports/evaluation/gcn_baseline.md
reports/evaluation/gat_baseline.md
```

### Phase 4 — Graph Transformer / Graphormer-Lite

- [ ] Implement coordinate token embedding.
- [ ] Implement landmark ID embedding.
- [ ] Implement region ID embedding.
- [ ] Implement centrality encoding.
- [ ] Implement shortest-path attention bias.
- [ ] Implement region-relation attention bias.
- [ ] Implement Graph Transformer layers.
- [ ] Implement classifier head.
- [ ] Train and evaluate Graphormer-lite.

Deliverables:

```text
src/models/graphormer_lite.py
src/graph/shortest_paths.py
src/graph/structural_bias.py
reports/evaluation/graphormer_lite.md
```

### Phase 5 — Model Selection and Production Inference

- [ ] Compare GCN, GAT, and Graphormer-lite.
- [ ] Select best model based on macro-F1, per-class recall, and latency.
- [ ] Implement inference wrapper.
- [ ] Implement quality-gated prediction.
- [ ] Export model weights and config.
- [ ] Add unit tests for inference failure cases.

Deliverables:

```text
reports/evaluation/model_comparison.md
models/best_model.pt
models/best_model_config.yaml
src/inference/predictor.py
```

---

## 18. Acceptance Criteria

### Data Pipeline

- [ ] RAF-DB annotations parsed correctly.
- [ ] Landmark extraction success rate reported.
- [ ] Failed extractions are tracked and not silently discarded.
- [ ] Processed dataset contains no raw images.
- [ ] Normalized landmarks have shape `[478, 3]`.
- [ ] No NaN/Inf values in processed landmarks.

### Model Training

- [ ] GCN baseline trains end-to-end.
- [ ] GAT baseline trains end-to-end.
- [ ] Graphormer-lite trains end-to-end.
- [ ] All models produce 7-class logits.
- [ ] Metrics are computed on validation and test sets.
- [ ] Confusion matrix is generated.

### Model Quality

Initial target:

```text
Graphormer-lite must outperform GCN on macro-F1 or balanced accuracy.
```

Preferred target:

```text
Graphormer-lite must outperform both GCN and GAT on macro-F1.
```

Do not use accuracy as the only selection metric.

### Production Behavior

- [ ] Inference accepts 480x480 image/frame input.
- [ ] MediaPipe extraction runs locally.
- [ ] Raw image is not stored by inference code.
- [ ] Low-quality/no-face cases return a structured failure response.
- [ ] Latency is measured and reported.

---

## 19. Experiment Tracking

Use one of:

```text
MLflow
Weights & Biases
ClearML
CSV/JSON logs if external tools are not allowed
```

Track:

```text
model name
config
random seed
train/val/test metrics
class weights
loss curves
confusion matrix
best checkpoint path
inference latency
landmark extraction settings
```

Run at least 3 seeds for the final selected architecture:

```text
seed 42
seed 123
seed 2025
```

Report mean and standard deviation.

---

## 20. Testing Requirements

### Unit Tests

Implement tests for:

```text
landmark normalization shape
normalization numerical stability
graph edge construction
shortest path matrix shape
attention bias matrix shape
dataset item format
GCN forward pass
GAT forward pass
Graphormer-lite forward pass
inference no-face failure
```

### Minimum Test Examples

```python
def test_normalized_landmarks_shape():
    assert landmarks_norm.shape == (478, 3)


def test_graphormer_forward_shape():
    logits = model(batch)
    assert logits.shape == (batch_size, 7)
```

---

## 21. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| MediaPipe fails on some RAF-DB images | Biased training data | Report failure rate per class and use quality-gated filtering. |
| Landmark-only signal is weaker than image pixels | Lower accuracy than CNN/Vision Transformer | Use graph topology, region features, and Graph Transformer attention bias. |
| Graph Transformer overfits | Poor test generalization | Use dropout, label smoothing, landmark noise, early stopping, and multiple seeds. |
| Class imbalance | Poor minority-class recall | Use class weights, focal loss experiments, macro-F1 model selection. |
| Train-production mismatch | Poor 480x480 production behavior | Validate on production-style 480x480 data. |
| Attention maps misinterpreted | False explainability | Use ablation tests, not attention alone. |
| Landmark data remains biometric-derived | Privacy/security risk | Store securely; avoid raw images; document retention policy. |

---

## 22. Definition of Done

The task is complete when the ML Engineer provides:

1. Processed RAF-DB landmark-only dataset.
2. Data quality report.
3. Normalization and graph construction modules.
4. GCN baseline implementation and report.
5. GAT baseline implementation and report.
6. Graphormer-lite implementation and report.
7. Model comparison report with macro-F1, balanced accuracy, per-class metrics, confusion matrix, and latency.
8. Best model checkpoint and config.
9. Production-style inference wrapper for 480x480 image/frame input.
10. Unit tests for preprocessing, graph construction, model forward pass, and inference failure modes.
11. Clear README with reproduction instructions.

---

## 23. Recommended First Implementation Order

```text
1. RAF-DB parser
2. MediaPipe landmark extraction
3. Normalization
4. Processed JSONL/parquet dataset
5. Graph edge construction
6. GCN baseline
7. GAT baseline
8. Graphormer-lite
9. Evaluation report
10. Production inference wrapper
```

Do not start with Graphormer-lite before the landmark extraction and GCN baseline are verified.

---

## 24. Minimal Reproduction Commands

The engineer should implement commands similar to:

```bash
python -m src.data.extract_landmarks \
  --config configs/data/rafdb.yaml \
  --output data/processed/landmarks

python -m src.training.train \
  --config configs/models/gcn.yaml

python -m src.training.train \
  --config configs/models/gat.yaml

python -m src.training.train \
  --config configs/models/graphormer_lite.yaml

python -m src.training.evaluate \
  --checkpoint models/best_model.pt \
  --split test
```

---

## 25. Notes on SigLIP2

SigLIP2 is not part of the current implementation. It is not the correct starting backbone for this task because the current architecture must use normalized landmark coordinates, not image patches.

Possible future use:

```text
landmark graph embedding
→ align with SigLIP2 text embeddings for emotion prompts
```

This can be explored later as semantic regularization, not as the core model.

---

## 26. Final Engineering Principle

Prioritize a reproducible, security-compatible graph pipeline over a complex model that is difficult to validate.

The target architecture is not simply “emotion classification.” It is:

```text
landmark-only, graph-structured, quality-gated facial expression classification
```

The model output must be described as **facial expression category**, not a definitive internal emotional state or clinical diagnosis.
