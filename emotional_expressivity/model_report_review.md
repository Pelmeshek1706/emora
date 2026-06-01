# Model Report Review

Source report: `emotional_expressivity/Emotional Expressivity Biomarkers for PTSD, Depression, and Anxiety Detection.pdf`

Review scope:

- only models or model stacks explicitly mentioned in the report were included
- I separate `study-specific psychiatric models` from `runtime/tooling models` the report recommends for this repo
- `model card` means a public model card or official model documentation page
- if no public model card was found, I say so directly rather than guessing

Checked on: `2026-05-22`

## Bottom line

- The strongest models in the report are mostly `video-first`, `context-sensitive`, and often `multimodal`; they are not simple image FER classifiers.
- The most reusable visual outputs across studies are `AUs`, `head pose`, `gaze/eye-region signals`, and `temporal dynamics`, not just 7-class emotion probabilities.
- Public model cards are common for `py-feat` components and `MediaPipe` bundles, but are usually absent for the psychiatric research models in the papers.
- Most study models report `AUC`, `F1`, or `accuracy`; very few report clinically useful `specificity`.

## 1. Study-Specific CV Models Mentioned In The Report

| Model / paper | Target | Dataset and specificity | Data option | Visual inputs / features | How it was trained / validated | Model outputs | Reported performance | Specificity | Model card |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Alghowinem et al., 2013` | Depression screening | Clinically validated depression interview videos; exact `N` was not recoverable from the inspected report/source. Strong point: real clinical setting. Weak point: dataset details are incomplete in the accessible source. | Video | Head pose and movement from a `3D face model -> 2D AAM` pipeline; low-level motion plus statistical functionals. | `SVM`; subject-independent, leave-one-subject-out style validation per the report summary. | Binary depressed vs non-depressed recognition score. | `71.2%` average recognition. | `NR` | No public model card found; paper/source only. |
| `Harati et al., 2020` | Depression severity / DBS recovery | `12` severe MDD patients, repeated longitudinal clinical interviews during DBS recovery. Specific to severe treatment-resistant depression; very small `N`, but clinically grounded and repeated-measures. | Video | Dynamic facial variability from muted interview video; multiscale entropy and temporal variability features. | Ordinal regression with `elastic-net` regularization against severity/recovery levels. | Ordinal depression-severity estimate. | The report says the model separated severity/recovery levels, but the accessible summary did not expose a full metric table. | `NR` | No public model card found; paper/source only. |
| `Jiang et al., 2021` | Remission and treatment-response classification in MDD | `365` interviews, `88` hours, `12` depressed DBS patients followed over time. Strong longitudinal design; weak generalizability because `N` is small and clinical context is narrow. | Video | `7` basic emotions, `AUs`, temporal statistics; report says a regional CNN detector plus an `ImageNet`-pretrained emotion CNN and `OpenFace` AU extraction. | Leave-one-subject-out validation over longitudinal recordings. | Remission class and treatment-response class; feature stream includes framewise emotions and AUs aggregated over time. | `AUC 0.72` for remission and `0.75` for treatment response. | `NR` | No public model card found; paper/source only. |
| `Mahayossanunt et al., 2023` | Depression classification | `474` clinical interview videos from Chulalongkorn University; `134` depressed and `340` non-depressed. Better sample size than most studies in the report, but still private/single-program data. | Video | Gaze angles, angle/radian features, AU intensity, facial-expression features; privacy-protected structured facial features rather than raw face frames at inference-time. | `LSTM` with attention-based intermediate fusion and label smoothing. | Binary depression prediction. | Accuracy `91.67%`, `F1 88.89%`, precision `91.40%`, recall `87.03%`. | Not explicitly reported in the report. | No public model card found; paper/source only. |
| `Kim et al., 2023` | Depressive symptom association in older adults | `59` older adults without cognitive impairment. Specific to older-adult affect and symptom tracking, not a general depression-diagnosis benchmark. | Video | `OpenFace 2.0` AUs across posed and spontaneous emotion tasks; `17` AUs across six elicited emotions. | `PCA` plus multiple regression; this is an analysis pipeline, not a deployed classifier. | AU components associated with symptom severity; no direct diagnostic output. | Association study, not a classifier benchmark. | Not applicable. | No public model card found for this study pipeline; `OpenFace 2.0` itself has docs/paper, not a model card. |
| `Jin et al., 2025` | Depression classification and severity estimation | `E-DAIC`, `219` participants (`163` train / `56` val / `10` test in the report summary). Better benchmark value because the dataset is public and clinically anchored. | Video plus audio | `OpenFace` facial features, gaze, head pose, AUs, plus audio `MFCC`; explanations aggregated every `100` frames. | `TSNet-DD` for video, `GCN-LSTM` for audio, `VAFN` for fusion. The paper reports separate unimodal and multimodal training. | Video-only depression score, multimodal depression score, and severity estimate (`MAE` reported). | Video-only `F1 0.853`, `AUC 0.912`; fused `F1 0.922`, `AUC 0.950`, `MAE 3.51`. | `NR` | No public model card found; paper/source only. |
| `Schultebraucks et al., 2022` | PTSD and depression after trauma | `81` trauma survivors, one month after ED admission; free-speech interviews. Strong ecological validity and clear clinical target; still modest `N`. | Video plus audio plus text | Facial emotion/intensity, movement parameters, speech prosody, and language markers from free-speech responses. | Supervised deep neural network over multimodal features; report notes two hidden layers and repeated cross-validation/internal test evaluation in the source summary. | PTSD classification, depression classification, and symptom-severity regressions. | PTSD `AUC 0.90`, weighted `F1 0.83`; depression `AUC 0.86`, weighted `F1 0.82`. | `NR` | No public model card found; paper/source only. |
| `Aathreya et al., 2025` | Child PTSD classification | `18` children across `7` conversational contexts. Very small dataset, but notable because it is de-identified and explicitly context-aware. | Video-derived de-identified temporal features | AU intensities, facial landmarks, eye gaze, head pose; AU sequences were the strongest baseline family in the report. | Transformer with learnable Fourier encoding on AU sequences; experiments compared contexts and feature families. | PTSD vs non-PTSD prediction from AU/landmark/gaze/pose sequences. | Full metric table was not exposed in the accessible abstract/report summary. | `NR` | No public model card found; paper/source only. |
| `Ren et al., 2025` | GAD screening | `60` GAD patients and `60` matched controls; diagnosis benchmarked against `MINI`. Stronger clinical specificity than most rows here, but still a single diagnostic setup. | Facial-expression responses during an `IAPS` elicitation paradigm; likely video or repeated frame-based capture rather than raw still-photo classification. | Seven expression categories, especially `neutral`, `anger`, and `fear`. | Not a learned multivariate CV classifier in the report summary; the study used validity analysis plus `ROC` screening on expression indicators. | Expression intensity markers and ROC cutoffs, not an end-to-end diagnosis network. | `AUC 0.792` for anger, `0.727` for fear, `0.723` for neutral. | Yes. Neutral: `52%` sensitivity / `85%` specificity. Anger: `92%` sensitivity / `62%` specificity. Fear: `92%` sensitivity / `52%` specificity. | No public model card found; paper/source only. |
| `Zhou et al., 2023` | Depression / anxiety / apathy in MCI | `319` older adults with mild cognitive impairment. Useful multiclass symptom study, but population-specific to older adults with MCI. | Video plus audio plus text | Speech, facial-expression, and text features from open-source toolkits. | `Random forest` / multiclass symptom models. The report does not expose a detailed split strategy. | Multiclass symptom label. | Accessible summaries report weighted-average `F1 96.6%`, accuracy `87.4%`, precision `86.6%`, recall `87.6%`. | `NR` | No public model card found; paper/source only. |
| `AnxietyFaceTrack, 2025 preprint` | Social anxiety-like state, not formal disorder diagnosis | `91` students, `1,173` non-overlapping `10`-second smartphone samples. Good for naturalistic social-state detection; not a clinical psychiatric dataset. | Smartphone video | `OpenFace`-derived `669` retained features from landmarks, eye gaze, head pose, AU presence, AU intensity, and face-edge/jawline geometry. | `Random Forest` was best; `5`-fold CV; top-feature ablations. | Multiclass or binary anxiety-state class. | Multiclass accuracy `0.91`, `F1 0.90`, `AUC 0.98`; binary accuracies `0.92-0.93`. | `NR` | No public model card found; preprint only. |

Not included in the table above: `Fujiwara et al., 2015`, because the report treats it as a behavioral emotional-reactivity pilot rather than an automated computer-vision model.

## 2. Runtime / Tooling Models Also Mentioned In The Report

These are the practical model components the report names for `OpenWillis`, `py-feat`, `OpenFace`, or `MediaPipe`. They matter because they are the models most likely to be reused in this repository.

| Model / component | Where it appears in the report / repo context | Dataset / training provenance | Data option | Outputs | Current role for this repo | Model card / official doc |
| --- | --- | --- | --- | --- | --- | --- |
| `py-feat ResMaskNet` | Current OpenWillis emotion model path per local notes in `mediapipe_landmark_emotion_pipeline.md`. | Public `py-feat` card says it is a residual masking network with U-Net structure for `7` emotions. Local repo notes say the installed `feat==0.6.2` setup uses a `FER2013`-style configuration and `AutumnQiu/fer2013`; local notes also capture hyperparameters such as `224x224`, `lr=1e-4`, `batch_size=48`, `weight_decay=1e-3`, `max_epoch_num=50`. | Single face image / video frame. | `7` emotion probabilities: `anger`, `disgust`, `fear`, `happiness`, `sadness`, `surprise`, `neutral`. | This is the emotion head most directly aligned with the current `openwillis.face.facial_emotion.py` path. | Public card: [py-feat/resmasknet](https://huggingface.co/py-feat/resmasknet). Supporting docs: [py-feat included models](https://py-feat.org/pages/models.html). |
| `py-feat svm_emo` | Mentioned in local notes as an alternate py-feat emotion model, not the current default. | `py-feat` docs say it is an `SVM` trained on HOG features from `ExpW`, `CK+`, and `JAFFE`. | Single face image / video frame after HOG extraction. | `7` discrete emotion outputs over the same basic categories. | Useful as a lightweight or interpretable alternative baseline if the repo wants to compare against `ResMaskNet`. | Public card: [py-feat/svm_emo](https://huggingface.co/py-feat/svm_emo). Supporting docs: [py-feat included models](https://py-feat.org/pages/models.html). |
| `py-feat xgb_au` | The local OpenWillis path uses py-feat AU extraction; local macOS workaround comments explicitly mention the `XGBoost` AU detector. | `py-feat` docs say it is an `XGBoost` AU detector trained on HOG features extracted from `BP4D`, `DISFA`, `CK+`, `UNBC-McMaster Shoulder Pain`, and `AFF-Wild2`. | Single face image / video frame after HOG extraction. | `20` AU probabilities in the py-feat output space. | This is the AU head most directly relevant to the current repo outputs and to the PTSD/depression literature in the report. | Public card: [py-feat/xgb_au](https://huggingface.co/py-feat/xgb_au). Supporting docs: [py-feat included models](https://py-feat.org/pages/models.html). |
| `py-feat RetinaFace` | Mentioned in local notes as the current default face detector in the installed `feat==0.6.2` runtime. | Public card says it is a PyTorch implementation of `RetinaFace`, evaluated on `WIDER FACE`; the card notes a `mobilenet0.25` backbone by default and support for `resnet50`. | Single image or video frame. | Face bounding boxes, confidence scores, and `10` facial landmark keypoints. | Front-end face detection for the current py-feat-based emotional-expressivity path. | Public card: [py-feat/retinaface](https://huggingface.co/py-feat/retinaface). |
| `py-feat MobileFaceNet` | Mentioned in local notes as the current default landmark model in the installed py-feat runtime. | `py-feat` exposes it as a landmark detector, but the public card is thin and does not document the landmark-training dataset. The original `MobileFaceNet` paper is a compact CNN for face verification, trained with `ArcFace`-style supervision on refined `MS-Celeb-1M`; that is the backbone lineage, not necessarily the full py-feat landmark-head training recipe. | Single image or video frame. | Facial landmark coordinates used downstream by AU and emotion heads. | Landmark stage in the current py-feat stack. | Public card exists but is sparse: [py-feat/mobilefacenet](https://huggingface.co/py-feat/mobilefacenet). Model-family reference: [MobileFaceNets paper](https://arxiv.org/abs/1804.07573). |
| `OpenFace 2.0` | Explicitly named in the report as a preferred interpretable feature extractor; also used inside several studies in the table above. | Composite toolkit rather than one single model. Official docs/paper say it includes models for facial landmark detection, head pose estimation, eye gaze estimation, facial action unit recognition, and aligned-face/HOG feature extraction. Training is distributed across submodules rather than one unified published training recipe. | Images and videos; often used on video. | Landmarks, head pose, eye gaze, AU presence/intensity, aligned faces, HOG features. | Strong candidate if this repo expands beyond py-feat emotion summaries toward `AUs + pose + gaze + dynamics`. | No single model card. Official repo: [OpenFace GitHub](https://github.com/TadasBaltrusaitis/OpenFace). Official paper: [OpenFace 2.0 paper](https://par.nsf.gov/servlets/purl/10099458). |
| `MediaPipe Face Landmarker bundle` | Explicitly recommended in the report as a fallback/cross-check stack; also directly relevant to the local `mediapipe_landmark_emotion` research path. | Official MediaPipe docs say the bundle packages three models: `BlazeFace` face detector, `FaceMesh-V2`, and a `Blendshape` predictor. The docs also expose direct model-card links for the packaged components. | Images, videos, or live streams. | `478` three-dimensional face landmarks, `52` blendshape scores, and facial transformation matrices. | Best low-friction path if the repo wants privacy-preserving geometry and expression proxies without running the full py-feat image stack. | Official doc with bundled model-card links: [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker). |
| `py-feat img2pose` | Mentioned in py-feat docs as the face/pose detector family; the broader report also recommends explicit head-pose outputs. | Public card says `img2pose` uses a `Faster R-CNN` architecture. The original paper says the model uses a `ResNet-18` backbone and is trained on `WIDER FACE` with a mix of weakly supervised and human-annotated pose labels. | Single image or video frame. | Face boxes plus full `6DoF` head pose. | Valuable if this repo decides to elevate head-pose features to first-class outputs. | Public card: [py-feat/img2pose](https://huggingface.co/py-feat/img2pose). Paper: [img2pose CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Albiero_img2pose_Face_Alignment_and_Detection_via_6DoF_Face_Pose_Estimation_CVPR_2021_paper.pdf). |

## 3. What The Models Actually Output

This is where the report is most useful: the outputs are not interchangeable.

| Output family | Models that produce it | Notes |
| --- | --- | --- |
| `7` basic emotion probabilities / labels | `py-feat ResMaskNet`, `py-feat svm_emo`, parts of `Jiang et al., 2021`, `Ren et al., 2025` | Useful, but the report repeatedly argues these are not enough by themselves for psychiatric screening. |
| `AUs` | `py-feat xgb_au`, `OpenFace 2.0`, `Kim et al., 2023`, `Jiang et al., 2021`, `Jin et al., 2025`, `Aathreya et al., 2025`, `AnxietyFaceTrack` | This is the most reusable output family across depression and PTSD work in the report. |
| Head pose / head movement | `OpenFace 2.0`, `py-feat img2pose`, `Alghowinem et al., 2013`, `Jin et al., 2025`, `Aathreya et al., 2025`, `AnxietyFaceTrack` | Particularly important for anxiety-like states and still underused in the current repo summary path. |
| Gaze / eye-region signals | `OpenFace 2.0`, `MediaPipe Face Landmarker` indirectly via landmarks/blendshapes, `Jin et al., 2025`, `Aathreya et al., 2025`, `AnxietyFaceTrack` | The report treats gaze as helpful but usually weaker alone than AUs plus pose plus audio. |
| Landmark geometry / blendshapes | `MediaPipe Face Landmarker`, `OpenFace 2.0`, `Aathreya et al., 2025`, `AnxietyFaceTrack` | Strong for privacy-preserving pipelines because geometry can be stored without raw video. |
| Diagnostic / severity score | `Alghowinem`, `Harati`, `Jiang`, `Mahayossanunt`, `Jin`, `Schultebraucks`, `Zhou`, `AnxietyFaceTrack` | These are study-specific heads. Most do not have standalone public model artifacts or model cards. |

## 4. Video Versus Raw Photo

The report is much more favorable to `video` than to `single-image` psychiatric modeling.

| Mostly frame/image-level models | Mostly video/sequence models |
| --- | --- |
| `py-feat ResMaskNet` | `Alghowinem et al., 2013` |
| `py-feat svm_emo` | `Harati et al., 2020` |
| `py-feat xgb_au` | `Jiang et al., 2021` |
| `py-feat RetinaFace` | `Mahayossanunt et al., 2023` |
| `py-feat MobileFaceNet` | `Kim et al., 2023` |
| `py-feat img2pose` | `Jin et al., 2025` |
| `MediaPipe Face Landmarker` can run on a single image, but it is often more useful as a video stream primitive | `Schultebraucks et al., 2022` |
|  | `Aathreya et al., 2025` |
|  | `Zhou et al., 2023` |
|  | `AnxietyFaceTrack, 2025` |

Interpretation:

- `single-frame models` are mostly reusable `feature extractors`
- the better psychiatric papers are mostly `windowed`, `sequence`, or `multimodal`
- if this repo stays at framewise mean/std summaries, it will remain materially weaker than the best evidence in the report

## 5. Model-Card Availability Summary

| Has a public model card or official model page | No public model card found |
| --- | --- |
| `py-feat/resmasknet` | `Alghowinem et al., 2013` |
| `py-feat/svm_emo` | `Harati et al., 2020` |
| `py-feat/xgb_au` | `Jiang et al., 2021` |
| `py-feat/retinaface` | `Mahayossanunt et al., 2023` |
| `py-feat/mobilefacenet` | `Kim et al., 2023` study pipeline |
| `py-feat/img2pose` | `Jin et al., 2025` |
| `MediaPipe Face Landmarker` bundle docs and component model cards | `Schultebraucks et al., 2022` |
|  | `Aathreya et al., 2025` |
|  | `Ren et al., 2025` |
|  | `Zhou et al., 2023` |
|  | `AnxietyFaceTrack, 2025` |
|  | `OpenFace 2.0` has strong official docs and a paper, but not a model-card style artifact |

## 6. What This Means For The Current OpenWillis Repo

If the goal is a stronger `emotional_expressivity` stack, the report supports this order of operations:

1. Keep `py-feat` emotions and `AUs` as the transparent baseline.
2. Add explicit `head pose`, `gaze`, `blink`, `quality/confidence`, and `temporal-dynamics` outputs.
3. Treat `video windows` as the unit of analysis, not only framewise means.
4. Prefer `AUs + pose + gaze + audio/text fusion` over "raw emotion scores only".
5. Treat `MediaPipe` geometry as a privacy-friendly fallback or cross-check, not as a perfect substitute for all AU/pose pipelines.

The report does **not** support jumping directly to a large opaque raw-video classifier on this repo's current evidence base.

## 7. Practical Ranking By Reuse Value For This Repo

| Rank | Model / family | Why it matters most here |
| ---: | --- | --- |
| 1 | `OpenFace 2.0` / `py-feat xgb_au` / AU-centric outputs | Best alignment with the literature's most reproducible facial biomarker family. |
| 2 | `py-feat img2pose` or another explicit head-pose path | Head dynamics are repeatedly useful, especially for anxiety-like states. |
| 3 | `MediaPipe Face Landmarker` | Best privacy-preserving geometry fallback and a good path for landmark-only research variants. |
| 4 | `py-feat ResMaskNet` | Useful baseline emotion head, but not enough by itself. |
| 5 | Study-specific multimodal models such as `Jin et al., 2025` and `Schultebraucks et al., 2022` | These are the strongest performance examples, but they depend on datasets, labels, and modalities that the current repo does not yet reproduce. |

## Sources

Primary local source:

- `emotional_expressivity/Emotional Expressivity Biomarkers for PTSD, Depression, and Anxiety Detection.pdf`

Local implementation sources:

- `openwillis-face/src/openwillis/face/facial_emotion.py`
- `emotional_expressivity/mediapipe_landmark_emotion/mediapipe_landmark_emotion_pipeline.md`
- `emotional_expressivity/model_training_report.md`

Public model cards / official docs:

- [py-feat model list](https://py-feat.org/pages/models.html)
- [py-feat/resmasknet](https://huggingface.co/py-feat/resmasknet)
- [py-feat/svm_emo](https://huggingface.co/py-feat/svm_emo)
- [py-feat/xgb_au](https://huggingface.co/py-feat/xgb_au)
- [py-feat/retinaface](https://huggingface.co/py-feat/retinaface)
- [py-feat/mobilefacenet](https://huggingface.co/py-feat/mobilefacenet)
- [py-feat/img2pose](https://huggingface.co/py-feat/img2pose)
- [OpenFace GitHub](https://github.com/TadasBaltrusaitis/OpenFace)
- [OpenFace 2.0 paper](https://par.nsf.gov/servlets/purl/10099458)
- [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker)
- [img2pose paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Albiero_img2pose_Face_Alignment_and_Detection_via_6DoF_Face_Pose_Estimation_CVPR_2021_paper.pdf)

Study sources already cited or cross-checked in local notes:

- [Alghowinem et al., 2013](https://researchportalplus.anu.edu.au/en/publications/head-pose-and-movement-analysis-as-an-indicator-of-depression/)
- [Harati et al., 2020](https://scholars.mssm.edu/en/publications/classifying-depression-severity-in-recovery-from-major-depressive-2/)
- [Kim et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10459725/)
- [Jin et al., 2025](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2025.1508772/full)
- [Schultebraucks et al., 2022](https://www.cambridge.org/core/journals/psychological-medicine/article/deep-learningbased-classification-of-posttraumatic-stress-disorder-and-depression-following-trauma-utilizing-visual-and-auditory-markers-of-arousal-and-mood/733197598EA30BC8379D151173AEFF8F)
- [Aathreya et al., 2025](https://www.sciencedirect.com/science/article/abs/pii/S0167865525001928)
- [Zhou et al., 2023](https://www.sciencedirect.com/science/article/pii/S002074892300127X)
- [AnxietyFaceTrack, 2025](https://arxiv.org/abs/2502.16106)
