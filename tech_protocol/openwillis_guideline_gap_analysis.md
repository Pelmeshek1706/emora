# OpenWillis Guideline Gap Analysis for AIREST Video Collection

Reviewed on: 2026-05-22

## Purpose

This note identifies collection decisions that are not clearly covered by OpenWillis and records the questions AIREST must answer independently before using video-derived face, gaze, calibration, and QC data in a clinical-site workflow.

Local AIREST context used:

- `airest-gaze/airest_cv`
- `airest-face`

Jira references:

- [ARST-169](https://pelmeshek.atlassian.net/browse/ARST-169): Video Data Collection Protocol & Technical Documentation
- [ARST-178](https://pelmeshek.atlassian.net/browse/ARST-178): Review OpenWillis video data collection guidance

OpenWillis source links reviewed:

- [OpenWillis main documentation](https://openwillis.brooklyn.health/?pvs=32)
- [Guidelines for Video-Based Data Collection](https://openwillis.brooklyn.health/Guidelines-for-Video-Based-Data-Collection-17483a8fe04780d98d26e6fd47cbdfe6)
- [Guidelines for Facial Expressivity Measures](https://openwillis.brooklyn.health/Guidelines-for-Facial-Expressivity-Measures-17483a8fe047800daf7ad2b237f06654)
- [Facial Expressivity v2.2](https://openwillis.brooklyn.health/Facial-Expressivity-v2-2-1b483a8fe04780ae9c63d9e4034a8463)
- [Head Movement v1.0](https://openwillis.brooklyn.health/Head-Movement-v1-0-1b483a8fe0478029b580eb35c756066a)
- [Eye blink rate v1.0](https://openwillis.brooklyn.health/Eye-blink-rate-v1-0-15883a8fe047809f82fcef89964973e3)

## Short Conclusion

OpenWillis gives useful general advice for video quality and facial expressivity baselining, but it does not provide a complete real-world clinical data collection protocol. Most OpenWillis material is post-collection processing documentation: it describes how existing videos are analyzed, which inputs functions accept, which framewise or summary outputs are produced, and which model assumptions matter.

AIREST therefore cannot treat OpenWillis as a collection SOP. AIREST must define the clinical capture protocol, calibration pass/fail logic, QC thresholds, storage policy, timestamp model, participant/assistant workflow, and session artifact schema independently.

## Nature of the Source Material

| Source | What it describes | Real collection practice or post-collection processing? | AIREST interpretation |
| --- | --- | --- | --- |
| OpenWillis main documentation | Library purpose, setup, function pages, how-to guides, research guidelines, community information. | Mainly library/documentation overview. | Useful for provenance and scope, not a collection protocol. |
| Guidelines for Video-Based Data Collection | General advice: face should mostly face the camera; direct frontal camera placement is ideal; video calls can work; resolution is less important than face visibility; obstructions, shadows, poor lighting, and face-like background objects should be avoided. | Partial high-level collection guidance. It does not describe a site SOP, exact hardware, pass/fail thresholds, or actual clinical collection operations. | Use as qualitative guidance only. Convert into AIREST-owned measurable requirements. |
| Guidelines for Facial Expressivity Measures | Recommends baseline videos for clinical experiments and gives examples such as reading a sentence/passage or sitting and looking at the camera for about 10 seconds. It says baseline content depends on the experiment. | Experimental-design advice, not an operational protocol. | AIREST must decide the actual baseline task, timing, placement in workflow, and acceptance criteria. |
| Facial Expressivity v2.2 | Function syntax, `filepath`, `baseline_filepath`, `bbox_list`, sampling rate, normalization, speaking split, framewise landmarks/displacement, and summaries. | Mainly post-collection processing. | Useful for offline feature extraction and data schema planning, not enough for capture decisions. |
| Head Movement v1.0 | Function syntax and processing method: sample frames, estimate pose, calculate XY and rotational displacement, summarize mean/std. | Post-collection processing. | Useful for feature definitions; AIREST must define capture stability, pose QC, and camera geometry. |
| Eye blink rate v1.0 | Function syntax and processing method: use MediaPipe FaceMesh on video frames, calculate EAR, detect blink minima, output EAR, blink events, and summary. | Post-collection processing. | Useful for offline blink extraction; AIREST must define camera FPS, eye visibility QC, blink/gaze interaction policy, and failure handling. |

## What OpenWillis Clearly Covers

OpenWillis clearly supports these points:

- Video-derived functions are intended to calculate behavioral characteristics from the head/face region in videos.
- A target participant should be visible enough, with enough usable frames, for downstream summaries to be meaningful.
- The models work best when the person faces the camera; direct camera placement is preferred when collecting new experimental data.
- Previously collected videos can still be processed if the person faces mostly toward the lens; frames without a recognizable face are not processed.
- Very high resolution is not the main determinant of quality because face crops are resampled for model processing; a modern camera with the face as the subject is expected to be sufficient.
- Obstructions, shadows, poor lighting, and background items that can be mistaken for faces are practical quality risks.
- Baseline videos are recommended for facial expressivity in clinical experiments, but baseline content is experiment-specific.
- OpenWillis feature pages define offline processing inputs and outputs for facial expressivity, head movement, and blink rate.

## Collection Decisions Not Clearly Covered

| Decision area | What OpenWillis says | Gap for AIREST | Unresolved AIREST questions |
| --- | --- | --- | --- |
| Clinical-site SOP | Provides general collection considerations. | No end-to-end assistant workflow, site-room setup checklist, pre-session system check, or repeat/abort rules. | What exact steps must a clinical assistant follow before, during, and after capture? Which failures block the session versus allow review? |
| Laptop and OS requirements | Not specified. | No minimum CPU/GPU/RAM, OS/browser matrix, USB/camera backend policy, or local runtime constraints. | Which devices are approved? Is capture browser-owned, Python/OpenCV-owned, or split by modality? Which OS camera permission path is supported? |
| Webcam model, resolution, and delivered FPS | Says modern cameras are likely sufficient and high resolution is not the main barrier. Function pages define analysis sampling rates, not camera capture requirements. | No required webcam specs, actual delivered FPS checks, dropped-frame policy, camera index policy, or resolution/FPS fallback. | Should AIREST require 1080p/30 fps, 720p/60 fps, or another profile? What delivered FPS threshold fails face, gaze, blink, or head-pose capture? |
| Camera position and participant distance | Recommends a mostly frontal camera and direct placement when possible. | No distance, face-size, bounding-box, lens-height, screen-camera offset, seating, or tripod/laptop-angle requirements. | What face bounding-box size/range is acceptable? How far should the participant sit from the screen/camera? What screen-camera geometry is required for gaze? |
| Lighting and exposure | Warns against poor lighting and shadows across the face. | No measurable low-light/overexposure threshold or assistant remediation rule. | What proxy metrics define low light, glare, harsh shadows, or overexposure? Should the app block capture or only flag QC? |
| Background and other faces | Warns that face-like background objects can confuse models and notes preprocessing can separate multiple faces. | No operational policy for interviewer presence, caregivers, mirrors, pictures, or multiple detected faces during capture. | Must only the participant be visible? If another person enters, should the frame be rejected, the session repeated, or a target-face track selected? |
| Participant appearance and occlusion | Mentions hats, glasses, and obstructions as risks. | No participant instructions for glasses, masks, hair, hats, lighting reflection, or medical constraints. | Which occlusions are allowed? Should glasses be removed when possible? How should clinical constraints be documented when occlusion cannot be fixed? |
| Baseline recording | Strongly recommends baseline for facial expressivity and says content depends on the experiment. | No AIREST-specific baseline task, timing, duration, script, or QC rule. | Is the baseline neutral sitting, reading, fixation, or task-specific? How long is it? Is it repeated each visit/session? What makes a baseline invalid? |
| Gaze calibration | OpenWillis face documentation does not cover gaze calibration. | AIREST gaze repo has calibration logic, but no persisted pass/fail report or acceptance threshold. | What calibration grid, sample count, validation stage, error threshold, retry limit, and assistant action are required? How are blink-contaminated samples excluded? |
| Screen/task geometry | Not covered. | Gaze and attention-bias capture need screen dimensions, fullscreen state, stimulus location, browser zoom, display scaling, and screen brightness constraints. | Which display size/resolution/scaling is allowed? Must fullscreen be enforced? How are stimulus events synchronized to camera frames? |
| Timestamp and synchronization model | Not covered. | AIREST current gaze outputs use server wall-clock time after processing and lack camera frame timestamps, browser timestamps, and monotonic clocks. | Which timestamp is authoritative? How are camera frames, gaze samples, task events, audio/video media, and browser events aligned? |
| QC thresholds | Says sufficient usable frames are needed and missing face frames are skipped. | No numerical thresholds for face-present ratio, missing landmarks, eye visibility, pose range, frame drops, repeated frames, calibration error, or session validity. | What are warning versus blocking thresholds? Are thresholds feature-specific for face, gaze, blink, and head movement? |
| Missingness semantics | Says frames without a recognized face are not processed. | Does not define how missing frames affect denominators, summaries, recapture, or downstream modeling. | Are summaries computed over valid frames or total session time? What missingness level invalidates a feature family? How is missingness stored in schemas? |
| Raw video versus derived features | Function docs assume an input video path. | No privacy/storage decision for AIREST clinical capture. | Does AIREST store raw video, derived landmarks/features only, or both temporarily? What are retention, encryption, de-identification, and export rules? |
| Session artifacts and schema | Function pages show function outputs, not clinical session packages. | No session manifest, device metadata, calibration report, QC report, checksums, or schema versioning. | What exact files are produced per session? What schema versioning and checksum policy is required? |
| Error handling and recapture | Not covered. | No assistant-facing actions for camera failure, calibration failure, face loss, low FPS, low light, or upload/export failure. | Which failures trigger immediate retry? Which allow continuation with QC flags? What is the maximum retry burden for participants? |
| Realtime versus offline processing | Function pages are offline-style video processing. | AIREST must decide which checks run live for gating versus offline for final features. | Which features are realtime QC only? Which are offline research features? Should OpenWillis run during collection, after collection, or only in research batches? |
| Clinical visit variability | Baseline guidance notes confounds such as fatigue and medication effects, but does not define a protocol. | No standardization for time of day, medication state, fatigue, interviewer behavior, task order, or repeated visits. | Which contextual covariates must be captured? What must stay constant across visits? |
| Dry-run validation | Not covered as an OpenWillis protocol. | AIREST needs evidence that the selected hardware and workflow actually deliver usable face/gaze/blink/head data. | What dry-run package proves readiness: screenshots, logs, QC report, export manifest, checksums, sample feature tables, and review sign-off? |

## AIREST Gaze Repo-Specific Implications

The adjacent AIREST gaze repository is a useful prototype, but it does not close the OpenWillis collection gaps. It currently implements a Python gaze API, MediaPipe FaceMesh eye/iris extraction, calibration, point-of-gaze estimation, FastAPI/Socket.IO demo flow, and prototype outputs. It is not yet a clinical capture platform.

Important local gaps already documented in `/Users/pelmeshek1706/Desktop/projects/airest-gaze/airest_cv`:

- `docs/current_repo_risks.md` identifies blockers around session isolation, timestamp quality, calibration validation, camera failure handling, stale landmarks, privacy controls, and synchronization.
- `docs/gaze_part_structure.md` states that the current code should be read as a gaze and calibration core plus demo integration, not as the full clinical data-capture platform.
- `docs/current_repo_data_schema.md` states that the current files lack a central schema, session manifest, stable session IDs, synchronized timestamps, explicit failure flags, and calibration quality metadata.
- Current web gaze output is a prototype `{time, slide, x, y}` JSON written to `api_test_results/web_gaze_data.json`; it is not a session-scoped artifact set with QC, metadata, and checksums.

OpenWillis does not answer the AIREST-specific gaze questions. AIREST must define these independently:

1. Calibration target layout, fixation behavior, sample timing, retry behavior, and validation error threshold.
2. Whether calibration and gaze capture use the Python server camera path, browser camera APIs, or a unified capture controller.
3. How gaze samples are synchronized with screen stimulus events and media recordings.
4. How blinks, eye landmark loss, off-screen estimates, and out-of-bounds gaze are represented.
5. How calibration failure, low-quality calibration, and participant inability to calibrate are handled clinically.
6. Whether gaze accuracy requirements differ by task, such as attention bias versus general screen engagement.

## Priority Unresolved Questions for AIREST

### P0: Must be answered before clinical dry-run

1. What is the approved capture stack: browser, Python/OpenCV, or hybrid?
2. What laptop, webcam, OS, browser, camera permission, and network/offline requirements are allowed?
3. What are the required camera resolution and delivered FPS, and how is delivered FPS measured?
4. What exact participant position, camera angle, distance, face-size range, and screen geometry are required?
5. What are the live blocking QC checks before recording starts?
6. What calibration pass/fail threshold and retry policy will AIREST use?
7. What timestamp model aligns camera frames, gaze samples, task events, audio/video, and exported features?
8. Does AIREST store raw video, derived features, or both, and under what retention/encryption policy?
9. What session folder, manifest, checksum, and schema-version contract defines a valid completed session?

### P1: Should be answered before pilot data collection

1. What baseline clip is required for facial expressivity, how long is it, and where does it occur in the task flow?
2. What participant-facing script controls gaze, posture, movement, glasses, hats, masks, and speaking behavior?
3. What assistant-facing troubleshooting rules cover low light, glare, face loss, multiple faces, failed calibration, and camera failure?
4. Which OpenWillis-derived features are included in the MVP: facial landmarks, head pose, blink rate, facial expressivity, emotion/action units, or a smaller QC-only subset?
5. Which feature families are realtime, offline production, or research-only?
6. How are missingness and skipped frames represented in feature tables and QC summaries?
7. What contextual covariates must be captured: medication state, fatigue, time of day, interviewer presence, room, device, and participant constraints?

### P2: Can be refined after dry-runs and pilot evidence

1. Should thresholds differ by task type or clinical cohort?
2. Should AIREST normalize head movement and expressivity by face size, baseline clip, or both?
3. What minimum usable duration is needed per feature family?
4. What model/library versions are locked for production versus research runs?
5. How much manual review is acceptable for borderline QC sessions?

## Practical Recommendation

Use OpenWillis as a reference implementation and feature-definition source, not as the AIREST collection protocol. Convert its qualitative advice into measurable AIREST requirements:

- "Face mostly toward lens" becomes explicit face-present, pose-range, and face-size thresholds.
- "Poor lighting/shadows are bad" becomes low-light, overexposure, and glare QC proxies with assistant actions.
- "Modern camera should suffice" becomes approved camera profiles plus delivered-FPS and dropped-frame checks.
- "Baseline is recommended" becomes a fixed AIREST baseline task with duration, script, timing, and validity criteria.
- "Frames without face are skipped" becomes explicit missingness denominators, feature-specific validity thresholds, and stored failure reasons.

Until those decisions are made, OpenWillis can support offline analysis of already captured videos, but it cannot by itself make AIREST clinical video collection repeatable, auditable, or site-ready.
