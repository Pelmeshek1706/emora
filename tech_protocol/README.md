# AIREST Technical Data-Capture Protocol

This directory documents the MVP technical protocol for collecting interview
audio, video, transcript, feature, and quality-control data for AIREST research
and clinical screening workflows related to PTSD, depression, and anxiety.

The protocol is based on the source document below and the working research
summary provided for this repository. It is intentionally written as structured
Markdown. No JSON objects are used in this file.

## Source Document

| Field | Value |
| --- | --- |
| Source file | `Interview Data Capture for PTSD, Depression, and Anxiety Detection.pdf` |
| Local path | `tech_protocol/Interview Data Capture for PTSD, Depression, and Anxiety Detection.pdf` |
| File role | Research summary for video, audio, interview, QC, and metadata requirements |
| Review date | 2026-05-21 |
| Source size observed locally | About 52 KB |
| Scope | Interview data capture for PTSD, depression, and anxiety detection |

## Related Mapping Notes

| Note | Purpose |
| --- | --- |
| [OpenWillis Landmark Output Schema Note](openwillis_landmark_schema_note.md) | Maps the 468-landmark OpenWillis / MediaPipe Face Mesh output, displacement derivation, region summaries, and numbered landmark visualizations. |

## Executive Summary

AIREST data capture should be treated as a standardized clinical measurement
pipeline, not as a simple interview recording. The MVP should use a repeatable
browser-based protocol with stable camera placement, controlled audio capture,
neutral baseline collection, strict quality checks, participant-only transcript
protection, and metadata-first storage.

The central engineering requirement is that every session must be auditable:
AIREST should preserve not only model features, but also capture conditions,
device settings, frame/audio quality signals, processing confidence, errors,
interruptions, and exclusion reasons.

## Key Research Findings

### Video Capture

Reliable facial, head-pose, gaze, and expressivity studies repeatedly use
standardized video acquisition. For AIREST, the practical video target is:

| Requirement | MVP Value |
| --- | --- |
| Target resolution | 720p minimum target |
| Absolute minimum resolution | 640 x 480 |
| Preferred resolution | 1080p when storage and compute allow |
| Frame rate | 30 fps |
| Camera position | Front-facing or near-front-facing |
| Framing | Face centered, shoulders visible |
| Camera stability | Fixed webcam or stable laptop camera |
| Lighting | Even lighting, no strong shadows, no strong backlight |
| Room | Quiet and controlled |

30 fps appears as a repeated minimum in literature using facial action units,
head pose, gaze, facial movement, blink dynamics, and expressivity.

### Audio Capture

Audio must be controlled more strictly than video because poor sound quality
can degrade speech biomarkers more severely than moderately weak video degrades
facial features.

| Requirement | MVP Value |
| --- | --- |
| Preferred microphone | External or headset microphone |
| Avoid as primary source | Laptop built-in microphone |
| Minimum sample rate | 16 kHz |
| Preferred format | WAV, 44.1 kHz, 16-bit or better |
| Speaker separation | Separate interviewer and participant channels when possible |
| Fallback speaker separation | Diarization plus speaker role labels |
| Storage preference | Raw WAV or lossless/near-lossless audio when allowed |

If a human interviewer is present, the protocol should either capture separate
channels or perform diarization with explicit role labeling.

### Interview Protocol

The data collection process should not be a free-form clinical conversation.
The MVP should use a standardized, scripted, browser-based protocol.

| Requirement | MVP Value |
| --- | --- |
| Interview type | Semi-structured or scripted |
| Prompt sequence | Fixed and versioned |
| Interviewer | Virtual/self-guided preferred for MVP consistency |
| Human interviewer option | Strict training and role labeling required |
| Baseline | Required neutral video/audio segment before tasks |
| Session duration | Fixed target duration with allowed bounds |
| Task completion | Required task checklist |
| Interruptions | Explicitly logged |
| Missing data | Explicitly logged |
| Technical issues | Explicitly logged |

## Evidence Map From Research

| Study or Corpus | Observed Design Pattern | AIREST Implication |
| --- | --- | --- |
| DAIC-WOZ / E-DAIC | Standardized interview with virtual interviewer Ellie; includes audio, transcripts, landmarks, gaze, pose, action units, HOG features, frame-level confidence, detection success, interruptions, and excluded sessions | Save features and QC metadata together; include confidence and exclusion signals |
| Pittsburgh depression interview study | Multi-camera and multi-microphone interview setup; face, shoulders, body, patient, and interviewer captured; interview based around Hamilton Rating Scale for Depression | Multi-camera capture is useful for research, but MVP can use one strong frontal camera if focused on face, head, and gaze biomarkers |
| AViD / AVEC | Quiet setting, webcam and microphone, video standardized to 30 fps and 640 x 480, audio around 41 kHz and 16-bit | A standard webcam workflow can be acceptable when format and environment are controlled |
| PTSD veteran speech biomarker studies | Separate microphones or channels for interviewer and participant | Human-interviewer workflows need speaker-channel separation or diarization with role labels |
| MMPsy anxiety/depression corpus | Web application collection; control questions; filtering of invalid answers; removal of mute or too-short recordings; VAD, noise reduction, ASR, and manual semantic verification | Scalable MVP capture needs automatic quality filters during intake, not only after processing |

## MVP Base Requirements

These requirements are mandatory for the first clinical data-capture MVP.

1. Use a browser-based standardized protocol, not a free interview.
2. Target 720p at 30 fps webcam capture; accept 640 x 480 at 30 fps as the absolute minimum.
3. Prefer an external or headset microphone.
4. Capture a neutral baseline video/audio segment before interview tasks.
5. Run strict QC before, during, and after each session.
6. Protect participant-only transcripts through diarization and speaker role labeling.
7. Use metadata-first storage: preserve device specs, resolution, FPS, audio sample rate, confidence values, errors, interruptions, and exclusion reasons.
8. Do not store raw video by default if the regulatory or ethics strategy requires minimization; store derived face/video features and QC artifacts instead.

## Recommended MVP Capture Standard

| Area | Recommendation |
| --- | --- |
| Camera target | 720p minimum target, 30 fps |
| Absolute video minimum | 640 x 480, 30 fps |
| Preferred video | 1080p, 30 fps, fixed webcam |
| Audio device | External or headset microphone preferred |
| Audio minimum | WAV, 16 kHz |
| Audio preferred | WAV, 44.1 kHz, 16-bit or better |
| Room | Quiet, stable lighting, no backlight |
| Participant position | Face centered, shoulders visible |
| Baseline | Short neutral clip before tasks |
| Interview | Standardized scripted prompts |
| Metadata | Device, resolution, FPS, audio rate, lighting, QC flags |
| QC | Face visibility, frame drops, audio clipping, silence, interruptions |

## Browser-Based Session Flow

### 1. Pre-Session Intake

| Step | Required Behavior |
| --- | --- |
| Device check | Detect camera, microphone, browser, OS, and permissions |
| Camera preview | Verify face is visible and centered |
| Microphone preview | Verify microphone is present and not muted |
| Lighting check | Estimate brightness and flag strong backlight or low light |
| Noise check | Estimate background noise before recording starts |
| Consent state | Confirm consent and protocol version before capture |
| Participant ID | Use pseudonymized ID, never direct identity in media filenames |
| Protocol version | Store prompt set version and capture rules version |

### 2. Baseline Segment

| Step | Required Behavior |
| --- | --- |
| Baseline prompt | Ask participant to remain neutral and look toward the camera |
| Duration | Capture a short neutral audio/video segment |
| Purpose | Establish subject-specific neutral facial and acoustic baseline |
| QC | Confirm face presence, stable FPS, no mute audio, no clipping |
| Re-record rule | Require retry if baseline fails core QC |

### 3. Interview Tasks

| Step | Required Behavior |
| --- | --- |
| Prompt delivery | Use fixed scripted prompts in a stable sequence |
| Timing | Store prompt start and end timestamps |
| Response capture | Store response boundaries and completion status |
| Speaker roles | Label participant and interviewer turns |
| Interruptions | Log pauses, browser focus loss, interruptions, and technical issues |
| Human interviewer | Use separate audio channels when possible |
| Self-guided mode | Prefer for MVP consistency and lower operational variance |

### 4. Post-Session Export

| Step | Required Behavior |
| --- | --- |
| QC summary | Generate video, audio, transcript, and session QC summary |
| Checksums | Generate checksums for exported artifacts |
| Package validation | Confirm required artifacts exist before upload or analysis |
| Raw media policy | Apply configured raw-media retention rule |
| Access log | Record raw media access if raw media is stored |
| Exclusion decision | Store usable, needs review, or exclude status with reason |

## Quality-Control Requirements

### Video QC

| Check | MVP Rule |
| --- | --- |
| Face presence | Face detected in at least 90-95% of usable frames |
| FPS stability | Actual FPS remains close to target 30 fps |
| Camera freezes | No long frozen segments |
| Face size | Face is not too small or too large for landmark tracking |
| Face position | Face remains centered enough for stable tracking |
| Head turn | Head is not excessively turned away during relevant tasks |
| Brightness | Brightness is sufficient for face detection |
| Backlight | No strong backlight that hides facial details |
| Landmark confidence | Landmark confidence remains acceptable |
| Gaze tracking | Gaze or face tracking is valid during relevant tasks |
| Dropped frames | Dropped frame count is captured and thresholded |
| Bounding box stability | Face bounding box is stable across frames |
| Missing frames | Missing frames are counted and reported |

### Audio QC

| Check | MVP Rule |
| --- | --- |
| Microphone detection | Microphone detected before session starts |
| Mute detection | Reject mute or near-silent recordings |
| Clipping | Reject or flag clipped audio |
| Background noise | Flag excessive background noise |
| Speech duration | Require enough participant speech duration |
| Ultra-short answers | Flag ultra-short responses |
| VAD ratio | Voice activity ratio must be within expected bounds |
| Speaker separation | Require channel separation or diarization when interviewer exists |
| Diarization confidence | Store diarization confidence for each speaker segment |
| Transcript quality | Store ASR confidence or manual verification status |

### Session QC

| Check | MVP Rule |
| --- | --- |
| Required tasks | All required tasks completed |
| Baseline | Baseline captured and passed core QC |
| Timestamp sync | Audio, video, prompts, and feature timestamps synchronized |
| Interruptions | Major interruptions logged |
| Technical issues | Browser, device, network, and permission issues logged |
| Checksums | Exported artifacts have checksums |
| Package validation | Export package passes validation |
| Pseudonymization | Participant ID pseudonymized |
| Raw access | Raw media access logged when raw media exists |
| Exclusion status | Session marked usable, review, or excluded |

## Feature Requirements

### Video and Face Features

| Feature | Reason |
| --- | --- |
| Face presence rate | Confirms usable face data coverage |
| Facial landmarks | Base geometry for facial movement and expressivity |
| Head pose | Supports psychomotor and engagement indicators |
| Gaze direction | Supports attention and social engagement indicators |
| Blink dynamics | Supports blink rate and related temporal patterns |
| Action units | Supports facial affect and expression analysis |
| Emotion probabilities | Supports emotional expressivity summaries |
| Facial expressivity | Captures movement magnitude and variation |
| Mouth openness | Supports speaking and expressivity features |
| Frame-to-frame movement | Captures motion and stability |
| Detection confidence | Enables frame-level reliability scoring |
| Missing frames | Identifies gaps in measurement |
| Dropped frames | Identifies capture instability |
| Face bounding box stability | Detects framing and tracking problems |

### Audio and Speech Features

| Feature | Reason |
| --- | --- |
| Speech rate | Core speech biomarker |
| Pauses | Captures latency, hesitation, and silence structure |
| Articulation rate | Measures speech production independent of pauses |
| Pitch / F0 | Core prosodic feature |
| Energy | Captures loudness and vocal activation |
| Voice quality | Supports acoustic health and affect indicators |
| Formants | Supports vocal tract and articulation analysis |
| Spectral features | Supports broad acoustic modeling |
| Voiced/unvoiced ratio | Captures phonation structure |
| Interviewer speaking time | Allows participant-only analysis |
| Participant speaking time | Core denominator for speech features |
| Interruptions | Captures protocol and interaction artifacts |
| Silence ratio | Measures silence and disengagement patterns |
| Transcript quality | Helps filter unreliable language features |
| Diarization quality | Protects participant-only language analysis |

### Text and Language Features

| Feature | Reason |
| --- | --- |
| Participant-only transcript | Prevents interviewer language from contaminating features |
| Speaker role labels | Enables participant/interviewer separation |
| Lexical richness | Supports language complexity analysis |
| Sentiment | Captures affective language signals |
| First-person pronouns | Relevant in depression and anxiety language studies |
| POS features | Supports syntactic analysis |
| Semantic coherence | Captures organization of speech |
| Repetitions | Captures perseveration or disfluency patterns |
| Tangentiality | Captures off-topic or disorganized responses |

## Metadata-First Storage Design

AIREST should store metadata at the same priority as derived features. Metadata
must explain whether a feature can be trusted, how it was generated, and which
capture conditions may have influenced it.

### Session Metadata

| Metadata Field | Required Detail |
| --- | --- |
| Participant ID | Pseudonymized identifier |
| Session ID | Unique session identifier |
| Site or deployment | Local, clinic, study, or remote setting |
| Consent status | Consent captured before recording |
| Protocol version | Version of prompt set and capture rules |
| Session start/end | Wall-clock timestamps and timezone |
| Browser and OS | Browser name/version and operating system |
| Device type | Laptop, desktop, tablet, or other |
| Completion status | Completed, partial, failed, or excluded |

### Video Metadata

| Metadata Field | Required Detail |
| --- | --- |
| Camera device label | Browser-provided or normalized camera label when allowed |
| Resolution | Actual width and height |
| FPS target | Requested FPS |
| FPS observed | Measured FPS across session |
| Codec/container | Media codec and container when raw or temporary media exists |
| Lighting flags | Low light, backlight, unstable light |
| Face detection rate | Percent of usable frames with detected face |
| Landmark confidence | Aggregate and frame-level confidence where available |
| Dropped frames | Count and percentage |
| Frozen segments | Start/end times and duration |
| Bounding box stats | Size, position, and stability summaries |

### Audio Metadata

| Metadata Field | Required Detail |
| --- | --- |
| Microphone device label | Browser-provided or normalized microphone label when allowed |
| Sample rate | Actual sample rate |
| Bit depth | Actual bit depth when available |
| Channels | Mono, stereo, or separated speaker channels |
| Format | WAV or other captured/exported format |
| Clipping rate | Count or percent of clipped samples |
| Noise estimate | Pre-session and in-session noise level |
| Speech duration | Participant and total speech duration |
| VAD ratio | Speech/non-speech ratio |
| Diarization confidence | Confidence by segment and overall |
| ASR confidence | Transcript confidence and review status |

### Protocol and Event Metadata

| Metadata Field | Required Detail |
| --- | --- |
| Prompt IDs | Versioned IDs for each question or task |
| Prompt timing | Start/end timestamps for prompt display |
| Response timing | Start/end timestamps for participant response |
| Baseline timing | Start/end timestamps for neutral baseline |
| Interruptions | Type, timestamp, duration, and severity |
| Browser events | Focus loss, permission loss, refresh, crash |
| Retry events | Which task was retried and why |
| Exclusion reason | Explicit reason if session or segment is excluded |

## Privacy and Raw Media Policy

The default MVP design should minimize raw video retention unless the study,
ethics approval, or regulatory plan explicitly requires raw media.

| Artifact | Default MVP Policy |
| --- | --- |
| Raw video | Do not store by default when minimization is required |
| Raw audio | Store only when allowed and needed for speech processing or audit |
| Derived video features | Store by default |
| Frame-level QC | Store by default |
| Audio features | Store by default |
| Participant-only transcript | Store by default when generated under consent |
| Full transcript with interviewer | Store only if allowed and role-labeled |
| QC reports | Store by default |
| Device metadata | Store by default with privacy review |
| Access logs | Required for raw media access |

If raw video is not retained, the system should still keep enough derived data
to audit feature quality: face detection confidence, frame counts, FPS stats,
missing frame counts, bounding box summaries, and feature extraction errors.

## Recommended Export Package Shape

This is a conceptual package structure for future implementation. It is shown
as a directory tree, not as JSON.

```text
session_export/
  README.md
  checksums.txt
  session_metadata.md
  protocol_events.md
  qc_report.md
  features/
    video_face_features.csv
    audio_speech_features.csv
    text_language_features.csv
    feature_dictionary.md
  transcripts/
    participant_only_transcript.txt
    diarized_transcript.txt
    transcript_qc.md
  media/
    audio.wav
    video_optional_retention.mp4
  logs/
    processing_log.txt
    raw_media_access_log.md
```

## MVP Acceptance Criteria

An MVP session is considered technically acceptable when all conditions below
are met.

| Area | Acceptance Criterion |
| --- | --- |
| Protocol | Browser-based standardized flow completed |
| Baseline | Neutral baseline captured and passed core QC |
| Video | 720p/30 fps target or at least 640 x 480/30 fps minimum |
| Face tracking | Face detected in at least 90% of usable frames |
| Audio | Microphone detected, not muted, no major clipping |
| Speech | Enough participant speech duration for feature extraction |
| Speaker roles | Participant speech separable from interviewer speech |
| Metadata | Device, capture, QC, protocol, and processing metadata saved |
| QC report | Video, audio, transcript, and session QC generated |
| Privacy | Raw media retention follows configured ethics/regulatory policy |
| Export | Checksums generated and package validation passed |

## Out of Scope for MVP

| Item | Reason |
| --- | --- |
| Multi-camera capture | Useful for research, but not required for face/head/gaze MVP |
| Full clinical diagnostic workflow | MVP focuses on data capture and biomarker extraction readiness |
| Raw video retention by default | Conflicts with minimization strategy unless explicitly approved |
| Free-form interviewer workflow | Reduces repeatability and makes measurement less standardized |
| Manual-only QC | Scalable capture requires automatic QC gates |

## Important Literature Limitation

Many papers describe extracted features and model performance in detail but do
not fully document camera model, lens, lighting, room geometry, codec,
microphone model, distance to participant, or exact capture setup. This weakens
reproducibility.

AIREST should explicitly document these technical parameters even when source
literature omits them.

## Practical Conclusion

The AIREST MVP should implement a clinical measurement pipeline with:

1. A repeatable browser-based interview protocol.
2. Stable 720p/30 fps webcam capture with 640 x 480/30 fps as the minimum.
3. External or headset microphone as the preferred audio source.
4. Required neutral baseline capture.
5. Automatic QC before, during, and after the session.
6. Participant-only transcript protection through diarization and role labels.
7. Metadata-first storage for capture conditions, confidence, errors, and interruptions.
8. Privacy-aware raw media minimization with retained derived features and QC artifacts.

This protocol should be treated as an MVP baseline. Any future research or
clinical workflow should update this README when thresholds, devices, consent
rules, feature definitions, or retention policies change.
