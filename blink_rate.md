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

## Research notes for AIREST clinical relevance

This section summarizes the user-provided research review:
- [Blink Rate and Its Relationship to Detecting PTSD, Major Depressive Disorder, and Anxiety Disorders.pdf](/Users/pelmeshek1706/Desktop/projects/airest-face/Blink%20Rate%20and%20Its%20Relationship%20to%20Detecting%20PTSD,%20Major%20Depressive%20Disorder,%20and%20Anxiety%20Disorders.pdf)

Bottom-line interpretation for AIREST:
- Blink rate is clinically interesting and worth keeping as an AIREST feature.
- The current literature does not support blink rate as a stand-alone diagnostic biomarker for PTSD, major depressive disorder, or anxiety disorders.
- The strongest direct evidence is for depression-spectrum conditions.
- PTSD and anxiety signals appear more context-dependent and currently less replicated.
- The best future use is as a low-cost auxiliary feature inside a multimodal prediction system, not as a single screening rule.

## Why this feature matters

The review supports keeping blink in AIREST for three reasons:

1. Blink rate has disorder-relevant signal.
- Depression studies repeatedly found blink-related abnormalities in at least some populations and tasks.
- PTSD work suggests blink behavior can change under threat/inhibitory-control conditions.
- Panic-disorder work suggests blink rate may increase at rest and during emotionally loaded audiovisual stimulation.

2. Blink is cheap to capture.
- Blink can be derived from ordinary webcam video.
- That makes it practical for remote or scalable psychiatric screening workflows.

3. Blink is complementary to other modalities.
- The review argues that blink is most useful when combined with other ocular, facial, speech, and self-report features.
- This matches AIREST better than a stand-alone biomarker framing.

4. Blink already has early prediction evidence in structured settings.
- The review summarizes engineering studies where broader eye-movement features reached about `70-75%` accuracy for depression classification.
- It also summarizes blink-feature models reporting about `88-92%` task-specific accuracy in structured interview-style tasks.
- Those numbers support feature importance, but not clinical readiness, because the reviewed reports do not establish usable thresholds, sensitivity/specificity, or strong external validation.

## Disorder-specific research relevance

### Major depressive disorder

Depression is the most promising current use case in the reviewed literature.

Key findings summarized in the review:
- Mackintosh et al. (1983, Br J Psychiatry, DOI `10.1192/BJP.143.1.55`) reported higher blink rate in depressed patients, with blink rate falling toward normal during treatment.
- Ebert et al. (1996, Neuropsychopharmacology, DOI `10.1016/0893-133X(95)00237-8`) did not find a baseline blink-rate difference in non-retarded, drug-naive MDD, but did find that sleep deprivation increased blink rate in patients and that the increase tracked short-term improvement.
- Lee et al. (2024, BMC Geriatrics, DOI `10.1186/s12877-024-05034-w`) found higher blink rate in late-life depression than in healthy older and younger controls, with positive correlations to symptom burden measures.
- Lee et al. (2025, J Geriatr Psychiatry Neurol, DOI `10.1177/08919887251334999`) suggests blink abnormalities may also depend on emotional-conflict task context rather than appearing as a simple tonic group difference.

Interpretation for AIREST:
- Blink features are worth retaining for depression prediction work.
- The likely value is not only raw blink rate, but also blink behavior under specific task conditions.
- Depression-related blink signal may reflect arousal, psychomotor state, autonomic tone, symptom burden, or treatment-responsive state change rather than a single stable trait marker.

### PTSD

Direct PTSD blink-rate evidence is much thinner.

Key finding summarized in the review:
- Rubin et al. (2017, Brain Sciences, DOI `10.3390/brainsci7020016`) found that women with PTSD showed higher spontaneous blink rates mainly under low-threat conditions during an inhibitory-control task, not a universal increase across all contexts.

Interpretation for AIREST:
- Blink may still be useful for PTSD-related prediction, but probably only when paired with the right eliciting context.
- Passive or resting blink rate alone may miss much of the signal.
- If AIREST later adds task-evoked ocular analysis, PTSD may be a stronger future target than if blink is captured only as a general background measure.

### Anxiety disorders

The direct anxiety literature is weak and narrow.

Key finding summarized in the review:
- Kojima et al. (2002, Psychiatry Clin Neurosci, DOI `10.1046/j.1440-1819.2002.01052.x`) reported higher blink rates in a small panic-disorder pilot at rest and during audiovisual stimulation.

Interpretation for AIREST:
- Blink has some future relevance for anxiety-spectrum prediction, especially panic-related phenotypes.
- But the current evidence is too sparse to treat blink as an established anxiety marker.
- The review did not find a strong replicated blink-rate case-control literature for generalized anxiety disorder or social anxiety disorder.

## What this means for future predictive use in AIREST

The reviewed research supports a conservative roadmap:

- Near-term use:
  Keep blink rate as a supportive feature in AIREST feature sets.

- Strongest future prediction target:
  Depression, especially when blink is combined with other ocular or behavioral features.

- Possible but less mature future target:
  PTSD, especially if AIREST uses standardized cognitive-control or emotional-conflict tasks rather than only passive observation.

- Weakest current target:
  Broad anxiety-disorder detection, because the literature is still too thin.

In practical model terms, the review supports:
- multimodal prediction
- task-sensitive modeling
- quality-controlled capture
- person-level validation
- cautious interpretation of internal task-specific accuracy reports

It does not support:
- one universal blink-rate cutoff for psychiatric diagnosis
- using blink rate alone for screening decisions
- treating blink rate as a disorder-specific marker without task context

## Engineering implications for AIREST

The review is directly relevant to what AIREST should store and derive.

Recommended AIREST blink feature set:
- blink count
- blink rate
- inter-blink intervals
- blink duration
- blink variability / burstiness
- task-modulated blink change if AIREST has multiple task segments
- extraction quality flags for poor lighting, occlusion, glasses glare, or face-loss windows

Reason:
- The review repeatedly suggests that the clinically useful signal is richer than a single `blinks/minute` number.
- Engineering studies in the review also indicate that broader feature sets outperform simple raw blink rate alone.

Important limitation for current OpenWillis output:
- `eye_blink_rate()` already gives count, rate, and event timing.
- It does not directly provide inter-blink variability, burstiness, or explicit duration columns.
- Those should be derived downstream in AIREST from the blink event table.

## Research caution for product claims

AIREST should avoid overstating this feature.

Claims that are supported by the review:
- Blink is a relevant psychophysiological feature.
- Blink has the strongest direct psychiatric evidence in depression.
- Blink may add value to multimodal prediction systems.
- Blink behavior is sensitive to task context and recording conditions.

Claims that are not currently supported:
- Blink rate alone can diagnose PTSD, MDD, or anxiety disorders.
- There is a validated clinical blink threshold with known sensitivity/specificity for these disorders.
- Webcam blink rate from ordinary short clips is already clinically validated for psychiatric detection.

## Suggested AIREST wording

If AIREST needs a short internal framing:

- Blink rate should be treated as a low-cost, clinically relevant auxiliary feature with the strongest current evidence in depression, emerging task-sensitive relevance for PTSD, and only preliminary support in anxiety disorders.
- Future psychiatric prediction work in AIREST should use blink as one component of a broader ocular and behavioral biomarker panel rather than as a stand-alone detector.
