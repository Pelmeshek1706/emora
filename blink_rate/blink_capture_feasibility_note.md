# Blink Capture Feasibility Note for AIREST

## Scope

This note answers three questions for production use in AIREST:

1. Is a webcam stream around 30 fps sufficient for stable blink detection?
2. Does blink extraction conflict with gaze tracking or calibration?
3. What risks should be recorded for low-FPS sessions?

Context provided by user:
- webcam target: `1080p`
- nominal frame rate: `30 fps`
- optional comparison target: `60 fps`

## Repo facts

The current blink path in this workspace is OpenWillis `eye_blink_rate(video)`.

Relevant implementation details:
- The detector reads the video FPS from OpenCV and uses that FPS directly for event timing and blink-rate calculation: [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:151), [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:286), [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:418)
- Every frame is processed. There is no frame skipping in the blink pipeline: [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:335)
- Each frame is resized to width `450` before MediaPipe FaceMesh runs: [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:343)
- Blink detection is based on troughs in a z-scored EAR trace using `scipy.signal.find_peaks`: [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:236), [eye_blink.py](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/eye_blink.py:360)
- Detection config is lightweight and does not impose a meaningful minimum blink duration beyond the sampled frames themselves: [eye.json](/Users/pelmeshek1706/Desktop/projects/airest-face/openwillis-face/src/openwillis/face/config/eye.json:4)

Important architectural finding:
- In this repo and in the adjacent AIREST repos I checked, there is no built-in gaze-tracking or gaze-calibration module. Blink extraction exists; gaze tracking does not currently exist as an integrated pipeline component here.

## What that means for expected FPS in AIREST

There is no explicit FPS requirement enforced by code.

The practical expectation is:
- AIREST blink extraction will accept whatever FPS is encoded in the recorded video.
- Timing accuracy depends on the delivered frame rate actually being stable and correctly reported.
- Because the algorithm uses frame count divided by FPS, variable-rate or dropped-frame recordings are a real production risk.

So the correct engineering answer is not "AIREST expects exactly 30 fps."
It is:
- AIREST currently assumes a reasonably constant frame rate.
- `30 fps` is acceptable for blink count/rate and coarse event timing.
- `60 fps` is preferable when available for stability and better onset/offset timing.

## Feasibility assessment

### 30 fps

`30 fps` gives a frame every `33.3 ms`.

Published blink timing ranges show that:
- a typical blink lasts on the order of `100-400 ms`
- a common spontaneous blink duration estimate is around `100-150 ms`
- full blink duration is often described as `150-400 ms`

At `30 fps`, that yields roughly:
- `3-12` frames across a full blink
- often only `3-5` frames for a short spontaneous blink

That is generally enough for:
- blink presence or absence
- blink counts
- blink rate
- coarse peak timing
- approximate blink duration

That is not strong enough for:
- precise onset/offset timing
- reliable capture of very short or incomplete blinks
- blink kinematics such as closing velocity, reopening velocity, or pause-phase duration

Conclusion for `30 fps`:
- Feasible for production blink extraction if the goal is blink count/rate and coarse event timing.
- Borderline for fine blink dynamics.
- Vulnerable to missed or distorted brief closures when sessions are noisy or frame delivery is unstable.

### 60 fps

`60 fps` gives a frame every `16.7 ms`.

That roughly doubles the temporal sampling:
- `6-24` frames across a `100-400 ms` blink

This materially improves:
- blink onset/offset stability
- recovery of short closures
- incomplete-blink separation
- robustness when a few frames are lost
- compatibility with any future shared use of the same video for webcam gaze estimation

Conclusion for `60 fps`:
- Clearly better than `30 fps` for blink capture.
- Still not a high-speed blink-dynamics setup.
- Strongly preferred if the actual delivered frame rate is truly `60`, not duplicated or interpolated frames.

## Feasibility assessment in plain language

The simplest way to think about this is:

- `30 fps` means the camera gives AIREST `30 snapshots each second`.
- A blink is quick, but not instant. Many normal blinks last around `0.1-0.4 seconds`.
- So at `30 fps`, AIREST usually sees a blink in a handful of snapshots.

That is why `30 fps` is usually good enough to answer:
- "Did the person blink?"
- "How many times did they blink?"
- "Was their blink rate roughly low, normal, or high?"

But `30 fps` is not good enough to answer very fine questions like:
- "Exactly when did the eyelid start closing?"
- "How long was the fully closed phase?"
- "How fast did the eyelid close and reopen?"

### Simple examples

Example 1: normal blink at `30 fps`

- Say a blink lasts about `150 ms`.
- At `30 fps`, one frame arrives every `33.3 ms`.
- That blink is visible in about `4-5` frames.
- In practice, that is usually enough for AIREST to count it correctly.

Example 2: short blink at `30 fps`

- Say the blink is closer to `100 ms`.
- That is only about `3` frames at `30 fps`.
- If one frame is blurry because the head moves, or one frame is dropped, the detector may still see the blink but the timing becomes rough.
- In a worse case, it may shorten the blink or miss it.

Example 3: the same blink at `60 fps`

- The same `150 ms` blink now appears in about `9` frames.
- That gives the algorithm more chances to see the eye closing, the lowest point, and the eye reopening.
- So `60 fps` does not magically create a medical-grade blink lab, but it makes detection noticeably more stable.

### One-sentence summary

For AIREST, `30 fps` is like having "enough photos to count blinks."
`60 fps` is like having "enough photos to count blinks more confidently and time them better."

## Practical FPS bands for AIREST

These are engineering recommendations for production use. They are not hard-coded repo limits.

### Minimum acceptable

- `25-30 fps`, delivered steadily

What this means:
- Below about `25 fps`, blink timing gets coarse quickly and short blinks become easier to miss.
- Around `25-30 fps`, AIREST can still do blink count and blink-rate extraction, but this is the lower safety band, not the comfort zone.

### Recommended

- `50-60 fps`, with `60 fps` preferred

What this means:
- This is the best practical range for the current AIREST blink use case.
- It improves onset/offset stability and reduces the chance that a fast blink disappears into only `2-3` useful frames.

### Maximum that is actually needed for current AIREST use

- `60 fps`

Why:
- For the current AIREST goal, the product value is blink presence, count, rate, and coarse timing.
- For those outputs, moving above `60 fps` usually gives diminishing returns compared with improving lighting, face framing, and capture stability.

Important exception:
- If you later want research-grade blink dynamics such as closing velocity, reopening velocity, or very precise closed-phase timing, then `100-250 fps` becomes relevant in the literature.
- That is beyond the current AIREST need.

## Practical FPS variation and jitter

The nominal FPS printed on the webcam box is not enough. What matters is the FPS that actually arrives during the session.

Recommended stability rule:
- aim to keep delivered FPS within about `±10%` of target for most of the recording

Examples:
- for a nominal `30 fps` stream, aim for roughly `27-33 fps`
- for a nominal `60 fps` stream, aim for roughly `54-66 fps`

Warning thresholds:
- if a `30 fps` session repeatedly drops below about `25 fps`, treat blink timing as low-confidence
- if a `60 fps` session repeatedly drops below about `50 fps`, much of the benefit over `30 fps` is lost

In human terms:
- stable `30 fps` is better than "fake 60 fps" with repeated frames, pauses, or bursts of dropped frames
- if you must choose, pick the mode that stays steady

## Does blink extraction conflict with gaze tracking or calibration?

### In the current codebase

No direct conflict is present, because there is no integrated gaze-tracking or gaze-calibration module in the current AIREST code I reviewed.

Blink extraction is an offline video-analysis pass. By itself it does not degrade gaze tracking because it is not sharing a live gaze pipeline here.

### In a future combined webcam pipeline

There is no fundamental feature conflict between blink extraction and gaze tracking.

The conflict is operational:
- blinks temporarily remove valid eye information for gaze estimation
- blinks during calibration reduce calibration sample quality
- low FPS makes those invalid windows harder to detect and mask cleanly
- heavy real-time processing can reduce delivered FPS and hurt both tasks at once

The practical rule is:
- blink extraction and gaze tracking can coexist on the same recording
- but gaze calibration should exclude blink-contaminated frames
- and gaze estimates during closed-eye or near-blink windows should be treated as invalid or low-confidence

## Production judgment for your webcam

For a webcam that truly delivers `1080p, 30 fps`:
- blink capture is technically feasible for AIREST
- stable blink count/rate extraction is realistic
- coarse blink event timing is realistic
- high-fidelity blink dynamics are not realistic

If the same webcam can truly deliver `60 fps`:
- use `60 fps` for blink capture
- if `60 fps` requires dropping from `1080p` to `720p`, that is often still a good trade for blink and gaze tasks, because the current blink code downsamples frames to width `450` anyway

The most important variable is not nominal `1080p`.
It is:
- actual delivered FPS
- face size in the frame
- lighting stability
- head pose stability

Because frames are resized to width `450`, higher raw resolution helps only indirectly. For this detector, face visibility and temporal stability matter more than keeping the full frame at `1080p`.

## Known risks for low-FPS sessions

Record these as explicit production risks.

### Risk 1: missed short blinks or incomplete blinks

At `30 fps`, a short blink may occupy only a few frames. Very brief closures can be under-sampled, merged, or missed.

Impact:
- undercounted blinks
- unstable duration estimates
- weak inter-blink interval accuracy

### Risk 2: poor timing granularity

Blink start, peak, and end times are quantized by frame period.

Impact:
- about `33 ms` timing granularity at `30 fps`
- about `17 ms` timing granularity at `60 fps`
- onset and offset timing are visibly noisier at `30 fps`

### Risk 3: pause phase often not captured at 30 fps

Published blink-kinematics work notes that the fully closed pause phase can be very short, around `13.7 +/- 3.3 ms` in one analysis, which is below a `30 fps` frame interval.

Impact:
- full-closure timing is unreliable
- kinematic interpretation is unreliable

### Risk 4: variable frame rate or dropped frames

The current OpenWillis path assumes a stable FPS and converts frame index to time by dividing by FPS.

Impact:
- wrong blink times if recording is variable-rate
- wrong blink rate if frame pacing is inconsistent
- hard-to-detect bias if browser capture silently drops frames under load

### Risk 5: low FPS amplifies gaze-calibration fragility

When gaze calibration is added, blink-contaminated frames are harder to reject cleanly at low sample rates.

Impact:
- noisier point acceptance
- more calibration retries
- worse spatial accuracy after calibration

### Risk 6: lighting and motion failures look worse at 30 fps

Blink detection depends on stable eye landmarks. Low light, motion blur, glasses glare, and head turns hurt MediaPipe eye landmarks more when fewer frames are available.

Impact:
- false troughs in EAR
- missing troughs
- more NaNs or face-loss windows

### Risk 7: nominal 60 fps may be fake 60 fps

Some webcam/browser paths expose `60 fps` nominally but deliver repeated frames or jitter under CPU or USB bandwidth pressure.

Impact:
- no real timing gain
- false confidence in temporal accuracy

## Recommendation

For AIREST production:

- `30 fps` is sufficient for blink count and blink rate extraction.
- `30 fps` is acceptable for coarse blink event timing.
- `30 fps` is not sufficient for trustworthy blink kinematics.
- `60 fps` should be preferred whenever the camera and capture stack truly deliver it.

Recommended operating target:
- minimum acceptable: stable `25-30 fps`
- preferred: stable `60 fps`
- maximum practically needed for current AIREST blink goals: stable `60 fps`

Recommended policy if gaze tracking/calibration is added:
- reject or mask blink windows for gaze estimation
- do not accept calibration samples during or immediately around a blink
- store actual frame timestamps if possible, instead of relying only on nominal FPS
- treat sessions below roughly `24-25 fps` or with obvious frame jitter as low-confidence for blink timing

## Bottom line

Blink capture in AIREST is technically feasible with your `1080p, 30 fps` webcam for production use if the product goal is blink count/rate and coarse blink timing.

If `60 fps` is available and real, it is the better operating mode.

Blink extraction does not inherently conflict with gaze tracking, but low FPS makes shared webcam use less robust because blinks create invalid gaze samples and can degrade calibration if those frames are not explicitly filtered.

## External references

### Widely cited background references

Citation counts below are from OpenAlex, checked on `2026-05-15`.

- `3214` citations: TFOS DEWS II Definition and Classification Report. Useful background on blink relevance and ocular-surface context. [DOI](https://doi.org/10.1016/j.jtos.2017.05.008)
- `944` citations: TFOS DEWS II Tear Film Report. Useful background on tear film and why blink behavior matters physiologically. [DOI](https://doi.org/10.1016/j.jtos.2017.03.006)
- `265` citations: Eye Movement and Pupil Measures: A Review. Good broad review of eye-movement measurement limits and interpretation. [DOI](https://doi.org/10.3389/fcomp.2021.733531)
- `224` citations: WebGazer: Scalable webcam eye tracking using user interactions. Foundational webcam-gaze paper. [PDF](https://cs.brown.edu/people/apapouts/papers/ijcai2016webgazer.pdf)
- `200` citations: Online webcam-based eye tracking in cognitive science: A first look. Widely cited early evaluation of webcam eye tracking in research use. [DOI](https://doi.org/10.3758/s13428-017-0913-7)
- `229` citations: Accelerating eye movement research via accurate and affordable smartphone eye tracking. Not webcam-specific, but useful high-citation reference for low-cost camera-based eye tracking. [DOI](https://doi.org/10.1038/s41467-020-18360-5)

### More direct but newer references

- `51` citations: Webcam eye tracking close to laboratory standards: Comparing a new webcam-based system and the EyeLink 1000. Directly relevant to `~30 Hz` webcam gaze tracking limits. [DOI](https://doi.org/10.3758/s13428-023-02237-8)
- `38` citations: Deep learning models for webcam eye tracking in online experiments. Directly relevant to blink plus gaze estimation from webcam recordings. [DOI](https://doi.org/10.3758/s13428-023-02190-6)
- `14` citations: Towards efficient calibration for webcam eye-tracking in online experiments. Directly relevant to calibration burden and low-sampling-rate online setups. [DOI](https://doi.org/10.1145/3517031.3529645)

### Blink-specific references used for the timing claims

- MediaPipe Face Mesh real-time landmarking overview: [mediapipe.readthedocs.io](https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html)
- Blink kinematics review with `75-400 ms` blink duration and pause-phase discussion: [Sensors 2019](https://www.mdpi.com/1424-8220/19/5/1121)
- Blink literature noting `25-30 fps` is sufficient for endogenous blink detection: [Pattern Recognition Letters 2016 abstract](https://www.sciencedirect.com/science/article/pii/S1077314216300054)
- Blink-completeness literature noting `30 fps` is sufficient to capture a fully closed eye, while `15 fps` is problematic: [Pattern Recognition Letters 2018 abstract](https://www.sciencedirect.com/science/article/abs/pii/S107731421830287X)
- Recent blink-dynamics study showing low frame rates are enough for amplitude and duration but not velocity metrics: [The Ocular Surface 2026 abstract](https://research.manchester.ac.uk/en/publications/video-frame-rate-influences-the-accuracy-of-blink-dynamics-measur/)

### Method note

The current AIREST blink path uses the EAR-style landmark method implemented in OpenWillis. That method traces back to Soukupová and Čech's blink-detection work, which is the conceptual basis for many later EAR implementations, even though that specific source is not cleanly citation-indexed in OpenAlex in the same way as the journal papers above.
