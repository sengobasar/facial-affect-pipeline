# Real-Time Multimodal Affective Computing System for Emotional Distress Detection using Facial and Vocal Signals

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00897B?logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.75+-FF6F00)]()
[![LibROSA](https://img.shields.io/badge/LibROSA-0.9+-orange)]()
[![Status](https://img.shields.io/badge/Status-Prototype-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A real-time multimodal affective computing system that detects emotional distress from **facial expressions and voice prosody** using dimensional emotion modeling, temporal tracking, and behavioral analysis.
>
> <img width="1917" height="984" alt="image" src="https://github.com/user-attachments/assets/ccc9447d-7e41-4a67-82e8-7631169907df" />


---

## What This System Does

This system processes **live webcam video and microphone audio simultaneously** to:

1. Detect faces and extract facial landmarks
2. Classify facial expressions into emotion categories
3. Extract voice features — MFCC, pitch, jitter, shimmer, and energy
4. Map both face and voice signals to **valence-arousal** dimensional space
5. Compute **distress scores** from each modality independently
6. Fuse face, voice, and behavioral features into a unified representation
7. Predict mental state (distressed vs. normal) with a logistic classifier
8. Assess risk levels and emotional trends in real-time

**Key Innovation:** Unlike basic emotion recognition systems, this implements **multimodal dimensional emotion modeling** — combining continuous valence-arousal representations from both facial expressions and speech prosody, rather than relying on discrete labels from a single modality.

---

## Research Context

This system is inspired by dimensional models of affect, particularly the **valence-arousal circumplex** framework where emotions are represented in continuous 2D space rather than discrete categories.

**Primary reference:**

**Mollahosseini, A., Hasani, B., & Mahoor, M.H. (2019).** *AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild.* IEEE Transactions on Affective Computing, Vol. 10, No. 1, pp. 18–31. [[paper]](https://arxiv.org/abs/1708.03985)

The AffectNet paper demonstrates that:
- Emotions can be modeled in continuous dimensional space (valence and arousal)
- Valence represents how positive/negative an emotion is
- Arousal represents the intensity/activation level
- This approach captures subtle emotional variations that discrete categories cannot

**This implementation extends beyond the paper by adding:**
- Real-time distress modeling from both face and voice
- Voice prosody analysis (MFCC, pitch, jitter, shimmer, energy)
- Multimodal feature fusion across visual and acoustic channels
- Temporal emotional tracking
- Behavioral micro-feature extraction (eye/mouth openness)
- Risk assessment and trend detection
- Mental state classification

---

## Mathematical Framework

### 1. Emotion Probability Distribution (Face)

DeepFace outputs a probability distribution over 7 emotion categories:

```
P(emotion) = {happy, sad, angry, fear, surprise, disgust, neutral}
```

Constraint:
```
Σ P(eᵢ) = 1
```

---

### 2. Facial Valence

Valence is computed as a weighted expectation over emotion probabilities:

```
Vf = Σ P(eᵢ) · wv(eᵢ)
```

**Valence weights:**
```
happy     → +1.0
surprise  → +0.4
neutral   →  0.0
sad       → -0.7
fear      → -0.8
anger     → -0.9
disgust   → -0.95
```

`Vf ∈ [-1, 1]` — positive values indicate positive emotions, negative values indicate distress.

---

### 3. Facial Arousal

```
Af = Σ P(eᵢ) · wa(eᵢ)
```

**Arousal weights:**
```
fear      → +0.9
anger     → +0.8
surprise  → +0.7
happy     → +0.6
neutral   →  0.0
sad       → -0.3
disgust   → -0.2
```

`Af ∈ [-1, 1]` — positive values indicate high activation, negative values indicate low activation.

---

### 4. Facial Distress Score

```
Df = (1 - Vf) · |Af|
```

**Examples:**
```
Fear (Vf=-0.8, Af=0.9):  Df = (1-(-0.8)) · 0.9 = 1.62  (high distress)
Happy (Vf=0.9, Af=0.6):  Df = (1-0.9) · 0.6 = 0.06     (low distress)
Neutral (Vf=0, Af=0):    Df = (1-0) · 0 = 0.0           (no distress)
```

---

### 5. Temporal Smoothing (Face)

A sliding window of the last `N` observations stabilizes predictions:

```
V̄f = mean(Vf₁, Vf₂, ..., Vfₙ)
Āf = mean(Af₁, Af₂, ..., Afₙ)
D̄f = mean(Df₁, Df₂, ..., Dfₙ)
```

Default window size: `N = 30` frames (~1 second at 30 FPS).

---

### 6. Behavioral Features

**Eye Aspect Ratio (EAR):**
```
        vertical_distance(upper_eyelid, lower_eyelid)
EAR = ──────────────────────────────────────────────────
      horizontal_distance(left_corner, right_corner)
```

**Mouth Aspect Ratio (MAR):**
```
        vertical_distance(upper_lip, lower_lip)
MAR = ────────────────────────────────────────────
      horizontal_distance(left_corner, right_corner)
```

---

### 7. Speech Signal Representation

Human speech is digitized as a discrete-time signal:

```
x[n]
```

where `n` = sample index and sampling rate `fs = 16,000 Hz`.

---

### 8. Short-Time Framing

Speech is analyzed in short overlapping frames:

```
Frame length: 25 ms
Frame step:   10 ms
Frame k:      xk[n]
```

---

### 9. Windowing

Each frame is multiplied by a Hamming window to reduce spectral leakage:

```
xw[n] = x[n] · w[n]

w[n] = 0.54 - 0.46 · cos(2πn / (N-1))
```

---

### 10. Fourier Transform

Frames are transformed from time → frequency domain:

```
X(k) = Σ x[n] · e^(-j2πkn/N)   for n = 0 to N-1
```

This produces the **power spectrum** of each frame.

---

### 11. Mel Filterbank

Human hearing is non-linear. Frequency is mapped to the Mel scale:

```
mel(f) = 2595 · log10(1 + f/700)
```

Triangular Mel filters `Hm(k)` are applied to the power spectrum. Energy per filter:

```
Em = Σ |X(k)|² · Hm(k)
```

---

### 12. MFCC (Mel Frequency Cepstral Coefficients)

MFCCs compress the Mel spectrum using the Discrete Cosine Transform:

```
MFCCn = Σ log(Em) · cos[πn/M · (m - 0.5)]   for m = 1 to M
```

The system extracts **13 MFCC coefficients** capturing vocal tract shape and speech timbre.

---

### 13. Pitch (Fundamental Frequency)

```
F0 = 1 / T
```

where `T` = pitch period (vocal fold vibration cycle).

| Pitch Behavior | Emotional Interpretation |
|---|---|
| High pitch | Excitement / stress |
| Low pitch | Sadness / fatigue |
| Stable pitch | Calm speech |

---

### 14. Jitter (Pitch Instability)

Jitter measures cycle-to-cycle variation in pitch period:

```
Jitter = (1 / N-1) · Σ |Ti - Ti+1|
```

where `Ti` = pitch period of cycle `i`. Higher jitter → stress, nervousness, emotional instability.

---

### 15. Shimmer (Amplitude Variation)

Shimmer measures cycle-to-cycle amplitude changes:

```
Shimmer = (1 / N-1) · Σ |Ai - Ai+1|
```

where `Ai` = peak amplitude of cycle `i`. Higher shimmer → vocal strain, emotional stress.

---

### 16. Voice Energy

```
Energy = Σ x[n]²
```

| Energy Level | Interpretation |
|---|---|
| High | Loud / excited |
| Low | Tired / withdrawn |

---

### 17. Voice Feature Vector

```
Fvoice = [MFCC1, MFCC2, ..., MFCC13, Pitch, Jitter, Shimmer, Energy]
```

Dimension: **17 features**

---

### 18. Voice Valence and Arousal

Voice valence:

```
Vv = w1·Pitch + w2·Energy + w3·MFCC
```

Voice arousal:

```
Av = w4·PitchVariance + w5·Jitter + w6·Shimmer
```

---

### 19. Voice Distress Score

Using the same formulation as facial distress:

```
Dv = (1 - Vv) · |Av|
```

---

### 20. Multimodal Feature Fusion

All face, voice, and behavioral features are concatenated into a single fusion vector:

```
F = [V̄f, Āf, D̄f, EAR, MAR, Vv, Av, Dv, Pitch, Jitter, Shimmer, Energy]
```

Dimension: **12 features**

Because features have different natural scales, normalization is applied before classification so that all features contribute equally.

---

### 21. Mental State Classification

A logistic regression model predicts distressed vs. normal state:

```
z = wᵀ · F + b
P(distressed) = σ(z) = 1 / (1 + e⁻ᶻ)
```

Decision rule:
```
P > 0.65         → Distressed
0.45 < P ≤ 0.65  → Elevated
P ≤ 0.45         → Normal
```

---

### 22. Risk Assessment

```
P < 0.3          → Low Risk
0.3 ≤ P < 0.7    → Moderate Risk
P ≥ 0.7          → High Risk
```

**Confidence score:**
```
conf = |P - 0.5| / 0.5 × 100%
```

`conf = 100%` = maximum certainty (P=0 or P=1). `conf = 0%` = maximum uncertainty (P=0.5).

---

### 23. Trend Detection

Emotional trajectory is estimated from recent distress probability history:

```
slope = P_last - P_first

slope > 0.1   → Increasing distress
slope < -0.1  → Decreasing distress
otherwise     → Stable
```

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Camera Input (Webcam)                          │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
                   ┌───────────────────────┐
                   │   Face Emotion Model  │
                   │  (MediaPipe + DeepFace│
                   │  468 landmarks, 7-class│
                   │  emotion probabilities)│
                   └───────────┬───────────┘
                               ↓
                   ┌───────────────────────┐
                   │  Valence / Arousal /  │
                   │  Distress (Face)      │
                   │  Vf, Af, Df           │
                   │  Temporal tracker     │
                   │  EAR, MAR             │
                   └───────────┬───────────┘
                               │
┌──────────────────────────────────────────────────────────────────┐
│                   Microphone Input                                │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
                   ┌───────────────────────┐
                   │  Audio Preprocessing  │
                   │  Framing + Windowing  │
                   │  Fourier Transform    │
                   │  Mel Filterbank       │
                   └───────────┬───────────┘
                               ↓
                   ┌───────────────────────┐
                   │  Voice Feature        │
                   │  Extraction           │
                   │  MFCC (13)            │
                   │  Pitch, Jitter        │
                   │  Shimmer, Energy      │
                   └───────────┬───────────┘
                               ↓
                   ┌───────────────────────┐
                   │  Voice Emotion Model  │
                   │  Vv, Av, Dv           │
                   └───────────┬───────────┘
                               │
              ┌────────────────┴────────────────┐
              │         Feature Fusion           │
              │  F = [Vf, Af, Df, EAR, MAR,     │
              │        Vv, Av, Dv,               │
              │        Pitch, Jitter,            │
              │        Shimmer, Energy]          │
              └────────────────┬────────────────┘
                               ↓
              ┌────────────────────────────────┐
              │  Logistic Mental State         │
              │  Classifier                    │
              │  P(distressed) = σ(wᵀF + b)    │
              └────────────────┬───────────────┘
                               ↓
              ┌────────────────────────────────┐
              │  Risk Analyzer                 │
              │  Low / Moderate / High         │
              │  + Confidence Score            │
              └────────────────┬───────────────┘
                               ↓
              ┌────────────────────────────────┐
              │  Trend Analyzer                │
              │  Increasing / Stable /         │
              │  Decreasing                    │
              └────────────────────────────────┘
```

---

## Project Structure

```
project/
│
├── face_module/
│   ├── camera/
│   │   └── camera_stream.py         # Webcam capture using OpenCV
│   │
│   ├── detection/
│   │   └── face_detector.py         # MediaPipe BlazeFace detection
│   │
│   ├── alignment/
│   │   └── face_landmarks.py        # 468-point facial landmark extraction
│   │
│   ├── emotion_model/
│   │   └── emotion_classifier.py    # DeepFace CNN emotion recognition
│   │
│   └── features/
│       ├── valence_arousal.py       # Facial dimensional emotion mapping
│       ├── distress_score.py        # Face distress computation
│       ├── temporal_tracker.py      # Sliding window emotional smoothing
│       ├── facial_behavior.py       # Eye/mouth openness ratios
│       └── feature_fusion.py        # Combine all features into vector
│
├── voice_module/
│   ├── audio/
│   │   └── audio_stream.py          # Microphone capture
│   │
│   └── features/
│       ├── mfcc_extractor.py        # 13-coefficient MFCC extraction
│       ├── pitch_extractor.py       # Fundamental frequency (F0)
│       └── jitter_shimmer.py        # Pitch/amplitude variation
│
├── emotion_model/
│   └── voice_emotion_model.py       # Voice valence/arousal/distress model
│
├── models/
│   ├── mental_state_classifier.py   # Logistic regression (12-D fusion)
│   └── blaze_face_short_range.tflite
│
├── output/
│   ├── risk_analyzer.py             # Risk level computation
│   └── trend_analyzer.py           # Emotional trajectory detection
│
├── main.py                          # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

---

## Demo

> Add screenshots to `docs/` to populate this table.

| Input | Detected State |
|---|---|
| ![](docs/demo_happy.png) | Low Risk — Happy expression, calm voice |
| ![](docs/demo_sad.png) | Moderate Risk — Sad expression, low energy voice |
| ![](docs/demo_distressed.png) | High Risk — Fear expression, high jitter/shimmer |

---

## Dependencies

```txt
# Vision
opencv-python>=4.5.0
mediapipe>=0.10.0
deepface>=0.0.75
tensorflow>=2.8.0

# Audio
librosa>=0.9.0
sounddevice>=0.4.0
pyaudio>=0.2.11

# Core
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
```

**Installation:**
```bash
pip install -r requirements.txt
```

---

## Usage

### Real-Time Detection

```bash
python main.py
```

**Sample console output:**
```
Frame 120
─────────────────────────────────────────────────────────
FACE MODALITY
  Emotion Probabilities:
    happy:    0.12
    sad:      0.68
    angry:    0.05
    fear:     0.08
    surprise: 0.02
    disgust:  0.03
    neutral:  0.02

  Dimensional Affect (Face):
    Valence:  -0.54
    Arousal:   0.32
    Distress:  0.49

  Temporal Average (30 frames):
    Valence:  -0.48
    Arousal:   0.29
    Distress:  0.43

  Behavioral Features:
    Eye Openness:   0.28
    Mouth Openness: 0.15

VOICE MODALITY
  MFCC:     [-4.2, 1.1, -0.8, ...]
  Pitch:     138 Hz
  Jitter:    0.021
  Shimmer:   0.034
  Energy:    0.61

  Dimensional Affect (Voice):
    Valence:  -0.38
    Arousal:   0.44
    Distress:  0.62

FUSION & CLASSIFICATION
  Mental State:    Distressed
  Probability:     0.73
  Risk Level:      High
  Confidence:      92%
  Trend:           Increasing distress
─────────────────────────────────────────────────────────
```

---

## What Makes This System Novel

### 1. Multimodal Architecture
Combines **facial expression** and **voice prosody** in a unified pipeline — most systems use only one modality.

### 2. Dimensional Emotion Modeling
Both face and voice are mapped to continuous valence-arousal space rather than discrete emotion labels, capturing subtle variations.

### 3. Dual Distress Metric
An explicit distress score `D = (1-V)·|A|` is computed independently from each modality and combined in fusion.

### 4. Voice Prosody Analysis
MFCC, pitch, jitter, shimmer, and energy are well-established depression and stress biomarkers used in clinical research (DAIC-WOZ, MODMA, Interspeech challenges).

### 5. Behavioral Micro-Features
Eye and mouth openness ratios capture non-expressive signals such as fatigue or withdrawal even when facial expression appears neutral.

### 6. Temporal Tracking
A sliding window over face features distinguishes momentary expressions from sustained emotional states.

### 7. Feature Fusion
All modalities are concatenated into a 12-dimensional vector that feeds a single classifier — capturing both *what* emotion is expressed and *how* it is expressed.

### 8. Risk Assessment and Trend Monitoring
Actionable risk levels (low/moderate/high) with confidence scores and worsening/improving trajectory detection.

---

## Comparison to Research Literature

| Component | Basic FER Systems | AffectNet Paper | This System |
|---|---|---|---|
| Emotion categories | ✓ | ✓ | ✓ |
| Valence-arousal (face) | ✗ | ✓ | ✓ |
| Voice prosody features | ✗ | ✗ | ✓ |
| Valence-arousal (voice) | ✗ | ✗ | ✓ |
| Distress modeling | ✗ | ✗ | ✓ |
| Multimodal fusion | ✗ | ✗ | ✓ |
| Temporal tracking | ✗ | ✗ | ✓ |
| Behavioral features | ✗ | ✗ | ✓ |
| Mental state classification | ✗ | ✗ | ✓ |
| Risk assessment | ✗ | ✗ | ✓ |
| Trend detection | ✗ | ✗ | ✓ |

---

## Technical Implementation Details

### Face Detection
- **Model:** MediaPipe BlazeFace — lightweight, real-time on CPU
- **Speed:** ~200 FPS
- **Output:** Face bounding box `(x, y, width, height)`

### Facial Landmarks
- **Model:** MediaPipe FaceMesh
- **Points:** 468 3D facial landmarks
- **Usage:** EAR and MAR computation

### Emotion Recognition
- **Framework:** DeepFace (VGG-Face CNN, pretrained on VGGFace2)
- **Output:** 7-class probability distribution

### Audio Processing
- **Library:** LibROSA + SoundDevice
- **Sample rate:** 16,000 Hz
- **Features:** 13 MFCCs, pitch (F0), jitter, shimmer, energy

### Mental State Classifier
- **Model:** Logistic Regression (scikit-learn)
- **Input:** 12-dimensional fusion vector
- **Training:** Placeholder weights — requires labeled training data for production

---

## Current Limitations

### 1. Placeholder Classification Weights
The mental state classifier uses heuristic weights, not trained from labeled data. **Resolution:** Collect an annotated dataset with ground-truth mental state labels.

### 2. No Clinical Validation
Has not been validated against clinical assessments (PHQ-9, GAD-7). **Future work:** Clinical validation study.

### 3. Lighting and Noise Sensitivity
Face performance degrades in poor lighting; voice performance degrades in noisy environments. **Resolution:** Data augmentation and noise-robust models.

### 4. No Personalization
Uses population-average weights for both face and voice. **Future work:** User-specific baseline calibration.

### 5. Voice-Face Synchronization
The current prototype processes both streams independently. **Future work:** Synchronize streams at the frame level for tighter temporal alignment.

### 6. Voice Activity Detection
The current prototype extracts voice features even during silence. Future versions should incorporate **Voice Activity Detection (VAD)** to avoid extracting speech features from non-speech audio segments, which otherwise introduces noise into voice emotion estimates.

---

## Important Note on Emotion vs Mental State

This system detects **observable emotional signals**, not clinical mental health diagnoses.

Facial expressions and voice prosody provide **behavioral indicators**, but:

- Actors can simulate emotions
- Images or video feeds may not represent the subject's real emotional state
- Environmental factors (lighting, background noise) can affect both facial and vocal signals
- Individual baseline emotional expression varies significantly across people

Therefore the system estimates **relative distress indicators**, not definitive psychological conditions. All outputs should be interpreted as signals for further investigation, never as standalone conclusions.

---

## Future Enhancements

**Short-term:** Train classifier on labeled dataset, add noise-robust audio preprocessing, implement frame-level synchronization between modalities.

**Medium-term:** Add text sentiment analysis (BERT), deep learning fusion model (replace logistic regression), mobile deployment, personalized baseline calibration.

**Long-term:** Clinical validation study, longitudinal tracking across sessions, SHAP-based explainability, integration with mental health platforms.

---

## References

1. **Mollahosseini, A., Hasani, B., & Mahoor, M.H. (2019).** *AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild.* IEEE Transactions on Affective Computing, Vol. 10, No. 1, pp. 18–31. [[paper]](https://arxiv.org/abs/1708.03985)

2. **Russell, J.A. (1980).** *A Circumplex Model of Affect.* Journal of Personality and Social Psychology, Vol. 39, No. 6, pp. 1161–1178.

3. **Ekman, P., & Friesen, W.V. (1971).** *Constants Across Cultures in the Face and Emotion.* Journal of Personality and Social Psychology, Vol. 17, No. 2, p. 124.

4. **Ringeval, F., et al. (2019).** *AVEC 2019 Workshop and Challenge: State-Mind, Detecting Depression with AI.* — Depression speech biomarkers (jitter, shimmer, MFCC)

5. **Serengil, S.I., & Ozpinar, A. (2020).** *DeepFace: A Lightweight Face Recognition and Facial Attribute Analysis Framework.*

6. **Lugaresi, C., et al. (2019).** *MediaPipe: A Framework for Building Perception Pipelines.* arXiv:1906.08172.

7. **McFee, B., et al. (2015).** *librosa: Audio and Music Signal Analysis in Python.* Proceedings of the 14th Python in Science Conference.

---
## License

MIT License. See `LICENSE` for details.

---

## Disclaimer

**This is a research prototype, not a clinical diagnostic tool.**

The system produces relative indicators based on facial expressions and voice prosody, but:
- It is **not** validated against clinical mental health assessments
- It is **not** trained on clinical populations
- It should **not** be used for medical decision-making

Always consult qualified mental health professionals for diagnosis and treatment.

---

<div align="center">

**Built with multimodal dimensional emotion modeling and real-time affective computing**

Made with 🧠 and ☕

</div>
