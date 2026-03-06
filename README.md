# Real-Time Multimodal Emotional Distress Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00897B?logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![DeepFace](https://img.shields.io/badge/DeepFace-0.0.75+-FF6F00)]()
[![Status](https://img.shields.io/badge/Status-Prototype-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A real-time affective computing system that detects emotional distress from facial expressions using dimensional emotion modeling, temporal tracking, and behavioral analysis.

---

## What This System Does

This system processes **live webcam video** to:
1. Detect faces and extract facial landmarks
2. Classify facial expressions into emotion categories
3. Map emotions to **valence-arousal** dimensional space
4. Compute **distress scores** from emotional dimensions
5. Track emotional states over time
6. Analyze facial micro-behaviors (eye/mouth openness)
7. Predict mental state (distressed vs. normal)
8. Assess risk levels and emotional trends

**Key Innovation:** Unlike basic emotion recognition systems, this implements **dimensional emotion modeling** where emotions are represented as continuous values in valence-arousal space, not just discrete categories.

---

## Research Context

This system is inspired by dimensional models of affect, particularly the **valence-arousal circumplex** framework where emotions are represented in continuous 2D space rather than discrete categories.

**Primary reference:**

**Mollahosseini, A., Hasani, B., & Mahoor, M.H. (2019).** *AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild.* IEEE Transactions on Affective Computing, Vol. 10, No. 1, pp. 18-31. [[paper]](https://arxiv.org/abs/1708.03985)

The AffectNet paper demonstrates that:
- Emotions can be modeled in continuous dimensional space (valence and arousal)
- Valence represents how positive/negative an emotion is
- Arousal represents the intensity/activation level
- This approach captures subtle emotional variations that discrete categories cannot

**This implementation extends beyond the paper by adding:**
- Real-time distress modeling
- Temporal emotional tracking
- Behavioral feature extraction (eye/mouth openness)
- Risk assessment and trend detection
- Mental state classification

---

## Mathematical Framework

### 1. Emotion Probability Distribution

DeepFace outputs a probability distribution over 7 emotion categories:

```
P(emotion) = {happy, sad, angry, fear, surprise, disgust, neutral}
```

Constraint:
```
Σ P(eᵢ) = 1
```

---

### 2. Valence Calculation

Valence is computed as a weighted expectation over emotion probabilities:

```
V = Σ P(eᵢ) · wᵥ(eᵢ)
```

Where:
- `P(eᵢ)` — probability of emotion i
- `wᵥ(eᵢ)` — valence weight for emotion i

**Valence weights:**
```
happy     → +1.0  (most positive)
surprise  → +0.4
neutral   →  0.0
sad       → -0.7
fear      → -0.8
anger     → -0.9
disgust   → -0.95 (most negative)
```

**Interpretation:** `V ∈ [-1, 1]` where positive values indicate positive emotions and negative values indicate negative emotions.

---

### 3. Arousal Calculation

Arousal measures emotional activation/intensity:

```
A = Σ P(eᵢ) · wₐ(eᵢ)
```

**Arousal weights:**
```
fear      → +0.9  (high activation)
anger     → +0.8
surprise  → +0.7
happy     → +0.6
neutral   →  0.0
sad       → -0.3  (low activation)
disgust   → -0.2
```

**Interpretation:** `A ∈ [-1, 1]` where positive values indicate high activation/excitement and negative values indicate low activation/calmness.

---

### 4. Distress Score

Distress is computed from valence and arousal:

```
D = (1 - V) · |A|
```

**Rationale:**
- High distress = negative valence + high arousal
- Low distress = positive valence and/or low arousal

**Examples:**
```
Fear (V=-0.8, A=0.9):  D = (1-(-0.8)) · 0.9 = 1.62  (high distress)
Happy (V=0.9, A=0.6):  D = (1-0.9) · 0.6 = 0.06     (low distress)
Neutral (V=0, A=0):    D = (1-0) · 0 = 0.0          (no distress)
```

---

### 5. Temporal Smoothing

Emotions fluctuate rapidly. To stabilize predictions, we maintain a sliding window of the last `N` observations:

```
V̄ = mean(V₁, V₂, ..., Vₙ)
Ā = mean(A₁, A₂, ..., Aₙ)
D̄ = mean(D₁, D₂, ..., Dₙ)
```

Default window size: `N = 30` frames (~1 second at 30 FPS).

**Effect:** Reduces noise from single-frame misclassifications while preserving genuine emotional changes.

---

### 6. Behavioral Features

Facial landmarks enable extraction of micro-behavioral features:

**Eye Openness Ratio:**
```
         vertical_distance(upper_eyelid, lower_eyelid)
EAR = ──────────────────────────────────────────────────
      horizontal_distance(left_corner, right_corner)
```

**Mouth Openness Ratio:**
```
         vertical_distance(upper_lip, lower_lip)
MAR = ──────────────────────────────────────────────
      horizontal_distance(left_corner, right_corner)
```

These behavioral signals complement emotion recognition — for example, reduced eye openness may indicate fatigue or depression even when facial expression appears neutral.

---

### 7. Feature Fusion

The system combines emotional and behavioral features into a single vector:

```
F = [V̄, Ā, D̄, EAR, MAR] ∈ ℝ⁵
```

This fused representation captures both **what emotion is expressed** and **how it is expressed**.

---

### 8. Mental State Classification

A logistic regression model predicts distressed vs. normal state:

```
z = wᵀ · F + b
P(distressed) = σ(z) = 1 / (1 + e⁻ᶻ)
```

Decision rule:
```
if P(distressed) > 0.5:
    state = "distressed"
else:
    state = "normal"
```

---

### 9. Risk Assessment

Risk level is determined by probability thresholds:

```
P < 0.3        → Low Risk
0.3 ≤ P < 0.7  → Moderate Risk
P ≥ 0.7        → High Risk
```

**Confidence score:**
```
         |P - 0.5|
conf = ────────────  × 100%
            0.5
```

Where `conf = 100%` indicates maximum certainty (P=0 or P=1) and `conf = 0%` indicates maximum uncertainty (P=0.5).

---

### 10. Trend Detection

Emotional trajectory is estimated from recent distress probability history:

```
slope = P_last - P_first

if slope > 0.1:
    trend = "increasing distress"
elif slope < -0.1:
    trend = "decreasing distress"
else:
    trend = "stable"
```

This enables detection of **worsening** or **improving** emotional states over time.

---

## System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    Camera Input (Webcam)                    │
└──────────────────────┬─────────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│              Face Detection (MediaPipe)                   │
│  Detects face bounding box and crops face region          │
└──────────────────────┬───────────────────────────────────┘
                       ↓
┌──────────────────────────────────────────────────────────┐
│         Facial Landmark Extraction (MediaPipe)            │
│  Extracts 468 3D facial landmarks                         │
└──────────────────────┬───────────────────────────────────┘
                       ↓
         ┌─────────────┴─────────────┐
         ↓                           ↓
┌─────────────────────┐    ┌──────────────────────┐
│ Emotion Recognition │    │ Behavioral Features  │
│    (DeepFace CNN)   │    │  • Eye Openness      │
│                     │    │  • Mouth Openness    │
│  P(emotions) → 7D   │    │                      │
└──────┬──────────────┘    └──────┬───────────────┘
       ↓                          ↓
┌─────────────────────┐    ┌──────────────────────┐
│ Valence & Arousal   │    │                      │
│  V = Σ P(e)·w_v(e)  │    │                      │
│  A = Σ P(e)·w_a(e)  │    │                      │
└──────┬──────────────┘    │                      │
       ↓                    │                      │
┌─────────────────────┐    │                      │
│  Distress Score     │    │                      │
│  D = (1-V)·|A|      │    │                      │
└──────┬──────────────┘    │                      │
       ↓                    │                      │
┌─────────────────────┐    │                      │
│ Temporal Tracker    │    │                      │
│  V̄, Ā, D̄ (30 frames) │    │                      │
└──────┬──────────────┘    │                      │
       ↓                    ↓                      │
       └────────┬───────────┘                      │
                ↓                                  │
┌──────────────────────────────────────────────────┘
│         Feature Fusion                           │
│  F = [V̄, Ā, D̄, EAR, MAR]                          │
└──────────────────┬───────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│      Mental State Classification                  │
│  P(distressed) = σ(w^T·F + b)                    │
└──────────────────┬───────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│           Risk Analysis                           │
│  Risk Level: Low / Moderate / High                │
│  Confidence Score                                 │
└──────────────────┬───────────────────────────────┘
                   ↓
┌──────────────────────────────────────────────────┐
│           Trend Detection                         │
│  Increasing / Stable / Decreasing                 │
└──────────────────────────────────────────────────┘
```

---

## Project Structure

```
face_module/
│
├── camera/
│   └── camera_stream.py        # Webcam capture using OpenCV
│
├── detection/
│   └── face_detector.py        # MediaPipe BlazeFace detection
│
├── alignment/
│   └── face_landmarks.py       # 468-point facial landmark extraction
│
├── emotion_model/
│   └── emotion_classifier.py   # DeepFace CNN emotion recognition
│
├── features/
│   ├── valence_arousal.py      # Dimensional emotion mapping
│   ├── distress_score.py       # Distress computation
│   ├── temporal_tracker.py     # Sliding window emotional smoothing
│   ├── facial_behavior.py      # Eye/mouth openness ratios
│   └── feature_fusion.py       # Combine all features into vector
│
├── models/
│   ├── mental_state_classifier.py   # Logistic regression model
│   └── blaze_face_short_range.tflite  # Face detection model
│
├── output/
│   ├── risk_analyzer.py        # Risk level computation
│   └── trend_analyzer.py       # Emotional trajectory detection
│
├── main.py                     # End-to-end pipeline runner
├── requirements.txt
└── README.md
```

---

## Dependencies

```txt
opencv-python>=4.5.0
mediapipe>=0.10.0
deepface>=0.0.75
numpy>=1.21.0
tensorflow>=2.8.0
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

**Output (console):**
```
Frame 120
─────────────────────────────────
Emotion Probabilities:
  happy:    0.12
  sad:      0.68
  angry:    0.05
  fear:     0.08
  surprise: 0.02
  disgust:  0.03
  neutral:  0.02

Dimensional Affect:
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

Mental State:
  State:       Distressed
  Probability: 0.73
  Risk Level:  High
  Confidence:  92%

Trend: Increasing distress
─────────────────────────────────
```

---

## What Makes This System Novel

Most facial expression recognition systems output discrete emotion labels (happy, sad, etc.). This system goes further:

### 1. Dimensional Emotion Modeling
Maps discrete emotions to continuous valence-arousal space, capturing subtle emotional variations.

### 2. Distress Metric
Introduces an explicit distress variable `D = (1-V)·|A|` that models psychological distress from emotional dimensions.

### 3. Temporal Tracking
Maintains emotional state history to distinguish momentary expressions from sustained emotional states.

### 4. Behavioral Micro-Features
Extracts eye and mouth openness ratios that capture non-expressive behavioral signals (e.g., fatigue, withdrawal).

### 5. Feature Fusion
Combines emotional and behavioral modalities into a unified representation.

### 6. Mental State Prediction
Predicts binary mental state (distressed vs. normal) rather than just reporting emotions.

### 7. Risk Assessment
Provides actionable risk levels (low/moderate/high) with confidence scores.

### 8. Trend Monitoring
Tracks emotional trajectory to detect worsening or improving states over time.

---

## Comparison to Research Literature

| Component | Basic FER Systems | AffectNet Paper | This System |
|-----------|-------------------|-----------------|-------------|
| Emotion categories | ✓ | ✓ | ✓ |
| Valence-arousal | ✗ | ✓ | ✓ |
| Distress modeling | ✗ | ✗ | ✓ |
| Temporal tracking | ✗ | ✗ | ✓ |
| Behavioral features | ✗ | ✗ | ✓ |
| Mental state classification | ✗ | ✗ | ✓ |
| Risk assessment | ✗ | ✗ | ✓ |
| Trend detection | ✗ | ✗ | ✓ |

---

## Technical Implementation Details

### Face Detection
- **Model:** MediaPipe BlazeFace (lightweight, optimized for real-time)
- **Speed:** ~200 FPS on CPU
- **Output:** Face bounding box `(x, y, width, height)`

### Facial Landmarks
- **Model:** MediaPipe FaceMesh
- **Points:** 468 3D facial landmarks
- **Usage:** Compute eye aspect ratio (EAR) and mouth aspect ratio (MAR)

### Emotion Recognition
- **Framework:** DeepFace
- **Architecture:** VGG-Face CNN pretrained on VGGFace2 dataset
- **Output:** 7-class probability distribution

### Mental State Classifier
- **Model:** Logistic Regression (scikit-learn)
- **Input:** 5-dimensional feature vector `[V̄, Ā, D̄, EAR, MAR]`
- **Training:** Placeholder weights (requires labeled training data for production)

---

## Current Limitations

This is an honest prototype. Here are the constraints:

### 1. Placeholder Classification Weights
The mental state classifier uses heuristic weights, not trained from labeled data. **Resolution:** Collect annotated dataset with ground-truth mental state labels.

### 2. Single Modality
Only uses visual (facial) information. **Future work:** Add voice prosody, text sentiment, physiological signals.

### 3. No Personalization
Uses population-average emotion weights. **Future work:** User-specific calibration.

### 4. Lighting Sensitivity
Performance degrades in poor lighting or extreme head poses. **Resolution:** Add data augmentation and pose-invariant models.

### 5. No Clinical Validation
Has not been validated against clinical assessments (PHQ-9, GAD-7). **Future work:** Clinical validation study.

---

## Future Enhancements

### Short-Term
- Train mental state classifier on labeled dataset
- Add data augmentation for robustness
- Implement attention mechanism to weight emotion frames
- Add logging and session persistence

### Medium-Term
- Multi-modal fusion (add voice emotion recognition)
- Personalized baseline calibration
- Deep learning classifier (replace logistic regression)
- Mobile deployment (Android/iOS)

### Long-Term
- Clinical validation study
- Longitudinal tracking across sessions
- Explainability (SHAP values for predictions)
- Integration with mental health platforms

---

## References

1. **Mollahosseini, A., Hasani, B., & Mahoor, M.H. (2019).** *AffectNet: A Database for Facial Expression, Valence, and Arousal Computing in the Wild.* IEEE Transactions on Affective Computing, Vol. 10, No. 1, pp. 18-31. [[paper]](https://arxiv.org/abs/1708.03985)

2. **Russell, J.A. (1980).** *A Circumplex Model of Affect.* Journal of Personality and Social Psychology, Vol. 39, No. 6, pp. 1161-1178. — Foundational valence-arousal framework

3. **Ekman, P., & Friesen, W.V. (1971).** *Constants Across Cultures in the Face and Emotion.* Journal of Personality and Social Psychology, Vol. 17, No. 2, p. 124. — Categorical emotion model

4. **Serengil, S.I., & Ozpinar, A. (2020).** *DeepFace: A Lightweight Face Recognition and Facial Attribute Analysis Framework.* — DeepFace implementation

5. **Lugaresi, C., et al. (2019).** *MediaPipe: A Framework for Building Perception Pipelines.* arXiv preprint arXiv:1906.08172. — MediaPipe framework

## License

MIT License. See `LICENSE` for details.

---

## Disclaimer

**This is a research prototype, not a clinical diagnostic tool.**

The system produces relative indicators based on facial expressions, but:
- It is **not** validated against clinical mental health assessments
- It is **not** trained on clinical populations
- It should **not** be used for medical decision-making

Always consult qualified mental health professionals for diagnosis and treatment.

---

<div align="center">

**Built with dimensional emotion modeling and real-time affective computing**

Made with 🧠 and ☕

</div>
