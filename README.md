# 🎭 Real-Time Facial Emotion Recognition System (FER-2013)

[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![TFLite](https://img.shields.io/badge/Inference-TFLite%20XNNPACK-00A86B?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/lite)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV%204.x-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Performance](https://img.shields.io/badge/Realtime%20Throughput-30%2B%20FPS%20(CPU)-brightgreen)](#-performance--benchmarks)
[![Dataset](https://img.shields.io/badge/Dataset-FER--2013-blue)](#-model-architecture--dataset)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An enterprise-grade, high-performance Computer Vision and Deep Learning desktop application for real-time facial expression analysis and emotion recognition. Capable of processing high-resolution live webcam feeds at **30+ FPS on standard CPU hardware** with dual inference backends, anti-jitter temporal smoothing, multi-face tracking, and a futuristic sci-fi visual HUD.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Emotion Taxonomy & Color Palette](#-emotion-taxonomy--color-palette)
- [Installation & Setup](#-installation--setup)
- [Quickstart Guide](#-quickstart-guide)
  - [Live Webcam Stream](#1-live-webcam-application)
  - [Static Image Inference](#2-static-image-testing)
  - [Synthetic Pipeline Verification](#3-synthetic-pipeline-verification)
- [Interactive Controls & Hotkeys](#-interactive-controls--hotkeys)
- [Model Architecture & Dataset](#-model-architecture--dataset)
- [Mathematical & Algorithmic Foundations](#-mathematical--algorithmic-foundations)
- [Viva & AI Interview Preparation Guide](#-viva--ai-interview-preparation-guide)
- [Troubleshooting & FAQs](#-troubleshooting--faqs)
- [Roadmap](#-future-roadmap)
- [License & Acknowledgments](#-license--acknowledgments)

---

## 🚀 Key Features

* **⚡ Dual-Engine Inference Architecture**:
  * **TFLite (`emotion_model.tflite`)**: Ultra-lightweight graph (~7.2 MB) compiled with XNNPACK delegates for high-throughput, low-latency CPU inference (**30–60 FPS**).
  * **Keras (`emotion_model.keras`)**: Full-precision deep neural network (~21.8 MB) running native TensorFlow execution.
  * **On-the-Fly Engine Switching**: Toggle between engines during live video without restarting the stream (press `M`).
* **🛡️ Anti-Jitter Temporal Smoothing**:
  * Integrates an **Exponential Moving Average (EMA)** filter ($\alpha = 0.65$) combined with a rolling probability queue to eliminate frame-to-frame prediction flickering caused by micro-blinks, lighting variance, or sensor noise.
* **🎨 High-Tech Cyberpunk HUD Visualizer**:
  * Precision sci-fi corner-bracket bounding boxes.
  * Dynamic, emotion-specific accent colors and visual icons for instant state identification.
  * Real-time 7-class horizontal probability distribution breakdown with percentage confidence meters.
  * Live status bar displaying real-time FPS, active backend engine, and hotkey legend.
* **👥 Multi-Face Concurrency**:
  * Simultaneously detects, tracks, crops, and classifies multiple faces within the camera frame with zero crashes.
* **📸 Snapshot Capture Engine**:
  * Save high-resolution annotated frames directly to disk with a single keystroke (`S`), automatically tagged with ISO timestamps.
* **🧪 Offline & Headless Testing Suite**:
  * Run inference on static images or generate synthetic geometric facial proxies for automated CI/CD pipeline verification without requiring a physical camera.

---

## 🧠 System Architecture

```
                                  [ Camera Feed / Input Image ]
                                                │
                                                ▼
                                   [ OpenCV Video Capture ]
                                                │
                                                ▼
                             [ Haar Cascade Face Detector ]
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
             [ Face 1 Crop ROI ]                               [ Face N Crop ROI ]
                       │                                                 │
                       ▼                                                 ▼
         [ Preprocessing Pipeline ]                         [ Preprocessing Pipeline ]
         • Grayscale Conversion                             • Grayscale Conversion
         • Resize to (48 x 48)                              • Resize to (48 x 48)
         • Reshape (1, 48, 48, 1)                           • Reshape (1, 48, 48, 1)
                       │                                                 │
                       ▼                                                 ▼
        [ Inference: TFLite / Keras ]                      [ Inference: TFLite / Keras ]
                       │                                                 │
                       ▼                                                 ▼
         [ Raw Softmax Vector (1x7) ]                       [ Raw Softmax Vector (1x7) ]
                       │                                                 │
                       ▼                                                 │
       [ Temporal Smoothing Filter (EMA) ]                               │
         P_t = α * P_curr + (1 - α) * P_prev                             │
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                                    [ HUD Rendering Engine ]
                                    • Tech-Bracket Corners
                                    • Emotion-Colored Bounding Box
                                    • 7-Class Horizontal Bar Graph
                                    • FPS & Active Backend Counter
                                                │
                                                ▼
                                    [ High-FPS Output Display ]
```

---

## 📂 Repository Structure

```text
face understanding/
│
├── realtime_emotion.py       # Main real-time desktop application (Webcam, HUD, Hotkeys)
├── test_image.py             # Static image testing utility & synthetic test generator
│
├── emotion_model.tflite      # Optimized TensorFlow Lite model (~7.2 MB) for fast CPU inference
├── emotion_model.keras       # Full-precision TensorFlow Keras model (~21.8 MB)
├── model_metadata.json       # Model configuration, input dimensions, and class index mapping
│
├── snapshots/                # Directory where saved screenshots and frame captures are stored
├── test_synthetic_face.jpg   # Generated test input for automated pipeline verification
├── test_synthetic_output.jpg # Output artifact from synthetic verification test
└── README.md                 # Complete project documentation and viva defense manual
```

---

## 📊 Emotion Taxonomy & Color Palette

The model classifies facial crops into **7 distinct emotion categories** according to the FER-2013 standard. Each emotion is paired with a distinct OpenCV color accent (BGR format):

| Class Index | Emotion | BGR Code | Hex Code | Visual Symbol | Characteristic Facial Cues |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Angry** | `(36, 36, 220)` | `#DC2424` | 😠 `[ANGRY]` | Furrowed eyebrows, glaring eyes, tightened lips |
| **1** | **Disgust** | `(34, 139, 34)` | `#228B22` | 🤢 `[DISGUST]` | Wrinkled nose, raised upper lip, narrowed eyes |
| **2** | **Fear** | `(160, 32, 240)` | `#F020A0` | 😨 `[FEAR]` | Raised eyebrows, widened eyes, slightly parted lips |
| **3** | **Happy** | `(30, 215, 96)` | `#60D71E` | 😄 `[HAPPY]` | Raised cheeks, corners of mouth drawn up (smile), crows-feet |
| **4** | **Neutral** | `(220, 220, 0)` | `#00DCDC` | 😐 `[NEUTRAL]` | Relaxed facial muscles, horizontal lip line, steady gaze |
| **5** | **Sad** | `(219, 112, 147)` | `#9370DB` | 😢 `[SAD]` | Drooping mouth corners, pulled-up inner eyebrows |
| **6** | **Surprise** | `(0, 215, 255)` | `#FFD700` | 😲 `[SURPRISE]` | High arched eyebrows, wide open eyes, dropped jaw |

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.9, 3.10, or 3.11
- **Hardware**: Standard webcam or USB camera (runs smoothly on CPU; GPU is optional)

### 2. Create a Virtual Environment (Recommended)

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Windows (Command Prompt)
python -m venv venv
.\venv\Scripts\activate.bat

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install opencv-python tensorflow numpy
```

> **Note**: TensorFlow packages include the complete TensorFlow Lite runtime and optimized XNNPACK delegates by default.

---

## 🎯 Quickstart Guide

### 1. Live Webcam Application

Launch the real-time webcam emotion detector with default settings (TFLite engine, camera index `0`):

```bash
python realtime_emotion.py
```

#### Advanced Command-Line Flags:

```bash
# Use an external or secondary camera (index 1)
python realtime_emotion.py --camera 1

# Start directly using the full Keras neural network engine
python realtime_emotion.py --backend keras

# Launch with custom window dimensions (e.g., Full HD)
python realtime_emotion.py --width 1920 --height 1080
```

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--camera` | `int` | `0` | Device index for `cv2.VideoCapture` |
| `--backend` | `str` | `tflite` | Inference engine (`tflite` or `keras`) |
| `--width` | `int` | `1280` | Target display window width in pixels |
| `--height` | `int` | `720` | Target display window height in pixels |

---

### 2. Static Image Testing

To run inference on static image files without a webcam:

```bash
# Process a picture and save the annotated result
python test_image.py --input path/to/portrait.jpg --output output_annotated.jpg

# Inspect the result immediately in an interactive preview window
python test_image.py --input path/to/portrait.jpg --show

# Force Keras backend for static testing
python test_image.py --input portrait.jpg --backend keras
```

---

### 3. Synthetic Pipeline Verification

Verify that model loading, tensor preprocessing, inference, and visualization work end-to-end without any physical images or camera hardware:

```bash
python test_image.py --generate-synthetic
```

This generates `test_synthetic_face.jpg` and saves the verified annotated HUD render to `test_synthetic_output.jpg`.

---

## ⌨️ Interactive Controls & Hotkeys

While the real-time webcam window is focused, use these interactive hotkeys:

| Key | Control | Description |
| :---: | :--- | :--- |
| <kbd>S</kbd> | **Capture Snapshot** | Saves the current frame with all active HUD overlays to `./snapshots/emotion_snapshot_YYYYMMDD_HHMMSS.jpg` |
| <kbd>B</kbd> | **Toggle Probabilities** | Shows or hides the 7-class live confidence bar chart |
| <kbd>M</kbd> | **Switch Inference Backend** | Instantly switches between **TFLite** and **Keras** on the fly without interrupting the stream |
| <kbd>H</kbd> | **Toggle Help Legend** | Displays or hides the hotkey reference bar at the bottom of the screen |
| <kbd>Q</kbd> / <kbd>ESC</kbd> | **Quit Application** | Safely releases webcam hardware, terminates background threads, and closes windows |

---

## 🧬 Model Architecture & Dataset

### Dataset: FER-2013 (Facial Expression Recognition 2013)
* **Image Dimensions**: $48 \times 48$ pixels
* **Color Mode**: Single-channel Grayscale
* **Classes**: 7 facial emotions (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
* **Dataset Characteristics**: Real-world unconstrained faces exhibiting head poses, occlusions, varying lighting conditions, and diverse age groups.

### Preprocessing Specifications (`model_metadata.json`)
```json
{
    "project_name": "FER-2013 Facial Emotion Recognition",
    "framework": "TensorFlow/Keras",
    "input_shape": [48, 48, 1],
    "color_mode": "grayscale",
    "normalization": "1.0 / 255.0",
    "class_names": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"],
    "num_classes": 7,
    "model_file": "emotion_model.keras"
}
```

### Deep CNN Structural Design:
1. **Feature Extraction Blocks**:
   * Repeated `Conv2D` layers with $3 \times 3$ kernels to extract hierarchical spatial features (edges $\to$ texture $\to$ facial contours $\to$ composite expressions).
   * `BatchNormalization` layers following convolutions to accelerate convergence and stabilize gradients.
   * `LeakyReLU` / `ReLU` non-linear activations.
   * `MaxPooling2D` ($2 \times 2$) for spatial downsampling and translation invariance.
   * `Spatial Dropout` ($0.25 - 0.40$) to prevent overfitting on subtle facial micro-patterns.
2. **Classification Head**:
   * `Flatten` / `GlobalAveragePooling2D`.
   * Fully connected `Dense` layers with `Dropout (0.5)`.
   * Final `Dense(7, activation='softmax')` output layer producing posterior probabilities $P(y_i \mid X)$.

---

## 📐 Mathematical & Algorithmic Foundations

### 1. Exponential Moving Average (EMA) Temporal Smoothing
Raw frame-by-frame neural network predictions on continuous video streams often suffer from high-frequency prediction jitter (e.g., fluctuating between *Neutral* and *Sad* within milliseconds).

To guarantee smooth, human-perceptible transitions without introducing visible lag, an EMA filter is applied to the probability distribution across consecutive frames:

$$\hat{P}_t = \alpha \cdot P_t + (1 - \alpha) \cdot \hat{P}_{t-1}$$

Where:
* $P_t \in \mathbb{R}^7$: The raw Softmax probability vector produced by the neural network at time $t$.
* $\hat{P}_{t-1} \in \mathbb{R}^7$: The filtered probability distribution vector from the preceding frame.
* $\hat{P}_t \in \mathbb{R}^7$: The smoothed prediction vector used for HUD visualization.
* $\alpha = 0.65$: The smoothing coefficient (smoothing factor).
  * If $\alpha \to 1.0$: Responsiveness is prioritized (instant updates, higher jitter).
  * If $\alpha \to 0.0$: Stability is prioritized (heavy inertia, delayed state changes).
  * $\alpha = 0.65$ provides the optimal empirical balance for facial video feeds at 30 FPS.

### 2. Softmax Normalization
The model's final dense layer converts unnormalized logits $z = [z_1, z_2, \dots, z_7]$ into a valid probability distribution:

$$P(y = i \mid \mathbf{x}) = \frac{e^{z_i}}{\sum_{j=1}^{7} e^{z_j}}, \quad \text{such that} \quad \sum_{i=1}^{7} P(y=i \mid \mathbf{x}) = 1.0$$

---

## 🎓 Viva & AI Interview Preparation Guide

This section compiles core theoretical and implementation questions commonly asked during academic viva defenses and technical machine learning interviews.

### Q1: Why convert facial crops to $48 \times 48$ single-channel Grayscale instead of RGB?
> **Answer**:
> 1. **Dimensionality & Throughput**: An RGB frame requires $48 \times 48 \times 3 = 6,912$ scalar values per face crop, whereas grayscale requires only $48 \times 48 \times 1 = 2,304$ values—a **66.7% reduction** in input dimensionality, significantly reducing multiply-accumulate (MAC) operations in the first convolutional layer.
> 2. **Invariance to Chromatic Bias**: Facial emotional expressions are expressed through geometric and structural deformations of facial musculature (e.g., zygomaticus major contractions during smiles, corrugator supercilii contractions during frowning). Chromatic data (skin tone, ambient lighting temperature) introduces spurious correlations and risks algorithmic bias without providing useful emotion signals.
> 3. **Dataset Alignment**: The benchmark FER-2013 dataset is inherently distributed as $48 \times 48$ single-channel grayscale arrays.

---

### Q2: Why does TensorFlow Lite (TFLite) outperform native Keras for real-time edge/desktop inference?
> **Answer**:
> * **Framework Overhead Elimination**: Standard Keras `model.predict()` or `model(tensor)` invokes Python runtime bindings, graph validation checks, tracing, and intermediate memory allocations on every invocation.
> * **XNNPACK Optimization**: TFLite uses the `XNNPACK` execution delegate—a library of highly optimized floating-point neural network inference operators specifically vectorized for modern x86 (AVX2/AVX-512) and ARM (NEON) SIMD instructions.
> * **Static Tensor Allocation**: The TFLite interpreter allocates all intermediate execution buffers during `allocate_tensors()` once at startup. It does not perform heap allocations during per-frame inference, completely eliminating garbage collection overhead.

---

### Q3: What is the purpose of Haar Cascade detection over deep learning detectors (e.g., MTCNN, RetinaFace) in this pipeline?
> **Answer**:
> * **Inference Speed**: OpenCV's Haar Cascade implementation utilizes **Integral Images** and AdaBoost cascade rejection trees to detect frontal faces in under **5 milliseconds** on standard CPUs.
> * **Resource Isolation**: By spending under 5ms on face localization, over 85% of CPU compute is preserved for the deep convolutional emotion classifier and HUD visual rendering, ensuring the stream comfortably maintains **30–60 FPS**.

---

### Q4: How does the application prevent visual jitter when multiple people are in the camera view?
> **Answer**:
> In multi-face scenarios, the primary face (index `0`, largest detected bounding box) is routed through the dedicated `TemporalSmoother` instance with history tracking, updating the live 7-class probability bar chart. Secondary faces are concurrently annotated with instantaneous bounding boxes and argmax labels, maintaining high frame rates without multiplying state tracking buffers.

---

### Q5: How do you handle class imbalance in FER datasets?
> **Answer**:
> In FER-2013, the *Disgust* class contains significantly fewer training samples (~547 images) compared to *Happy* (~8,989 images). Standard mitigation strategies include:
> 1. **Class-Weighted Loss Function**: Scaling categorical cross-entropy loss by inverse class frequencies:
>    $$w_c = \frac{N}{K \cdot N_c}$$
> 2. **Targeted Data Augmentation**: Applying horizontal flipping, subtle random rotations ($\pm 10^\circ$), zoom, and slight contrast jitter specifically targeting under-represented classes.
> 3. **Focal Loss**: Utilizing focal loss $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$ to down-weight easy examples and force the network to focus on hard, ambiguous negative examples.

---

## 🔧 Troubleshooting & FAQs

### 1. Camera fails to open (`[ERROR] Could not open webcam`)
* **Solution**: Check your webcam index. External webcams often register on index `1` or `2`:
  ```bash
  python realtime_emotion.py --camera 1
  ```
* **Windows Privacy Settings**: Ensure camera permissions are granted under **Windows Settings $\to$ Privacy & Security $\to$ Camera**.

### 2. Warning: `tf.lite.Interpreter is deprecated`
* **Details**: In recent TensorFlow builds, a notice is displayed suggesting migration to `ai_edge_litert`.
* **Action**: This is an informational deprecation warning for future TF 2.20+ releases. The current implementation uses the production-stable XNNPACK delegate and functions without issue.

### 3. Suppressing TensorFlow oneDNN Info Messages
* To clean up terminal output and suppress verbose oneDNN informational logs:
  * **PowerShell**:
    ```powershell
    $env:TF_ENABLE_ONEDNN_OPTS="0"
    $env:TF_CPP_MIN_LOG_LEVEL="2"
    python realtime_emotion.py
    ```
  * **Linux / macOS**:
    ```bash
    export TF_ENABLE_ONEDNN_OPTS=0
    export TF_CPP_MIN_LOG_LEVEL=2
    python realtime_emotion.py
    ```

---

## 🗺️ Future Roadmap

- [ ] **Landmark Mesh Integration**: Upgrade face localization to MediaPipe Face Mesh (468 landmarks) for head pose estimation (Yaw, Pitch, Roll).
- [ ] **Valence-Arousal Dimensional Mapping**: Map the 7 discrete emotions to continuous Circumplex Model coordinates (Valence vs Arousal).
- [ ] **Session Analytics & Telemetry**: Export CSV/JSON emotion trend reports over time for behavioral research and interview feedback.
- [ ] **Web / REST API Interface**: Add FastAPI streaming endpoint for browser-based remote emotion monitoring.

---

## 📜 License & Acknowledgments

- **License**: Released under the [MIT License](LICENSE).
- **Dataset**: Trained on the [FER-2013 Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) introduced by Pierre-Luc Carrier and Aaron Courville in 2013.
- **Computer Vision**: Powered by [OpenCV](https://opencv.org/) and [TensorFlow Lite](https://www.tensorflow.org/lite).
