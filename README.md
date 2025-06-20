# Sign2Text: Real-time Sign Language Recognition with Subtitle Generation

---

## 📌 Overview

**Sign2Text** is a real-time sign language recognition system that captures webcam input and displays corresponding words as subtitles.  
It is designed to assist communication by recognizing hand gestures using deep learning and computer vision.

### 🔧 Key Features

- **Auto prediction** without button press  
- **MediaPipe**-based hand keypoint extraction  
- **Conv1D + BiLSTM** sequence classification  
- **Top-3 predictions** with confidence scores  
- **Temperature scaling** for stable outputs  
- Real-time **PyQt5 GUI** (no keyboard interaction)  
- Data augmentation & evaluation support

---

## 📁 Project Structure

```
Sign2Text/
├── dataset/
│   ├── npy/
│   └── augmented_samples/
├── models/
│   └── L10/, L20/, ...
│       ├── sign_language_model_normalized.h5
│       ├── label_classes.npy
│       ├── X_mean.npy
│       └── X_std.npy
├── src/
│   ├── dataset_preprocessing/
│   │   ├── add_angles_to_merged.py
│   │   ├── batch_generate_csv.py
│   │   ├── create_total_seq.py
│   │   ├── merge_csv.py
│   │   └── zip_csv.py
│   ├── hold_out_test/
│   │   ├── holdout_test.py         # Run predictions on unseen samples
│   │   ├── auto_infer.py
│   │   ├── make_test_labels.py
│   │   ├── holdout_results.csv
│   │   └── test_labels.csv
│   ├── predict/
│   │   ├── predict_test_sample.py
│   │   ├── predict_test_sample_normalized.py
│   │   └── label_similarity_filter.py
│   ├── train/
│   │   ├── train_by_seq.py
│   │   └── train_by_seq_aug.py
│   ├── viz/
│   │   ├── merge_aug_origin_npy.py
│   │   ├── viz_confusion_top3.py
│   │   └── viz_history.py
│   └── webcam/
│       ├── webcam_test.py               # For data collection
│       ├── realtime_infer_test.py       # Lightweight prediction only
│       └── sign2text_gui.py             # PyQt5-based GUI app (Main App)
├── requirements.txt
└── README.md
```
---

## 🎮 GUI App (PyQt5)

### ▶️ Run the Real-time GUI App

```bash
python src/webcam/sign2text_gui.py
```

- Webcam preview + Korean font rendering
- Left panel: video feed
- Right panel:
  - Status (`대기 중` / `수집 중`)
  - Top-3 predictions (with confidence)
  - Result display (`신뢰도 부족` if below threshold)
- Buttons:
  - `수집 시작`: toggles sample collection

> 🔥 Temperature scaling and confidence thresholding included  
> 🔁 Sequence is auto-cleared after prediction

### 📸 Real-time Output Screenshots

![Image](https://github.com/user-attachments/assets/d05a8879-8151-4a6f-9e51-4b5ebebfeed3)  
<sub>*Predicting in real time (Top-3 result view)*</sub>

![Image](https://github.com/user-attachments/assets/1ee3861a-c134-4525-8335-0e06af8335aa)  
<sub>*Real-time GUI showing label “식당” recognized with high confidence*</sub>

---

## 🧠 Model Architecture

- **Input Shape**: `(sequence_length, 114)`
  - 84 keypoint features (21 points × 2 hands)
  - 30 joint angles
- **Layers**:
  - Conv1D → BatchNorm → Dropout  
  - Conv1D → BatchNorm → Dropout  
  - BiLSTM → Dropout  
  - Dense → Dropout → Softmax

---

## ⚙️ Sequence Configuration

```python
SEQ_NAME = "L20"
```

- `SEQ_NAME` defines the window size
- Supported values: `"L10"`, `"L20"`, etc.
- Make sure the model and `X_mean.npy`, `X_std.npy` in `models/L##` match this name

---
 ---

## 📂 Learning Data Composition

The Sign2Text project was trained on a curated dataset containing **61 sign language labels**, constructed from both the original AI Hub data and newly augmented samples. The distribution of labels is as follows:

| Category          | Label Count | Description |
|-------------------|-------------|-------------|
| 📦 Original only   | 31 labels   | Labels that exist only in the original dataset (`.npy` without augmentation) |
| 🔁 Common (Shared) | 19 labels   | Labels included in both the original and augmented data |
| ➕ Augmented only  | 11 labels   | Newly added labels from webcam-based augmentation |

---

### ✅ Label Lists by Type

#### 📦 Original-only (31)
`밥솥`, `출근`, `퇴사`, `포켓`, `여아`, `학업`, `여학교`, `백수`, `채팅`, `신학`,  
`뉴질랜드`, `남아`, `독서실`, `유학`, `국어학`, `다과`, `의학`, `위스키`, `울산`, `구직`,  
`학교연혁`, `문학`, `예습`, `사직`, `친아들`, `벌꿀`, `배드민턴`, `버스값`, `식당`, `월세`

#### 🔁 Common (Original + Augmented, 19)
`감기`, `개학`, `경찰서`, `독서`, `독일어`, `라면`, `병문안`, `보건소`, `수면제`, `술`,  
`슬프다`, `싫어하다`, `커피`, `콜라`, `퇴원`, `치료`, `학교`, `입원`, `월세`

#### ➕ Augmented-only (11)
`꿀`, `나(1인칭)`, `너(2인칭)`, `딸`, `아들`, `안녕하세요`, `영어`, `운동`, `입사`, `좋다`

---

### 🖼️ Visual Summary
<img width="707" alt="Image" src="https://github.com/user-attachments/assets/efdbad31-5711-4c92-bf4e-8626f7356886" />

*A Venn diagram showing the overlap between original and augmented label sets.*

---

### 🧠 Key Findings

- **Only 30 augmented labels** (common + augmented-only) were recognized reliably in real-time testing.
- **Original-only labels were not recognized**, even if included during training.
- This suggests that **data recency and augmentation quality** have a stronger impact on performance than just label presence.
- Labels with freshly collected webcam samples showed significantly higher prediction confidence.

---

## 🥕 Data Augmentation Workflow

1. Run:

```bash
python src/webcam/webcam_test.py
```

2. Press `s` → show gesture  
3. Press `w` → save `raw_seq_*.npy` and `norm_seq_*.npy`  
4. Data saved at: `dataset/augmented_samples/<label>/`

---

## 🧪 Train with Augmented Data

```bash
python src/train/train_by_seq_aug.py
```

- Merges raw + augmented samples
- Saves model and normalization stats to `models/L##/`

To train without augmentation:

```bash
python src/train/train_by_seq.py
```

---

## 📈 Evaluate on Hold-out Set

```bash
python src/hold_out_test/holdout_test.py
```

- Loads samples from `videos/`, uses `test_labels.csv`
- Outputs to `holdout_results.csv`
- Visualize results with:

```bash
python src/viz/viz_confusion_top3.py
```

---

## 🔍 Label Similarity Analysis

```bash
python src/predict/label_similarity_filter.py
```

- Computes cosine similarity between mean label vectors
- Useful to identify confusing signs
- Input: merged dataset with angles

---

## 📊 Visualization Tools

- `viz_history.py`: plot training history
- `viz_confusion_top3.py`: visualize confusion matrix
- `merge_aug_origin_npy.py`: merge original/augmented samples for comparison

---

## 📝 Notes

- On macOS, use `cv2.VideoCapture(1)` if `0` doesn't work
- Use Korean font: `AppleGothic` or `malgun.ttf` for readable text
- Recommended: collect 30+ samples per label for robust accuracy

---

## 🧩 Development Environment

- **OS & Tools**:
  - Windows 10 /  macOS (Apple Silicon)
  - Anaconda (Python 3.8.20)
  - Visual Studio Code

- **Core Libraries**:
  - `numpy==1.24.3`
  - `pandas==2.0.3`
  - `opencv-python==4.11.0.86`
  - `mediapipe==0.10.11`
  - `scipy==1.10.1`
  - `tqdm==4.64.1`
  - `pillow==10.4.0`
  - `tensorflow==2.13.0`
  - `keras==2.13.1`
  - `PyQt5==5.15.9`
 
---

## 👥 Team Members

| Role     | Name (GitHub)                                | Responsibility                | Details                                                        |
|----------|----------------------------------------------|-------------------------------|----------------------------------------------------------------|
| 🧑‍💼 Team Lead| [An Jihee](https://github.com/Anjihee)         | Modeling, Real-time Inference System | Built Conv1D + BiLSTM model and PyQt5-based GUI for real-time sign recognition |
| 👩‍💻 Member| [Kim Minseo](https://github.com/oweenia03)     | Data Collection, Preprocessing, Evaluation | Extracted raw keypoints, constructed labeled CSVs, and participated in testing |
| 👩‍💻 Member| [Lee Jimin](https://github.com/leejm429)       | Data Augmentation, Evaluation | Generated augmented sequences and conducted hold-out testing   |

---

## 📎 License

This project is part of the **Open Source Programming** course at **Sookmyung Women's University**.  
It uses [MediaPipe](https://github.com/google/mediapipe) and [TensorFlow](https://www.tensorflow.org/) under the Apache 2.0 License.
