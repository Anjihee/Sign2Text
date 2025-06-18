# Sign2Text: 실시간 수어 인식 및 자막 생성기

---

## 📌 프로젝트 개요

**Sign2Text**는 실시간 웹캠 입력을 받아 사용자의 수어(手語, Sign Language)를 인식하고, 대응하는 단어를 자막 형태로 출력하는 시스템입니다.

주요 구성 요소는 다음과 같습니다:

- Mediapipe 기반 손 keypoint 추출  
- Conv1D + BiLSTM 기반 시퀀스 분류 모델  
- 실시간 예측 및 데이터 수집 인터페이스  
- 키보드 조작 기반 보강 데이터 저장 기능  

---


## 📁 프로젝트 구조

```
Sign2Text/
├── dataset/
│   ├── npy/
│   │   ├── L10/
│   │   │   ├── X_selected_L10.npy
│   │   │   └── y_selected_pair_10.npy
│   │   ├── L20/
│   │   │   └── …
│   └── augmented_samples/
│       ├── 단어1/
│       │   ├── raw_seq_단어1_1.npy
│       │   └── norm_seq_단어1_1.npy
│       └── …
├── models/
│   ├── L10/
│   │   ├── sign_language_model_normalized.h5
│   │   ├── label_classes.npy
│   │   ├── X_mean.npy
│   │   └── X_std.npy
│   └── …
├── src/
│   │── predict/       
│   ├── train/                          # 예측 테스트 코드 
│   │   ├── train_by_seq.py             # (보강 미포함 학습)
│   │   └── train_by_seq_aug.py         # (보강 포함 학습)
│   └── webcam/
│       └── realtime_infer_test.py      # 실시간 예측(데이터 수집 미포함)
│       └── webcam_predict.py           # 실시간 예측 및 데이터 수집 포함 (W)
└── README.md
```

---

## 🧠 모델 구조

- **입력**: `(시퀀스 길이, 114)`  
  - 양손 좌표 42개 + 양손 관절 각도 30개 = 총 114차원
- **네트워크**:
  - Conv1D → BatchNorm → Dropout  
  - Conv1D → BatchNorm → Dropout  
  - BiLSTM → Dropout  
  - Dense → Dropout → Output(Softmax)

---

## ⚙️ 시퀀스 설정 (`SEQ_NAME`)

```python
SEQ_NAME = "L10"
- train_by_seq or webcam 아래의 py 코드에서 SEQ_NAME 변수를 L10,20,30,40,50으로 변경하여 시퀀스별 테스트 가능합니다.
```
---

## 🥮 모델 및 데이터 정보

* **입력 feature**: 총 114차원 (손 좌표 84 + 관절 각도 30)
* **시퀀스 길이**: 10 프레임 (Conv1D + LSTM 모델 입력)
* **모델 구조**: Conv1D + LSTM
* **정규화 유무**: 시험 병행 (정규화 O/X 모델 각각 존재)
* **코사인 유사도 분석**:
  * `label_similarity_filter.py`를 통해 각 라벨의 평균 벡터 간 유사도 계산
  * 평균 유사도 분포 시각화 (히스토그램, 산점도)
  * 유사도가 낮은 라벨 30개 선별 가능 → 학습셋 구성에 활용
  * 유사도 분석에 활용한 데이터셋인 merged_with_angles.csv은 용량 이슈로 인하여 추후 구글 드라이브로 업로드 예정

---

## ✅ 모델 파일 안내

- `src/train/train_by_seq.py` 실행 시: 보강 없이 학습된 모델이 `models/L##/` 폴더에 저장됨
- `src/train/train_by_seq_aug.py` 실행 시: 보강 데이터를 포함하여 학습된 모델이 저장됨
- 각 모델 폴더(`models/L10/`, `models/L20/` 등)에는 아래 파일들이 포함됨:

---

## 💬 시스템 시험 방법 (웹캠 실시간 예측)

- 다음 명령어로 실시간 예측 시스템을 실행:

```bash
python src/infer/webcam_predict.py
```
- 내부 코드의 SEQ_NAME = "L10" 값을 변경하여 시퀀스 길이 및 모델을 선택 가능
