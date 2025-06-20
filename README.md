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

# ✋📈 보강(증강) 데이터 & 실시간 인식 개선

---

| 파일 | 수정 및 구현 포인트 |
|------|-----------------------|
| **`src/webcam/webcam_test.py`** | `w` 키로 웹캠 프레임 → `raw_seq_*.npy` / `norm_seq_*.npy` 자동 저장<br>· 저장 경로 `dataset/augmented_samples/<label>/` / 라벨당 **≈30샘플** 수집 권장 |
| **`src/train_by_seq_aug.py`**  | 보강 시퀀스를 **원본 데이터와 통합 학습**<br>· `augmented_samples` 스캔 → shape 검증 후 `np.stack`<br>· 정규화 파라미터 저장(`X_mean.npy`, `X_std.npy`) & `EarlyStopping(patience=12)` |
| **`src/webcam/realtime_infer_test.py`** | **p 키 제거 → 완전 자동 예측** 로직 구현<br>· *손 사라짐 2 s ▶ predict* → 결과 4 s 표시 & 실시간 업데이트<br>· *손 사라짐 1 s ▶ sequence.reset()* → 깜빡임 무시<br>· **97~140 line**: 제스처 감지·결과 출력 핵심 (아래 상세) |

---

## 1 · 데이터 보강 흐름

A[한국수어사전 원본 영상] -->|학습용 모션 인식| B(webcam_test.py)
B -->|w 키로 저장| C[raw_seq_<label>_n.npy (20×114)]
B --> D[norm_seq_<label>_n.npy ((raw - μ) / σ)]
C & D --> E[dataset/augmented_samples/<label>/]
E --> F(train_by_seq_aug.py)

### 수집 절차

1. `python webcam_test.py` → **s** 수집 → 수어 시연  
2. **w** 저장 (라벨당 ≈ 30 샘플)  
3. `train_by_seq_aug.py` 학습 → `models/L20/…h5` 갱신 (SEQ_NAME으로 시퀀스 조정 가능)
4. `realtime_infer_test.py` 로 실시간 검증  

---

## 2 · `webcam_test.py` 발췌 (보강 저장)

```python
if key == ord("w"):
    save_dir = base_dir/"dataset"/"augmented_samples"/CURRENT_LABEL
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir/f"raw_seq_{CURRENT_LABEL}_{cnt}.npy",  seq_arr)
    np.save(save_dir/f"norm_seq_{CURRENT_LABEL}_{cnt}.npy", (seq_arr-X_mean)/X_std)
```

## 3 · `train_by_seq_aug.py` 통합 로직

```python
aug_dir = DATASET_DIR/"augmented_samples"
for label in aug_dir.iterdir():
    for fn in label.glob("norm_seq_*.npy"):
        seq = np.load(fn).squeeze(0)   # (1,20,114) → (20,114)
        if seq.shape == expected_shape:
            X_aug_list.append(seq)
            y_aug_list.append(label.name)
X_normalized = np.concatenate([X_normalized, np.stack(X_aug_list)], axis=0)
y_raw        = np.concatenate([y_raw, np.array(y_aug_list)], axis=0)
```

---

## 4 · 실시간 인식 개선 (`realtime_infer_test.py`)

### 자동 제스처 감지 & 결과 출력 (코드 97~140 line)

```python
# ① s 로 수집 시작 → collecting = True
if collecting and hand_detected and not hand_was_detected:
    gesture_active = True           # 제스처 시작
    sequence.clear()                # 새 버퍼

# ② 제스처 동안 손 사라지면 첫 타임스탬프 기록
if gesture_active:
    if not hand_detected:
        hands_gone_at = hands_gone_at or time.time()
    else:
        hands_gone_at = None

# ③ hands_gone_at 유지 시간이 2 s 이상 → model.predict()
if collecting and gesture_active and hands_gone_at \
   and time.time() - hands_gone_at >= 2:
    predict()                       # Conv1D‑BiLSTM 예측
    display_mode  = True            # 결과 4 s 표시
    display_timer = time.time()

# ④ 표시 종료(4 s) 또는 새 손 등장 시 최신 결과로 업데이트
if display_mode and time.time() - display_timer >= 4:
    display_mode = False
    latest_text  = ""
```

**핵심**
* p 키 제거 → 자동 타임스탬프 트리거 (`hands_gone_at`)
* 결과 유지 4 s → 이후 새 제스처 시 즉시 update
* 손 사라짐 1 s (`sequence.clear()`) → 버퍼 깔끔 리셋 (깜빡임 무시)


- `src/train/train_by_seq.py` 실행 시: 보강 없이 학습된 모델이 `models/L##/` 폴더에 저장됨
- `src/train/train_by_seq_aug.py` 실행 시: 보강 데이터를 포함하여 학습된 모델이 저장됨
- 각 모델 폴더(`models/L10/`, `models/L20/` 등)에는 아래 파일들이 포함됨:

## 5 · 작업 순서

1. 한국수어사전 영상 시청 & 모션 연습  
2. **webcam_test.py** → `s` 수집 → `w` 저장 × 30회  
3. **train_by_seq_aug.py** 로 통합 학습  
4. **realtime_infer_test.py** 로 실시간 성능 확인  

---

## 6 · 참고

* macOS 내장 카메라 = `VideoCapture(1)` 필요 가능성
