# Sign2Text - 실시간 수어 인식 시스템

---

## 📁 프로젝트 구조

```
dataset/
├─ X_selected.npy              # (N, 10, 114) - 선택된 학습용 feature 시퀀스
├─ y_selected_pair.npy         # (N, 2) - 각 시퀀스의 (word_id, label) 라벨 정보
├─ test_sample.npy             # 정규화 전 테스트용 단일 샘플
├─ webcam_seq_raw.npy          # 실시간 웹캠 입력 저장 시퀀스 (정규화 전)
├─ webcam_seq_norm.npy         # 실시간 웹캠 입력 저장 시퀀스 (정규화 후)

models/
├─ sign_language_model_normalized.h5    # 정규화된 데이터로 학습된 모델
├─ sign_language_model_improved.h5      # 정규화 없이 학습된 모델
├─ label_classes.npy                    # 인덱스 → word_id 매핑 정보
├─ X_mean.npy / X_std.npy               # 학습용 feature의 정규화 파라미터

src/
├─ train_lstm_conv1d_selected.py        # 정규화 X 모델 학습 스크립트
├─ train_lstm_conv1d_normalized.py      # 정규화 O 모델 학습 스크립트
├─ predict_test_sample.py               # 정규화 X 예측
├─ predict_test_sample_normalized.py    # 정규화 O 예측
├─ webcam_infer_selected.py             # 실시간 예측 (정규화 X)
├─ webcam_infer_normalized.py           # 실시간 예측 (정규화 O)
├─ label_similarity_filter.py           # 라벨별 평균 feature 벡터 기반 코사인 유사도 계산 및 시각화
```

---

## 🔍 문제 개요 (현황 공유)

* 테스트 데이터세트에서는 예측 정확도가 높음 (Top-1 정답률 양호)
* **실시간 웹캠 입력에서는 '입원'이 반복적으로 예측되는 문제 발생**

  * 입원 / 술 / 수면제 등 일부 클래스에 치우치는 경향
* 웹캠에서 추출한 feature 값 분포와 학습 데이터 feature 값 분포가 상이한 것으로 추정
* 예측 시퀀스가 유효하지 않은 상태(정지 프레임 포함)에서도 예측을 시도할 수 있음

---

## 💪 필요 행동 / 협조 요청

* [ ] **MediaPipe 좌표점 기준 정규화 방식 재검토**
* [ ] **실시간 입력값과 학습값 비교** 및 시각화 필요
* [ ] 웹캠 입력 시 저장된 `webcam_seq_*.npy` 데이터를 통해 **직접 모델에 넣고 테스트 가능**
* [ ] 예측 신뢰도(CONF_THRESHOLD) 및 시간 간격(INFERENCE_INTERVAL)도 시험 필요
* [ ] **코사인 유사도 기반 라벨 간 유사도 분석 시각화 활용 (`label_similarity_filter.py`)**

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

---

## ✅ 모델 파일 안내
* `src/train_lstm_conv1d_normalized.py` 또는 `src/train_lstm_conv1d_selected.py` 실행 시,
  자동으로 `models/` 폴더에 모델 생성됨
* 필요시 `label_classes.npy`, `X_mean.npy`, `X_std.npy`는 이미 포함되어 있음

---

## 💬 시스템 시험 방법

* `src/webcam_infer_normalized.py` 실행 (정규화된 모델 기준)
* 키보드 `s` 입력 시, 현재 10프레임 시퀀스를 `.npy`로 저장

  * `dataset/webcam_seq_raw.npy`
  * `dataset/webcam_seq_norm.npy`

* 저장된 `.npy`는 테스트용 코드에 주입하여 활용 가능

  * 예: `predict_webcam_sequence.py`
