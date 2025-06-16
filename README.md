Sign2Text - 실시간 수어 인식 시스템


⸻

ᴛ 프로젝트 구조

dataset/
  ├─ X_selected.npy            # (사용 선택된 데이터: N, 10, 114)
  ├─ y_selected_pair.npy       # (라벨 정보: (word_id, label))
  ├─ test_sample.npy           # 다크파일 시험용 샘플
  ├─ webcam_seq_raw.npy        # 실시간 입력 훌점 저장 (X_mean, X_std 적용 전)
  └─ webcam_seq_norm.npy       # 정규화 받은 입력

models/
  ├─ sign_language_model_normalized.h5  # 정규화 후 목표
  ├─ sign_language_model_improved.h5    # 정규화 X 목표
  ├─ label_classes.npy         # class index → word_id 및 label
  ├─ X_mean.npy / X_std.npy    # 정규화 값

src/
  ├─ train_lstm_conv1d_selected.py      # 정규화 X 후 모델링
  ├─ train_lstm_conv1d_normalized.py    # 정규화 O 모델링
  ├─ predict_test_sample.py             # 정규화 X 샘플 테스트
  ├─ predict_test_sample_normalized.py  # 정규화 O 샘플 테스트
  ├─ webcam_infer_selected.py           # 정규화 X 실시간
  ├─ webcam_infer_normalized.py         # 정규화 O 실시간


⸻

ᴛ 현재 문제 (개요)
	•	테스트에서는 괜찮은 성능을 보여주지만, 실시간 입력에서는 유영적으로 ‘입원’ 만 예측됨
	•	반복적으로 가장 많이 나오는 단어: 입원 / 수면제 / 술 등
	•	webcam 입력과 테스트 데이터가 보유하는 값의 범위가 다르면서, 부작용적인 구도 보존


⸻

ᴛ 보조 요청
	•	Mediapipe 입력 조정과 보석 규칙 업그레이드
	•	데이터 관련처리 및 연결고정 정보와 의 복합
	•	시그니처 데이터가 정확히 저장이 되는지 판단 보안
	•	해당 npy 데이터를 통한 설정 자바인드만 바꾸어 보기

⸻

📎 특징
	•	LSTM + Conv1D 구조
	•	114차원 feature (손 키포인트 + 각도)
	•	시퀀스 10 프레임
	•	정규화 O/X 실험 병행