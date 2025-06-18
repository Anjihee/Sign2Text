## 정규화 모델 테스트
import numpy as np
from tensorflow.keras.models import load_model
import os

# 🔧 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')
DATASET_DIR = os.path.join(BASE_DIR, '../dataset')

# 🔄 모델 및 메타데이터 로드
model = load_model(os.path.join(MODEL_DIR, 'sign_language_model_normalized.h5'))
label_classes = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
X_mean = np.load(os.path.join(MODEL_DIR, 'X_mean.npy'))
X_std = np.load(os.path.join(MODEL_DIR, 'X_std.npy'))
X = np.load(os.path.join(DATASET_DIR, 'X_selected.npy'))           # shape = (N, 10, 114)
y_selected_pair = np.load(os.path.join(DATASET_DIR, 'y_selected_pair.npy'), allow_pickle=True)
id2label = {wid: lbl for wid, lbl in y_selected_pair}

# 🔍 테스트할 샘플 인덱스 목록
test_indices = [280, 434, 39, 417, 585]

for i in test_indices:
    sample = X[i]                              # shape = (10, 114)
    label_id = y_selected_pair[i][0]
    true_label = id2label.get(label_id, label_id)

    # 정규화 및 reshape
    norm_sample = (sample - X_mean) / X_std
    norm_sample = norm_sample.reshape(1, 10, 114)

    # 예측
    y_pred = model.predict(norm_sample, verbose=0)[0]
    top3_idx = y_pred.argsort()[-3:][::-1]

    print(f"\n🧪 샘플 인덱스 {i} (정답: {true_label})")
    print("🔍 TOP 3 예측 결과:")
    for idx in top3_idx:
        pred_wid = label_classes[idx]
        pred_label = id2label.get(pred_wid, pred_wid)
        print(f"  {pred_label}: {y_pred[idx]:.3f}")