import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# 경로 설정
csv_path = "/Users/jihee/Documents/2025/2025PJ/OSP/Sign2Text/dataset/filtered_data.csv"
output_dir = "/Users/jihee/Documents/2025/2025PJ/OSP/Sign2Text/dataset"

# 하이퍼파라미터
sequence_length = 15  # 시퀀스 길이

# 데이터 로드
df = pd.read_csv(csv_path)

# 열 자동 감지
feature_cols = [col for col in df.columns if col not in ("label", "word_id")]
label_col = "label"

# 시퀀스 생성
sequences = []
labels = []

current_label = None
current_seq = []

for i, row in df.iterrows():
    label = row[label_col]
    features = row[feature_cols].values.astype(np.float32)

    if current_label is None:
        current_label = label

    if label == current_label:
        current_seq.append(features)
        if len(current_seq) == sequence_length:
            sequences.append(np.stack(current_seq))
            labels.append(label)
            current_seq = []
    else:
        current_label = label
        current_seq = [features]

# 배열로 변환
X = np.array(sequences)  # shape: (n, 15, 84)
le = LabelEncoder()
y = le.fit_transform(labels)

# 저장
np.save(os.path.join(output_dir, "X_seq.npy"), X)
np.save(os.path.join(output_dir, "y_encoded.npy"), y)
np.save(os.path.join(output_dir, "y_labels.npy"), np.array(labels))

print("✅ 시퀀스 데이터셋 생성 완료")
print(f"🧩 X shape: {X.shape}, y shape: {y.shape}")
print(f"🏷️ 라벨 예시: {np.unique(labels)[:10]}")