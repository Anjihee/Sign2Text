# src/train_conv1d.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# 📁 경로 설정
DATA_PATH = os.path.join("..", "dataset", "filtered_sentence_data.csv")
MODEL_SAVE_PATH = os.path.join("..", "models", "cnn1d_model.h5")
ENCODER_SAVE_PATH = os.path.join("..", "models", "label_encoder.pkl")

# 📂 1. 데이터 로딩
df = pd.read_csv(DATA_PATH)

# 🧼 2. X, y 분리 및 전처리
drop_cols = ["label"]
if "word_id" in df.columns:
    drop_cols.append("word_id")

X = df.drop(columns=drop_cols).astype(np.float32).values
X = np.nan_to_num(X)
X = X.reshape(-1, 84)  # 21 keypoints * 2 hands * (x, y)

# ✅ 정규화 함수
def normalize_relative(X):
    X_normalized = []
    for row in X:
        lx0, ly0 = row[0], row[21]   # 왼손 기준점
        rx0, ry0 = row[42], row[63]  # 오른손 기준점

        lx = row[0:21] - lx0
        ly = row[21:42] - ly0
        rx = row[42:63] - rx0
        ry = row[63:84] - ry0

        X_normalized.append(np.concatenate([lx, ly, rx, ry]))
    return np.array(X_normalized)

# ✅ 정규화 적용
X = normalize_relative(X)
X = X.reshape(-1, 84, 1)  # Conv1D 입력 형식

# 🏷️ 라벨 처리
y = df["label"].values
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# 📊 3. 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, stratify=y_encoded, random_state=42
)

# 🧱 4. 모델 구성
model = Sequential([
    Conv1D(128, kernel_size=5, activation='relu', input_shape=(84, 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(256, kernel_size=3, activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    LSTM(128, return_sequences=False),
    Dense(512, activation='relu'),
    Dropout(0.4),

    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 5. 학습
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

# 📈 6. 평가 및 저장
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ 테스트 정확도: {acc:.4f}")

model.save(MODEL_SAVE_PATH)
joblib.dump(le, ENCODER_SAVE_PATH)
print(f"💾 모델 저장 완료: {MODEL_SAVE_PATH}")
print(f"💾 라벨 인코더 저장 완료: {ENCODER_SAVE_PATH}")