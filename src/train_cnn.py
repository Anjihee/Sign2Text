# src/train_cnn.py

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# 📁 경로 설정
DATA_PATH = os.path.join("..", "dataset", "merged_labeled_vectors.csv")
MODEL_SAVE_PATH = os.path.join("..", "models", "cnn_model.h5")
ENCODER_SAVE_PATH = os.path.join("..", "models", "label_encoder.pkl")

# 📂 1. 데이터 로딩
print("📂 CSV 데이터 로딩 중...")
df = pd.read_csv(DATA_PATH)

# 🧼 2. 입력(X), 라벨(y) 분리 + 전처리
# ⚠️ 'word_id' 제외, float 변환 + NaN 제거
X = df.drop(columns=["label", "word_id"]).astype(np.float32).values
X = np.nan_to_num(X)

y = df["label"].values

# 🧠 3. 라벨 인코딩 + One-hot 변환
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# 📊 4. 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

# 🧱 5. 모델 구성 (MLP 형태)
model = Sequential([
    Dense(128, activation='relu', input_shape=(84,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y_onehot.shape[1], activation='softmax')  # 클래스 수만큼 출력
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 6. 학습 시작
print("🚀 모델 학습 시작...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop]
)

# 📈 7. 평가
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ 테스트 정확도: {acc:.4f}")

# 💾 8. 모델 및 인코더 저장
model.save(MODEL_SAVE_PATH)
joblib.dump(le, ENCODER_SAVE_PATH)

print(f"💾 모델 저장 완료: {MODEL_SAVE_PATH}")
print(f"💾 라벨 인코더 저장 완료: {ENCODER_SAVE_PATH}")