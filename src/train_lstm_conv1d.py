import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# 1. 데이터 불러오기
X = np.load("../dataset/X.npy")
y = np.load("../dataset/y.npy")

# 2. 라벨 인코딩 + 원-핫
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_onehot = to_categorical(y_encoded)

# 3. train/val 분할
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded
)

# 4. 모델 설계 (Conv1D + LSTM)
model = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    Conv1D(128, 3, activation='relu'),
    Dropout(0.3),
    LSTM(128),
    Dense(64, activation='relu'),
    Dense(y_onehot.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. 학습
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# 6. 모델 및 인코더 저장
model.save("models/conv1d_lstm_model.h5")
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ 모델 학습 및 저장 완료")
