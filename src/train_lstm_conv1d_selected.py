# 📄 train_lstm_conv1d_improved.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
import os

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '../dataset')
MODEL_DIR = os.path.join(BASE_DIR, '../models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 데이터 로드
X = np.load(os.path.join(DATASET_DIR, 'X_selected.npy'))
y_pairs = np.load(os.path.join(DATASET_DIR, 'y_selected_pair.npy'), allow_pickle=True)

# 라벨 추출 및 인코딩
y_raw = np.array([label for _, label in y_pairs])
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y = to_categorical(y_encoded)

# class weight 계산
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))

# train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y_encoded, random_state=42
)

# 모델 정의
model = Sequential([
    Conv1D(64, 3, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
    BatchNormalization(),
    Dropout(0.3),
    
    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    Dropout(0.3),

    Bidirectional(LSTM(128)),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')
])

# 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 콜백
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)

# 학습
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbackㅅs=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# 저장
model.save(os.path.join(MODEL_DIR, 'sign_language_model_improved.h5'))
np.save(os.path.join(MODEL_DIR, 'label_classes.npy'), label_encoder.classes_)