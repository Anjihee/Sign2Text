import os
import glob
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical

# ==== 설정 ====
SEQ_NAME = "L20"  # 예: 'L10', 'L20', 'L30'...
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))  # Sign2Text/
DATASET_DIR = os.path.join(BASE_DIR, 'dataset/npy', SEQ_NAME)
MODEL_DIR = os.path.join(BASE_DIR, 'models', SEQ_NAME)
AUG_DIR = os.path.join(BASE_DIR, 'dataset/augmented_samples')
os.makedirs(MODEL_DIR, exist_ok=True)

# ==== 1. 원본 데이터 로드 ====
X = np.load(os.path.join(DATASET_DIR, f'X_selected_{SEQ_NAME}.npy'))
y_pairs = np.load(os.path.join(DATASET_DIR, f'y_selected_pair_{SEQ_NAME[1:]}.npy'), allow_pickle=True)
y_raw = np.array([label for _, label in y_pairs])

# ==== 2. 정규화 ====
X_mean = np.mean(X, axis=(0, 1), keepdims=True)
X_std = np.std(X, axis=(0, 1), keepdims=True) + 1e-6
X_normalized = (X - X_mean) / X_std
np.save(os.path.join(MODEL_DIR, 'X_mean.npy'), X_mean)
np.save(os.path.join(MODEL_DIR, 'X_std.npy'), X_std)

# ==== 3. 보강 데이터 통합 ====
expected_shape = X_normalized.shape[1:]   # e.g. (window_size, 114)
X_aug_list, y_aug_list = [], []

if os.path.isdir(AUG_DIR):
    for label_name in os.listdir(AUG_DIR):
        folder = os.path.join(AUG_DIR, label_name)
        if not os.path.isdir(folder):
            continue
        for fn in glob.glob(os.path.join(folder, 'norm_seq_*.npy')):
            seq = np.load(fn)
            if seq.ndim == 3 and seq.shape[0] == 1:
                seq = seq.squeeze(0)
            if seq.shape != expected_shape:
                print(f"⚠️ 스킵됨: {fn} (shape {seq.shape} != {expected_shape})")
                continue
            X_aug_list.append(seq)
            y_aug_list.append(label_name)

    if X_aug_list:
        X_aug = np.stack(X_aug_list, axis=0)
        X_normalized = np.concatenate([X_normalized, X_aug], axis=0)
        y_raw = np.concatenate([y_raw, np.array(y_aug_list)], axis=0)
        print(f"✅ 보강 샘플 {len(X_aug_list)}개 추가됨")

# ==== 4. 라벨 인코딩 및 one-hot ====
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
y = to_categorical(y_encoded)

# ==== 5. 클래스 가중치 ====
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_encoded), y=y_encoded)
class_weight_dict = dict(enumerate(class_weights))

# ==== 6. 학습/검증 분할 ====
X_train, X_val, y_train, y_val = train_test_split(
    X_normalized, y, test_size=0.2, stratify=y_encoded, random_state=42
)

# ==== 7. 모델 정의 ====
input_shape = X_normalized.shape[1:]
model = Sequential([
    Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
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

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

# ==== 8. 학습 ====
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    class_weight=class_weight_dict
)

# ==== 9. 저장 ====
import pickle
history_path = os.path.join(MODEL_DIR, f'history_{SEQ_NAME}.pkl')
with open(history_path, 'wb') as f:
    pickle.dump(history.history, f)

model.save(os.path.join(MODEL_DIR, 'sign_language_model_normalized.h5'))
np.save(os.path.join(MODEL_DIR, 'label_classes.npy'), label_encoder.classes_)
print("✅ 모델 및 클래스 저장 완료")