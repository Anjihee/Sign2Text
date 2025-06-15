# 📄 test_eval.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import load_model

# === 경로 설정 ===
X = np.load('../dataset/X_selected.npy')
y_pairs = np.load('../dataset/y_selected_pair.npy', allow_pickle=True)
label_classes = np.load('../models/label_classes.npy', allow_pickle=True)
model = load_model('../models/sign_language_model_improved.h5')

# === 라벨 처리 ===
labels = np.array([label for _, label in y_pairs])
le = LabelEncoder()
le.classes_ = label_classes
y_encoded = le.transform(labels)

# === 검증 셋 분리 ===
_, X_val, _, y_val_encoded = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === 예측 ===
y_pred_prob = model.predict(X_val, verbose=0)
y_pred_encoded = np.argmax(y_pred_prob, axis=1)

# === 리포트 출력 ===
print("\n📊 Classification Report:")
print(classification_report(y_val_encoded, y_pred_encoded, target_names=le.classes_))

# === Confusion Matrix 시각화 ===
cm = confusion_matrix(y_val_encoded, y_pred_encoded)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()