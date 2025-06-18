#!/usr/bin/env python3
# src/viz/viz_combined.py

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from matplotlib import font_manager, rc
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, top_k_accuracy_score

# ─── 한글 폰트 설정 (Windows Malgun Gothic) ───────────────────────────────
font_path = "C:/Windows/Fonts/malgun.ttf"
if os.path.isfile(font_path):
    rc('font', family=font_manager.FontProperties(fname=font_path).get_name())
rc('axes', unicode_minus=False)
# ────────────────────────────────────────────────────────────────────────

# ==== 사용자 설정 ==========================================================
SEQ_NAME    = "L20"
WINDOW_SIZE = int(SEQ_NAME[1:])
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_DIR   = os.path.join(ROOT_DIR, 'models', SEQ_NAME)
DATA_DIR    = os.path.join(ROOT_DIR, 'dataset', 'npy', SEQ_NAME)
# ===========================================================================

# 1) 모델 & 클래스 로드
model = load_model(os.path.join(MODEL_DIR, 'sign_language_model_normalized.h5'))
labels = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
n_classes = len(labels)
all_idx   = np.arange(n_classes)
label2idx = {lbl:i for i, lbl in enumerate(labels)}

# 2) 데이터 로드 & y 추출
X = np.load(os.path.join(DATA_DIR, f'X_selected_{SEQ_NAME}_with_aug.npy'), allow_pickle=True)
y_raw = np.load(os.path.join(DATA_DIR, f'y_selected_pair_{WINDOW_SIZE}_with_aug.npy'), allow_pickle=True)

if np.issubdtype(y_raw.dtype, np.integer):
    y = y_raw
else:
    y = np.array([
        label2idx[pair[1]] if isinstance(pair, (list, tuple, np.ndarray))
        else label2idx[pair]
        for pair in y_raw
    ], dtype=int)

# 3) 예측
y_prob = model.predict(X, verbose=0)
y_pred = y_prob.argmax(axis=1)

# 4) 혼동 행렬 계산 및 정규화
cm      = confusion_matrix(y, y_pred, labels=all_idx)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

# 5) 클래스별 Top-3 정확도 계산
acc1 = accuracy_score(y, y_pred)
acc3 = top_k_accuracy_score(y, y_prob, k=3, labels=all_idx)

per_cls = {}
for idx, lbl in enumerate(labels):
    mask = (y == idx)
    if mask.sum() == 0:
        per_cls[lbl] = np.nan
    else:
        per_cls[lbl] = top_k_accuracy_score(
            y[mask], y_prob[mask], k=3, labels=all_idx
        )

df = pd.Series(per_cls).sort_values(ascending=False)
df_plot = df.fillna(0)

# ─── 6) 시각화 ───────────────────────────────────────────────────────────
plt.figure(figsize=(16,7))

# (A) 혼동 행렬
ax1 = plt.subplot2grid((1,2), (0,0))
sns.heatmap(
    cm_norm,
    ax=ax1,
    cmap='Blues',
    cbar_kws={'label':'정규화 빈도'},
    vmin=0, vmax=1.0,
    square=True,
    linewidths=0.3, linecolor='lightgray',
    xticklabels=labels, yticklabels=labels,
    annot=False
)
# 셀별 숫자 표시
th = cm_norm.max() / 2
for i in range(n_classes):
    for j in range(n_classes):
        val = cm[i,j]
        if val == 0: continue
        color = 'white' if cm_norm[i,j] > th else 'black'
        ax1.text(j+0.5, i+0.5, f"{val}", 
                 ha='center', va='center', color=color, fontsize=6)

ax1.set_title(f'Confusion Matrix ({SEQ_NAME})')
ax1.set_xlabel('Predicted')
ax1.set_ylabel('True')
ax1.set_xticklabels(labels, rotation=90, fontsize=7)
ax1.set_yticklabels(labels, rotation=0, fontsize=7)

# (B) 클래스별 Top-3 정확도
ax2 = plt.subplot2grid((1,2), (0,1))
df_plot.plot.barh(ax=ax2)
ax2.invert_yaxis()
ax2.set_title(f'Per-Class Top-3 Accuracy ({SEQ_NAME})')
ax2.set_xlabel('Top-3 Accuracy')
ax2.set_ylabel('')
ax2.tick_params(axis='y', labelsize=7)

plt.tight_layout()
plt.show()

# 7) 콘솔에 전반적인 성능도 출력
print(f"\n=== OVERALL ACCURACY ({SEQ_NAME}) ===")
print(f"Top-1 Accuracy: {acc1:.4f}")
print(f"Top-3 Accuracy: {acc3:.4f}")
