#!/usr/bin/env python3
# create_new_npy.py

import os
import numpy as np

# ==== 설정 ====
SEQ_NAME = "L20"
WINDOW   = int(SEQ_NAME[1:])

# ==== 경로 ====
ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BASE_NPY = os.path.join(ROOT, 'dataset', 'npy', SEQ_NAME)
AUG_DIR  = os.path.join(ROOT, 'dataset', 'augmented_samples')

# ==== 1) 원본 npy 로드 & 레이블 인덱스 변환 ====
X_base      = np.load(os.path.join(BASE_NPY, f'X_selected_{SEQ_NAME}.npy'), allow_pickle=True)
y_base_raw  = np.load(os.path.join(BASE_NPY, f'y_selected_pair_{WINDOW}.npy'), allow_pickle=True)

# 레이블 클래스 & 매핑
label_classes = np.load(os.path.join(ROOT, 'models', SEQ_NAME, 'label_classes.npy'),
                        allow_pickle=True)
label2idx     = {lbl: i for i, lbl in enumerate(label_classes)}

# y_base_idx: pair[1] 을 무조건 레이블로 간주
y_base_idx = np.array([
    # pair: array([ID, label]) 혹은 tuple(ID, label)
    label2idx[pair[1]]
    for pair in y_base_raw
], dtype=int)

# ==== 2) 증강 데이터(norm_seq) 로드 & 필터링 ====
_, win, feat_dim = X_base.shape
X_aug_list, y_aug_list = [], []

for lbl in os.listdir(AUG_DIR):
    label_dir = os.path.join(AUG_DIR, lbl)
    if not os.path.isdir(label_dir):
        continue

    for fn in os.listdir(label_dir):
        if not fn.startswith('norm_seq') or not fn.endswith('.npy'):
            continue

        seq = np.load(os.path.join(label_dir, fn), allow_pickle=True)
        # (1,win,feat_dim) → (win,feat_dim)
        if seq.ndim == 3 and seq.shape[0] == 1:
            seq = seq.squeeze(0)

        if seq.shape == (win, feat_dim):
            X_aug_list.append(seq)
            y_aug_list.append(lbl)
        else:
            print(f"⚠️ 무시: {fn:30s} shape {seq.shape} ≠ {(win, feat_dim)}")

if not X_aug_list:
    raise RuntimeError("❌ 유효한 증강 시퀀스가 없습니다!")

y_aug_idx = np.array([label2idx[lbl] for lbl in y_aug_list], dtype=int)

# ==== 3) 합치기 & 저장 ====
X_all = np.concatenate([X_base, np.stack(X_aug_list, axis=0)], axis=0)
y_all = np.concatenate([y_base_idx, y_aug_idx], axis=0)

out_X = os.path.join(BASE_NPY, f'X_selected_{SEQ_NAME}_with_aug.npy')
out_y = os.path.join(BASE_NPY, f'y_selected_pair_{WINDOW}_with_aug.npy')

np.save(out_X, X_all)
np.save(out_y, y_all)

print("✅ 증강 반영된 npy 저장 완료")
print(f"  {out_X} → shape {X_all.shape}")
print(f"  {out_y} → shape {y_all.shape}")
