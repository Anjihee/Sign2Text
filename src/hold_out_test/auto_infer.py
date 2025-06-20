# src/hold_out_test/auto_infer.py
#!/usr/bin/env python3

import os
import argparse
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# ─ argparse 세팅 ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Sign2Text 자동 모드 (hold_out_test/videos 내 영상 한 번에 예측)"
)
parser.add_argument(
    'video_name',
    help="hold_out_test/videos/ 폴더 내 파일명 (예: sample.mp4 또는 demo.mkv)"
)
parser.add_argument('--seq',   default='L20', help="윈도우 시퀀스 (예: L10, L20)")
parser.add_argument('--conf',  type=float, default=0.3, help="신뢰도 문턱값")
parser.add_argument('--temp',  type=float, default=2.5, help="온도 스케일링 T")
args = parser.parse_args()

SEQ_NAME    = args.seq
WINDOW_SIZE = int(SEQ_NAME[1:])
CONF_THRESH = args.conf
T           = args.temp

# ─ 모델 및 메타데이터 로드 ───────────────────────────────────────────────────────
ROOT_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MODEL_DIR     = os.path.join(ROOT_DIR, 'models', SEQ_NAME)
model         = load_model(os.path.join(MODEL_DIR, 'sign_language_model_normalized.h5'))
label_classes = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
X_mean        = np.load(os.path.join(MODEL_DIR, 'X_mean.npy'))
X_std         = np.load(os.path.join(MODEL_DIR, 'X_std.npy'))
id2label      = {i: lbl for i, lbl in enumerate(label_classes)}

# ─ 비디오 파일 경로 설정 ──────────────────────────────────────────────────────
THIS_DIR   = os.path.dirname(__file__)
VIDEOS_DIR = os.path.join(THIS_DIR, 'videos')
name, ext = os.path.splitext(args.video_name)
if ext.lower() in ('.mp4', '.mkv'):
    video_file = os.path.join(VIDEOS_DIR, args.video_name)
else:
    alt1 = os.path.join(VIDEOS_DIR, name + '.mp4')
    alt2 = os.path.join(VIDEOS_DIR, name + '.mkv')
    if os.path.exists(alt1):
        video_file = alt1
    elif os.path.exists(alt2):
        video_file = alt2
    else:
        raise FileNotFoundError(f"비디오를 찾을 수 없음: {alt1} 또는 {alt2}")

# ─ 비디오 캡처 ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(video_file)
if not cap.isOpened():
    raise IOError(f"비디오 열기 실패: {video_file}")

# ─ MediaPipe Hands 세팅 ──────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

def extract_rel(lms, W, H):
    if not lms:
        return [0]*42
    pts = [(p.x*W, p.y*H) for p in lms]
    bx, by = pts[0]
    rel = []
    for x, y in pts:
        rel += [x-bx, y-by]
    return rel

def calc_ang(lms):
    if not lms:
        return [0]*15
    ang = []
    for i in range(len(lms)-2):
        a = np.array([lms[i].x,   lms[i].y])
        b = np.array([lms[i+1].x, lms[i+1].y])
        c = np.array([lms[i+2].x, lms[i+2].y])
        ba = a - b; bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
        ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return ang[:15] + [0]*(15 - len(ang))

# ─ 프레임 순회하며 특징 수집 ────────────────────────────────────────────────────
sequence = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.flip(frame, 1)
    H, W = img.shape[:2]
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    left = right = []
    if results.multi_hand_landmarks:
        for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            if hd.classification[0].label == 'Left':
                left = lm.landmark
            else:
                right = lm.landmark
            mp_drawing.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)

    feats = (
        extract_rel(left, W, H) +
        extract_rel(right, W, H) +
        calc_ang(left) +
        calc_ang(right)
    )
    if any(abs(f) > 1e-6 for f in feats):
        sequence.append(feats)

cap.release()

# ─ 예측 실행 ────────────────────────────────────────────────────────────────────
if len(sequence) < WINDOW_SIZE:
    raise ValueError(f"프레임 부족: {len(sequence)} < {WINDOW_SIZE}")

import numpy as _np  # 분리 import
seq_arr   = _np.array(sequence, dtype=_np.float32)
n_windows = seq_arr.shape[0] - WINDOW_SIZE + 1
windows   = _np.stack([seq_arr[i:i+WINDOW_SIZE] for i in range(n_windows)], axis=0)
normed    = (windows - X_mean) / X_std

probs     = model.predict(normed, verbose=0)
logits    = _np.log(_np.clip(probs, 1e-12, 1.0))
scaled    = _np.exp(logits / T)
probs_T   = scaled / _np.sum(scaled, axis=1, keepdims=True)

scores    = probs_T.max(axis=1)
best_idx  = scores.argmax()
best_p    = probs_T[best_idx]
pred      = best_p.argmax()
conf      = best_p[pred]

print("\n=== 최종 예측 결과 ===")
print(f"Top-1: {id2label[pred]} (conf={conf:.3f})")
print("Top-3:")
for idx in best_p.argsort()[-3:][::-1]:
    print(f"  {id2label[idx]}: {best_p[idx]:.3f}")

if conf < CONF_THRESH:
    print("⚠️ 신뢰도 낮음; threshold:", CONF_THRESH)
else:
    print("✅ 예측 완료")

