#!/usr/bin/env python3
"""
auto_infer.py

- infer_from_video: 원본과 50% 속도 두 모드를 지원하는 speed 파라미터 추가
- ensemble_infer: 두 결과를 비교해 적합한 예측 선택
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# 전역 캐시 및 기본값
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
DEFAULT_SEQ = 'L20'
_model_cache = {}


def _load_resources(seq_name):
    if seq_name in _model_cache:
        return _model_cache[seq_name]
    model_dir = os.path.join(ROOT_DIR, 'models', seq_name)
    model = load_model(os.path.join(model_dir, 'sign_language_model_normalized.h5'))
    labels = np.load(os.path.join(model_dir, 'label_classes.npy'), allow_pickle=True)
    X_mean = np.load(os.path.join(model_dir, 'X_mean.npy'))
    X_std  = np.load(os.path.join(model_dir, 'X_std.npy'))
    id2label = {i: lbl for i, lbl in enumerate(labels)}
    window_size = int(seq_name[1:])
    _model_cache[seq_name] = (model, id2label, X_mean, X_std, window_size)
    return _model_cache[seq_name]


def infer_from_video(video_path, seq=DEFAULT_SEQ, conf=0.3, temp=2.5, speed=1.0):
    """
    비디오를 주어진 speed 비율로 재생해 프레임 예측 수행
    speed=1.0: 원본, speed=0.5: 절반 속도
    """
    model, id2label, X_mean, X_std, window_size = _load_resources(seq)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"비디오 열기 실패: {video_path}")

    # 프레임 타이밍 계산
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    proc_fps = fps * speed
    interval_ms = 1000.0 / proc_fps
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_ms = (total_frames / fps) * 1000.0

    hands = mp.solutions.hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    sequence = []
    t = 0.0
    while t < duration_ms:
        cap.set(cv2.CAP_PROP_POS_MSEC, t)
        ret, frame = cap.read()
        if not ret:
            break
        t += interval_ms

        img = cv2.flip(frame, 1)
        H, W = img.shape[:2]
        res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        left = right = []
        if res.multi_hand_landmarks:
            for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                if hd.classification[0].label == 'Left':
                    left = lm.landmark
                else:
                    right = lm.landmark

        def extract_rel(lms):
            if not lms: return [0]*42
            pts = [(p.x*W, p.y*H) for p in lms]
            bx,by = pts[0]; rel=[]
            for x,y in pts: rel += [x-bx, y-by]
            return rel

        def calc_ang(lms):
            if not lms: return [0]*15
            ang=[]
            for i in range(len(lms)-2):
                a=np.array([lms[i].x,lms[i].y]); b=np.array([lms[i+1].x,lms[i+1].y]); c=np.array([lms[i+2].x,lms[i+2].y])
                ba,bc = a-b, c-b
                cos = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
                ang.append(np.degrees(np.arccos(np.clip(cos, -1,1))))
            return ang[:15]+[0]*(15-len(ang))

        feats = extract_rel(left) + extract_rel(right) + calc_ang(left) + calc_ang(right)
        if any(abs(v)>1e-6 for v in feats):
            sequence.append(feats)
    cap.release()
    hands.close()

    if len(sequence) < window_size:
        raise ValueError(f"프레임 부족: {len(sequence)} < {window_size}")

    arr = np.array(sequence, dtype=np.float32)
    n_w = arr.shape[0] - window_size + 1
    windows = np.stack([arr[i:i+window_size] for i in range(n_w)], axis=0)
    normed = (windows - X_mean) / X_std

    probs = model.predict(normed, verbose=0)
    logits = np.log(np.clip(probs,1e-12,1.0)); scaled = np.exp(logits/temp)
    p_T = scaled / np.sum(scaled, axis=1, keepdims=True)

    scores = p_T.max(axis=1)
    idx = scores.argmax(); best = p_T[idx]
    pred = best.argmax(); confidence = float(best[pred])
    top3_inds = best.argsort()[-3:][::-1]
    top3 = [(id2label[i], float(best[i])) for i in top3_inds]

    return id2label[pred], confidence, top3


def ensemble_infer(video_path, seq=DEFAULT_SEQ, conf=0.3, temp=2.5, delta=0.1):
    # 원본 속도
    l1, c1, _ = infer_from_video(video_path, seq, conf, temp, speed=1.0)
    # 절반 속도
    l2, c2, _ = infer_from_video(video_path, seq, conf, temp, speed=0.5)
    if l1 == l2:
        return l1, (c1+c2)/2, [(l1,(c1+c2)/2)]
    if abs(c1-c2) >= delta:
        return (l1, c1, [(l1,c1),(l2,c2)]) if c1>c2 else (l2, c2, [(l1,c1),(l2,c2)])
    return l1, (c1+c2)/2, [(l1,c1),(l2,c2)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sign2Text 자동 모드')
    parser.add_argument('video_name', help='videos 폴더 내 파일명')
    parser.add_argument('--seq', default=DEFAULT_SEQ)
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--temp', type=float, default=2.5)
    args = parser.parse_args()
    video_file = os.path.join(os.path.dirname(__file__), 'videos', args.video_name)
    label, conf, candidates = ensemble_infer(video_file, args.seq, args.conf, args.temp)
    print(f"Top-1 Ensemble: {label} (conf={conf:.3f})")
    print("Candidates:")
    for lbl,sc in candidates:
        print(f"  {lbl}: {sc:.3f}")
