import os
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
import platform

# ==== 시퀀스 설정 ====
SEQ_NAME    = "L20"                  # 사용할 시퀀스 이름 (예: L10, L20, L30…)
WINDOW_SIZE = int(SEQ_NAME[1:])      # 시퀀스 길이 (숫자 부분)
CONF_THRESH = 0.3                    # 예측 허용 신뢰도 문턱값
T           = 2.5                    # 온도 스케일링 파라미터

# ==== 경로 설정 ====
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MODEL_DIR     = os.path.join(BASE_DIR, 'models', SEQ_NAME)
model         = load_model(os.path.join(MODEL_DIR, 'sign_language_model_normalized.h5'))
label_classes = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
X_mean        = np.load(os.path.join(MODEL_DIR, 'X_mean.npy'))
X_std         = np.load(os.path.join(MODEL_DIR, 'X_std.npy'))
id2label      = {i: lbl for i, lbl in enumerate(label_classes)}

# ==== MediaPipe 설정 ====
mp_hands   = mp.solutions.hands
hands      = mp_hands.Hands(max_num_hands=2,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ==== 폰트 설정 ====
try:
    if platform.system() == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf"
    else:
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    font = ImageFont.truetype(font_path, 32)
except:
    font = ImageFont.load_default()

def draw_text(img, text, pos=(10, 50), color=(255,255,0)):

    """OpenCV 이미지 위에 한글 텍스트 그리기."""
    pil  = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def extract_rel(lms, W, H):
    """랜드마크 좌표를 base point 대비 상대좌표로 변환."""
    if not lms:
        return [0]*42
    pts = [(p.x*W, p.y*H) for p in lms]
    bx, by = pts[0]
    rel = []
    for x, y in pts:
        rel += [x - bx, y - by]
    return rel

def calc_ang(lms):
    """랜드마크들 간의 관절 각도 계산 (최대 15개)."""
    if not lms:
        return [0]*15
    ang = []
    for i in range(len(lms)-2):
        a  = np.array([lms[i].x, lms[i].y])
        b  = np.array([lms[i+1].x, lms[i+1].y])
        c  = np.array([lms[i+2].x, lms[i+2].y])
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
        ang.append(np.degrees(np.arccos(np.clip(cos, -1,1))))
    # 15개 고정 크기로 패딩
    return ang[:15] + [0]*(15 - len(ang))

# ==== 웹캠 초기화 ====
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 웹캠 열기 실패")
        exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

cv2.namedWindow("Sign2Text", cv2.WINDOW_NORMAL)
print(f"[{SEQ_NAME}] s=수집 시작/중지, q=종료")

# ==== 상태 변수 ====
sequence       = deque()
collecting     = False
latest_text    = ""
gesture_active = False
hand_was_detected = False
hands_gone_at  = None
display_mode   = False
display_timer  = 0

# ==== 메인 루프 ====
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img = cv2.flip(frame, 1)
    H, W = img.shape[:2]
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # 손 감지 여부
    hand_detected = bool(results.multi_hand_landmarks)

    # 수집 토글/종지
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('s'):
        collecting = not collecting
        if collecting:
            sequence.clear()
            latest_text = ""
            print("🔘 수집 시작")
        else:
            print("🔘 수집 중지")

    # ① 제스처 시작 감지 (수집 중 + 손 첫 진입)
    if collecting and hand_detected and not hand_was_detected:
        gesture_active = True
        sequence.clear()
        hands_gone_at = None
        print("🔘 제스처 시작")

    hand_was_detected = hand_detected

    # 랜드마크 그리기 & 특징 수집
    left, right = [], []
    if results.multi_hand_landmarks:
        for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
            if hd.classification[0].label == 'Left':
                left = lm.landmark
            else:
                right = lm.landmark
            mp_drawing.draw_landmarks(img, lm, mp_hands.HAND_CONNECTIONS)


    feats = extract_rel(left, W, H) + extract_rel(right, W, H) + calc_ang(left) + calc_ang(right)
    if collecting and any(abs(f)>1e-6 for f in feats):
        sequence.append(feats)

    # ② 제스처 중 손 사라짐 기록
    if gesture_active:
        if not hand_detected:
            if hands_gone_at is None:
                hands_gone_at = time.time()
        else:
            hands_gone_at = None

    # ③ 2s 후 자동 예측 트리거
    if collecting and gesture_active and hands_gone_at and time.time() - hands_gone_at >= 2.0 and not display_mode:
        seq_arr   = np.array(sequence, dtype=np.float32)
        n_windows = len(seq_arr) - WINDOW_SIZE + 1
        windows   = np.stack([seq_arr[i:i+WINDOW_SIZE] for i in range(n_windows)], axis=0)
        normed    = (windows - X_mean) / X_std
        preds     = model.predict(normed, verbose=0)

        # 온도 스케일링
        logits  = np.log(np.clip(preds, 1e-12, 1.0))
        scaled  = np.exp(logits / T)
        preds_T = scaled / np.sum(scaled, axis=1, keepdims=True)

        # 창별 최고점 -> Top-3
        window_scores = preds_T.max(axis=1)
        best_win      = window_scores.argmax()
        best_pred     = preds_T[best_win]
        top3          = best_pred.argsort()[-3:][::-1]

        print("=== Top 3 ===")
        for idx in top3:
            print(f"{id2label[idx]}: {best_pred[idx]:.2f}")
        print("=============")

        # 문턱값 판정
        top1_conf = best_pred[top3[0]]
        if top1_conf > CONF_THRESH:
            latest_text = f"{id2label[top3[0]]} ({top1_conf:.2f})"
            print("✅ 예측:", latest_text)
        else:
            latest_text = ""
            print(f"❗ 신뢰도 부족: {top1_conf:.2f}")

        sequence.clear()
        gesture_active = False
        hands_gone_at  = None
        # 결과 출력 유지 시작
        display_mode  = True
        display_timer = time.time()

    # ⑤ 결과 4초 유지
    if display_mode and time.time() - display_timer >= 4.0:
        display_mode = False
        latest_text  = ""

    # 화면 표시
    status = f"{'수집 중' if collecting else '대기 중'} | len={len(sequence)}/{WINDOW_SIZE}"
    img = draw_text(img, status, pos=(10,50))
    if latest_text:
        img = draw_text(img, f"결과: {latest_text}", pos=(10,100))

    cv2.imshow("Sign2Text", img)

# 정리
cap.release()
cv2.destroyAllWindows()
