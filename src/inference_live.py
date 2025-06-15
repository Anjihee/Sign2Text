import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from collections import deque

# 📁 한글 폰트 경로 (macOS 기준)
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = ImageFont.truetype(font_path, 32)

# 📦 모델 및 라벨 인코더 로드
model = load_model("../models/cnn1d_model.h5")
label_encoder = joblib.load("../models/label_encoder.pkl")

# 🤚 MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 🔍 좌표 추출 함수 (정규화 적용)
def extract_dual_hand_landmarks(results, image_width, image_height):
    coords_l = np.zeros((21, 2))
    coords_r = np.zeros((21, 2))

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label
        coords = []
        for lm in hand_landmarks.landmark:
            x_px = lm.x * image_width
            y_px = lm.y * image_height
            coords.append([x_px, y_px])
        coords = np.array(coords)

        if label == 'Left':
            coords_l = coords
        elif label == 'Right':
            coords_r = coords

    lx0, ly0 = coords_l[0]
    rx0, ry0 = coords_r[0]

    lx = coords_l[:, 0] - lx0
    ly = coords_l[:, 1] - ly0
    rx = coords_r[:, 0] - rx0
    ry = coords_r[:, 1] - ry0

    coords_final = np.concatenate([lx, ly, rx, ry])
    return coords_final.astype(np.float32).reshape(1, 84, 1)

# 📷 실시간 예측 시작
cap = cv2.VideoCapture(0)
print("🟢 실시간 수어 인식 시작 (Q 키로 종료)")

# ✅ 스무딩용 softmax 버퍼
prediction_buffer = deque(maxlen=5)
last_label = ""
cooldown = 0
CONFIDENCE_THRESHOLD = 0.6
last_debug_time = 0
debug_interval = 3.0  # 3초마다 디버깅 출력

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        height, width = image.shape[:2]
        coords = extract_dual_hand_landmarks(results, width, height)

        if coords is not None and coords.shape == (1, 84, 1):
            if np.all(coords == 0):
                continue

            prediction = model.predict(coords, verbose=0)[0]
            prediction_buffer.append(prediction)

            current_time = time.time()
            if current_time - last_debug_time > debug_interval:
                print("📌 좌표 일부 (왼손 x[:3]):", coords.flatten()[:3])
                print("📌 좌표 일부 (왼손 y[:3]):", coords.flatten()[21:24])
                print("📌 좌표 일부 (오른손 x[:3]):", coords.flatten()[42:45])
                print("📌 좌표 일부 (오른손 y[:3]):", coords.flatten()[63:66])
                print("----------------------------------------")
                avg_probs = np.mean(prediction_buffer, axis=0)
                top3 = avg_probs.argsort()[-3:][::-1]
                print(f"[{time.strftime('%H:%M:%S')}] 🎯 상위 예측 결과:")
                for i in top3:
                    label = label_encoder.inverse_transform([i])[0]
                    prob = avg_probs[i]
                    print(f" - {label}: {prob:.3f}")
                last_debug_time = current_time

            # 스무딩된 softmax 평균으로 최종 예측 결정
            if len(prediction_buffer) == prediction_buffer.maxlen:
                avg_probs = np.mean(prediction_buffer, axis=0)
                best_idx = np.argmax(avg_probs)
                best_confidence = avg_probs[best_idx]

                if best_confidence > CONFIDENCE_THRESHOLD:
                    best_label = label_encoder.inverse_transform([best_idx])[0]
                    last_label = f"{best_label} ({best_confidence:.2f})"
                    cooldown = 15

    if cooldown > 0:
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), last_label, font=font, fill=(0, 255, 0))
        image = np.array(img_pil)
        cooldown -= 1

    cv2.imshow("Sign2Text (softmax 평균 스무딩)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()