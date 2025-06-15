import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image
from collections import deque

# 📁 한글 폰트 경로 (macOS)
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = ImageFont.truetype(font_path, 32)

# 📦 모델 및 라벨 인코더 로드
model = load_model("../models/lstm_seq_model.h5")
label_encoder = joblib.load("../models/label_encoder.pkl")

# 🤚 MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 🔍 좌표 추출 함수
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
    return coords_final.astype(np.float32).reshape(84,)

# 📷 실시간 예측
cap = cv2.VideoCapture(0)
print("🟢 실시간 수어 인식 시작 (Q 키로 종료)")

sequence_length = 10
frame_window = deque(maxlen=sequence_length)
prediction_window = deque(maxlen=5)  # smoothing용

display_label = ""
last_display_time = 0
display_interval = 3  # 자막 갱신 간격 (초)
confidence_threshold = 0.75

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

        if coords is not None and not np.all(coords == 0):
            frame_window.append(coords)

        if len(frame_window) == sequence_length:
            input_seq = np.array(frame_window).reshape(1, sequence_length, 84)
            prediction = model.predict(input_seq, verbose=0)[0]
            predicted_idx = np.argmax(prediction)
            prediction_window.append(predicted_idx)

            smoothed_idx = max(set(prediction_window), key=prediction_window.count)
            smoothed_label = label_encoder.inverse_transform([smoothed_idx])[0]
            confidence = prediction[smoothed_idx]

            current_time = time.time()
            if confidence > confidence_threshold and current_time - last_display_time > display_interval:
                display_label = f"{smoothed_label} ({confidence:.2f})"
                last_display_time = current_time

                # 디버깅 출력
                print(f"[{time.strftime('%H:%M:%S')}] 예측: {smoothed_label}, 확률: {confidence:.3f}")
                print("📌 좌표 일부:")
                print(f" - 왼손 x[:3]: {coords[0:3]}")
                print(f" - 왼손 y[:3]: {coords[21:24]}")
                print(f" - 오른손 x[:3]: {coords[42:45]}")
                print(f" - 오른손 y[:3]: {coords[63:66]}")
                print("-" * 40)
    else:
        # 손이 감지되지 않으면 프레임 초기화
        frame_window.clear()
        prediction_window.clear()
        display_label = ""

    # 자막 표시
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 30), display_label, font=font, fill=(0, 255, 0))
    image = np.array(img_pil)

    cv2.imshow("Sign2Text (좌표 디버깅 포함)", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()