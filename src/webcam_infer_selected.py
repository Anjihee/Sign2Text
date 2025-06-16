import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import os
from PIL import ImageFont, ImageDraw, Image
import time

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')
DATASET_DIR = os.path.join(BASE_DIR, '../dataset')

# 모델 및 데이터 로드
model = load_model(os.path.join(MODEL_DIR, 'sign_language_model_improved.h5'))
label_classes = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
y_selected_pair = np.load(os.path.join(DATASET_DIR, 'y_selected_pair.npy'), allow_pickle=True)
X_selected = np.load(os.path.join(DATASET_DIR, 'X_selected.npy'))

# word_id → label 매핑
id2label = {wid: lbl for wid, lbl in y_selected_pair}

# 폰트 설정
FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = ImageFont.truetype(FONT_PATH, 32)

def draw_korean_text(image, text, position=(10, 50), color=(255, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# MediaPipe 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 시퀀스 설정
sequence = deque(maxlen=10)
latest_word = ""
CONF_THRESHOLD = 0.5
last_inference_time = 0
INFERENCE_INTERVAL = 3.0
last_debug_time = 0
DEBUG_INTERVAL = 5.0
zero_count = 0

# ✅ 상대 좌표 (픽셀 단위) 추출 함수
def extract_relative_coordinates(landmarks, image_shape):
    if not landmarks or len(landmarks) < 1:
        return [0] * 42
    h, w = image_shape[:2]
    base_x, base_y = landmarks[0].x * w, landmarks[0].y * h
    coords = []
    for lm in landmarks:
        coords.append((lm.x * w) - base_x)
        coords.append((lm.y * h) - base_y)
    return coords

# 각도 계산 함수
def calculate_angles(landmarks):
    if not landmarks:
        return [0] * 15
    angles = []
    for i in range(len(landmarks) - 2):
        a = np.array([landmarks[i].x, landmarks[i].y])
        b = np.array([landmarks[i+1].x, landmarks[i+1].y])
        c = np.array([landmarks[i+2].x, landmarks[i+2].y])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angles.append(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))
    return angles[:15] + [0] * (15 - len(angles)) if len(angles) < 15 else angles[:15]

# 웹캠 시작
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    left_hand, right_hand = [], []
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            if label == 'Left':
                left_hand = hand_landmarks.landmark
            elif label == 'Right':
                right_hand = hand_landmarks.landmark
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # ✅ feature 추출
    features = (
        extract_relative_coordinates(left_hand, image.shape) +
        extract_relative_coordinates(right_hand, image.shape) +
        calculate_angles(left_hand) +
        calculate_angles(right_hand)
    )

    if len(features) == 114:
        if sum(features) == 0:
            zero_count += 1
            if zero_count >= 3:
                sequence.clear()
                zero_count = 0
        else:
            zero_count = 0
            sequence.append(features)

            now = time.time()
            if now - last_debug_time > DEBUG_INTERVAL:
                last_debug_time = now
                real_time_vector = np.round(features[:10], 3)
                train_sample_vector = np.round(X_selected[0, 0, :10], 3)
                diff_vector = np.round(real_time_vector - train_sample_vector, 3)
                print("🎥 실시간 feature [앞 10개]:", real_time_vector.tolist())
                print("📚 학습 feature  [앞 10개]:", train_sample_vector.tolist())
                print("📏 차이               :", diff_vector.tolist())

    if len(sequence) == 10:
        valid_frames = [f for f in sequence if sum(f) != 0]
        if len(valid_frames) >= 8:
            now = time.time()
            if now - last_inference_time > INFERENCE_INTERVAL:
                last_inference_time = now

                input_seq = np.array(sequence).reshape(1, 10, 114)
                y_pred = model.predict(input_seq, verbose=0)
                confs = y_pred[0]
                top3_idx = confs.argsort()[-3:][::-1]

                print("🔍 TOP 3 예측 결과:")
                for idx in top3_idx:
                    wid = label_classes[idx]
                    label = id2label.get(wid, wid)
                    print(f"  {label}: {confs[idx]:.3f}")

                pred_word_id = label_classes[np.argmax(y_pred)]
                conf = np.max(y_pred)
                label = id2label.get(pred_word_id, pred_word_id)

                if conf > CONF_THRESHOLD:
                    latest_word = f"{label} ({conf:.2f})"

    image = draw_korean_text(image, latest_word, position=(10, 50), color=(255, 255, 0))
    cv2.imshow('Sign2Text 실시간 수어 인식', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()