
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import os
from PIL import ImageFont, ImageDraw, Image
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')
DATASET_DIR = os.path.join(BASE_DIR, '../dataset')

model = load_model(os.path.join(MODEL_DIR, 'sign_language_model.h5'))
label_classes = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
y_selected_pair = np.load(os.path.join(DATASET_DIR, 'y_selected_pair.npy'), allow_pickle=True)

# word_id → label 매핑
id2label = {wid: lbl for wid, lbl in y_selected_pair}

FONT_PATH = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = ImageFont.truetype(FONT_PATH, 32)

def draw_korean_text(image, text, position=(10, 50), color=(255, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

sequence = deque(maxlen=10)
latest_word = ""
CONF_THRESHOLD = 0.6
last_inference_time = 0
INFERENCE_INTERVAL = 3.0

def extract_relative_coordinates(landmarks):
    if not landmarks or len(landmarks) < 1:
        return [0] * 42
    base_x, base_y = landmarks[0].x, landmarks[0].y
    coords = []
    for lm in landmarks:
        coords.extend([lm.x - base_x, lm.y - base_y])
    return coords

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
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angles.append(np.degrees(angle))
    return angles[:15] if len(angles) >= 15 else angles + [0] * (15 - len(angles))

def get_feature_vector(left_hand, right_hand):
    left_coords = extract_relative_coordinates(left_hand)
    right_coords = extract_relative_coordinates(right_hand)
    left_angles = calculate_angles(left_hand)
    right_angles = calculate_angles(right_hand)
    feature = left_coords + right_coords + left_angles + right_angles
    return feature if len(feature) == 114 else [0] * 114

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

    # 손 감지 상태 출력
    if left_hand and not right_hand:
        print("🖐️ 왼손만 감지됨")
    elif right_hand and not left_hand:
        print("🖐️ 오른손만 감지됨")
    elif not left_hand and not right_hand:
        print("❌ 손 미감지")

    # feature 추출 및 디버깅 출력
    feature = get_feature_vector(left_hand, right_hand)
    if sum(feature) != 0:
        sequence.append(feature)
        print("🧩 Feature vector 추가됨 (길이:", len(feature), ")")
    else:
        print("⚠️ feature 값 없음, 패스")

    current_time = time.time()
    if len(sequence) == 10 and (current_time - last_inference_time) > INFERENCE_INTERVAL:
        input_seq = np.expand_dims(sequence, axis=0)
        y_pred = model.predict(input_seq, verbose=0)
        pred_idx = np.argmax(y_pred)
        conf = np.max(y_pred)

        top3_idx = np.argsort(y_pred[0])[-3:][::-1]
        print("🔍 TOP 3 예측:")
        for i in top3_idx:
            label = label_classes[i]
            print(f" - {label}: {y_pred[0][i]:.3f}")

        if conf > CONF_THRESHOLD:
            label = label_classes[pred_idx]
            latest_word = f"{label} ({conf:.2f})"
            print(f"✅ 자막 출력: {latest_word}")
            last_inference_time = current_time

    image = draw_korean_text(image, latest_word)
    cv2.imshow('Sign2Text 실시간 수어 인식 (Debug)', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
