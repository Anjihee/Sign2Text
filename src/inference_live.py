import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import ImageFont, ImageDraw, Image

#  한글 폰트 경로 (macOS 기본)
font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
font = ImageFont.truetype(font_path, 32)

# 모델 및 인코더 불러오기
model = load_model("../models/cnn1d_model.h5")
label_encoder = joblib.load("../models/label_encoder.pkl")

# MediaPipe 손 인식 초기화 (양손)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 양손 좌표 추출 함수
def extract_dual_hand_landmarks(results):
    coords_l = np.zeros((21, 2))  # 왼손
    coords_r = np.zeros((21, 2))  # 오른손

    if not results.multi_hand_landmarks or not results.multi_handedness:
        return None

    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
        label = handedness.classification[0].label  # 'Left' or 'Right'
        coords = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark])
        if label == 'Left':
            coords_l = coords
        elif label == 'Right':
            coords_r = coords

    # 좌우 손 좌표 연결 → (84,) 벡터
    full_coords = np.concatenate([coords_l.flatten(), coords_r.flatten()])
    full_coords = full_coords.astype(np.float32)
    return full_coords.reshape(1, 84, 1)

# 실시간 예측 시작
cap = cv2.VideoCapture(0)
print("📷 양손 기반 실시간 수어 인식 시작 (종료: Q 키)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        # 손 랜드마크 시각화
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        coords = extract_dual_hand_landmarks(results)
        if coords is not None:
            prediction = model.predict(coords, verbose=0)
            predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = np.max(prediction)

            #  한글 자막 출력 (PIL 사용)
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 30), f"{predicted_label} ({confidence:.2f})", font=font, fill=(0, 255, 0))
            image = np.array(img_pil)

    cv2.imshow("Sign2Text 양손 Inference", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()