import os
import cv2
import time
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFrame
from PIL import Image, ImageFont, ImageDraw
import platform

# ==== 시퀀스 설정 ====
SEQ_NAME = "L20"
WINDOW_SIZE = int(SEQ_NAME[1:])
CONF_THRESH = 0.3
T = 2.5

# ==== 경로 설정 ====
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
MODEL_DIR = os.path.join(BASE_DIR, 'models', SEQ_NAME)
model = load_model(os.path.join(MODEL_DIR, 'sign_language_model_normalized.h5'))
label_classes = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
X_mean = np.load(os.path.join(MODEL_DIR, 'X_mean.npy'))
X_std = np.load(os.path.join(MODEL_DIR, 'X_std.npy'))
id2label = {i: lbl for i, lbl in enumerate(label_classes)}

# ==== MediaPipe 설정 ====
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# ==== 폰트 설정 ====
try:
    if platform.system() == "Windows":
        font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 32)
    else:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/AppleGothic.ttf", 32)
except:
    font = ImageFont.load_default()

# OpenCV 이미지 위에 한글 텍스트 그리기
def draw_text(img, text, pos=(10,50), color=(255,255,0)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

def extract_rel(lms, W, H):
    if not lms:
        return [0]*42
    pts = [(p.x*W, p.y*H) for p in lms]
    bx, by = pts[0]
    rel = []
    for x, y in pts:
        rel += [ x-bx, y-by ]
    return rel

def calc_ang(lms):
    if not lms:
        return [0]*15
    ang = []
    for i in range(len(lms)-2):
        a = np.array([lms[i].x, lms[i].y])
        b = np.array([lms[i+1].x, lms[i+1].y])
        c = np.array([lms[i+2].x, lms[i+2].y])
        ba = a - b
        bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc)+1e-6)
        ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return ang[:15] + [0]*(15 - len(ang))

class Sign2TextApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign2Text 실시간 수어 인식")
        self.setFixedSize(980, 600)
        self.setStyleSheet("background-color: #ffffff; color: black;")

        self.title_label = QLabel("\U0001F9E0 Sign2Text | 실시간 수어 인식 시스템")
        self.title_label.setStyleSheet("font-size: 22px; font-weight: bold; color: #333333; margin-bottom: 10px")
        self.title_label.setAlignment(Qt.AlignCenter)

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setFrameShape(QFrame.Box)

        self.status_label = QLabel("상태: 대기 중")
        self.status_label.setStyleSheet("font-size: 18px; color: #0077cc;")
        self.status_label.setAlignment(Qt.AlignLeft)

        self.result_label = QLabel("결과:")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #0066aa")
        self.result_label.setAlignment(Qt.AlignLeft)

        self.top3_title = QLabel("Top-3 예측 결과")
        self.top3_title.setStyleSheet("font-size: 16px; color: #222222;")
        self.top3_title.setAlignment(Qt.AlignLeft)

        self.top3_label = QLabel("")
        self.top3_label.setStyleSheet("font-size: 14px; color: #444444")
        self.top3_label.setAlignment(Qt.AlignLeft)

        self.start_btn = QPushButton("수집 시작")
        self.start_btn.setFixedWidth(280)
        self.start_btn.setStyleSheet("padding: 10px; font-size: 16px; background-color: #007acc; color: white; border-radius: 5px")
        self.start_btn.clicked.connect(self.toggle_collect)

        info_layout = QVBoxLayout()
        info_layout.setAlignment(Qt.AlignLeft)
        info_layout.setSpacing(6)
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.result_label)
        info_layout.addWidget(self.top3_title)
        info_layout.addWidget(self.top3_label)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.start_btn)
        button_layout.setSpacing(10)
        button_layout.setAlignment(Qt.AlignLeft)

        right_layout = QVBoxLayout()
        right_layout.addSpacing(10)
        right_layout.addLayout(info_layout)
        right_layout.addSpacing(15)
        right_layout.addSpacing(150)
        right_layout.addLayout(button_layout)

        content_layout = QHBoxLayout()
        content_layout.addWidget(self.image_label)
        content_layout.addLayout(right_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.title_label)
        main_layout.addLayout(content_layout)

        self.setLayout(main_layout)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(66)  # 약 15 FPS로 제한 (OpenCV와 비슷하게)

        self.sequence = deque()
        self.collecting = False
        self.latest_text = ""
        self.hand_was_detected = False
        self.gesture_active = False
        self.lost_count = 0
        self.display_mode = False
        self.display_count = 0

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        hand_detected = bool(results.multi_hand_landmarks)

        if self.collecting and hand_detected and not self.hand_was_detected:
            self.gesture_active = True
            self.sequence.clear()

        self.hand_was_detected = hand_detected

        if self.gesture_active:
            self.lost_count = self.lost_count + 1 if not hand_detected else 0

        if self.collecting and self.display_mode and len(self.sequence) > 0:
            self.display_mode = False

        left, right = [], []
        if results.multi_hand_landmarks:
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                if hd.classification[0].label == 'Left':
                    left = lm.landmark
                else:
                    right = lm.landmark
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        feats = extract_rel(left, W, H) + extract_rel(right, W, H) + calc_ang(left) + calc_ang(right)
        if self.collecting and any(abs(f) > 1e-6 for f in feats):
            self.sequence.append(feats)

        if self.collecting and self.gesture_active and self.lost_count >= 30 and not self.display_mode:
            if len(self.sequence) >= WINDOW_SIZE:
                seq_arr = np.array(self.sequence)
                windows = np.stack([seq_arr[i:i+WINDOW_SIZE] for i in range(len(seq_arr)-WINDOW_SIZE+1)], axis=0)
                normed = (windows - X_mean) / X_std
                preds = model.predict(normed, verbose=0)

                logits = np.log(np.clip(preds, 1e-12, 1.0))
                scaled = np.exp(logits / T)
                preds_T = scaled / np.sum(scaled, axis=1, keepdims=True)

                win_idx = preds_T.max(axis=1).argmax()
                best_pred = preds_T[win_idx]
                class_idx = best_pred.argmax()
                best_conf = best_pred[class_idx]

                top3_idx = best_pred.argsort()[-3:][::-1]
                top3 = [f"{id2label[i]}: {best_pred[i]:.2f}" for i in top3_idx]
                self.top3_label.setText("\n".join(top3))

                if best_conf > CONF_THRESH:
                    self.latest_text = f"{id2label[class_idx]} ({best_conf:.2f})"
                else:
                    self.latest_text = f"신뢰도 부족 ({best_conf:.2f})"

                self.result_label.setText(f"결과: {self.latest_text}")
                self.display_mode = True
                self.display_count = 0

            self.gesture_active = False
            self.lost_count = 0

        if self.display_mode:
            self.display_count += 1
            if self.display_count >= 60:
                self.display_mode = False
                self.latest_text = ""
                self.sequence.clear()

        frame = draw_text(frame, f"상태: {'수집 중' if self.collecting else '대기 중'} | {len(self.sequence)}/{WINDOW_SIZE}", color=(0, 102, 204) if self.collecting else (50, 50, 50))
        img = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

    def toggle_collect(self):
        self.collecting = not self.collecting
        self.sequence.clear()
        self.latest_text = ""
        self.result_label.setText("결과:")
        self.top3_label.setText("")
        self.status_label.setText("상태: 수집 중" if self.collecting else "상태: 대기 중")
        self.start_btn.setText("수집 중지" if self.collecting else "수집 시작")

if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = Sign2TextApp()
    win.show()
    sys.exit(app.exec_())