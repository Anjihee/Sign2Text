# sign2text_gui.py (📐 왼쪽 정렬 개선 + 버튼 아래 배치)

import sys
import os
import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from tensorflow.keras.models import load_model
from PIL import Image, ImageFont, ImageDraw
from PyQt5.QtWidgets import (
    QApplication, QLabel, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, QFrame
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
import platform

# ==== 설정 ====
SEQ_NAME = "L20"                 # 사용할 시퀀스 이름 (예: L10, L20, L30…)
WINDOW_SIZE = int(SEQ_NAME[1:])  # 시퀀스 길이 (숫자 부분)
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
hands = mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.7, 
                    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

try:
    if platform.system() == "Windows":
        font_path = "C:/Windows/Fonts/malgun.ttf"
    else:
        font_path = "/System/Library/Fonts/Supplemental/AppleGothic.ttf"
    font = ImageFont.truetype(font_path, 24)
except:
    font = ImageFont.load_default()

# OpenCV 이미지 위에 한글 텍스트 그리기
def draw_text(img, text, pos=(10, 30), color=(0, 0, 0)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# 랜드마크 좌표를 base point 대비 상대 좌표로 변환
def extract_rel(lms, W, H):
    if not lms:
        return [0]*42
    pts = [(p.x*W, p.y*H) for p in lms]
    bx, by = pts[0]
    rel = []
    for x, y in pts:
        rel += [x - bx, y - by]
    return rel

# 랜드마크들 간의 관절 각도 계신 (최대 15개)
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
        cos = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
        ang.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return ang[:15] + [0]*(15 - len(ang))

# GUI 개선
class Sign2TextApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign2Text 실시간 수어 인식")
        self.setFixedSize(980, 600)
        self.setStyleSheet("background-color: #ffffff; color: black;")

        self.title_label = QLabel("🧠 Sign2Text | 실시간 수어 인식 시스템")
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
        self.pred_btn = QPushButton("예측")
        for btn in (self.start_btn, self.pred_btn):
            btn.setFixedWidth(280)
            btn.setStyleSheet(
                "padding: 10px; font-size: 16px; background-color: #007acc; color: white; border-radius: 5px")

        self.start_btn.clicked.connect(self.toggle_collect)
        self.pred_btn.clicked.connect(self.predict_sign)

        info_layout = QVBoxLayout()
        info_layout.setAlignment(Qt.AlignLeft)
        info_layout.setSpacing(6)
        info_layout.addWidget(self.status_label)
        info_layout.addWidget(self.result_label)
        info_layout.addWidget(self.top3_title)
        info_layout.addWidget(self.top3_label)

        button_layout = QVBoxLayout()
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.pred_btn)
        button_layout.setSpacing(10)
        button_layout.setAlignment(Qt.AlignLeft)

        right_layout = QVBoxLayout()
        right_layout.addSpacing(10)  
        right_layout.addLayout(info_layout)
        right_layout.addSpacing(15)   
        right_layout.addSpacing(150)  # ← 버튼을 위로 올릴 수 있음 (예: 180 ~ 200 사이로)
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
        self.timer.start(30)

        # 기존 상태변수를 인스턴스 변수로 관리
        self.sequence = deque()
        self.collecting = False
        self.latest_text = ""

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        left, right = [], []
        if results.multi_hand_landmarks:
            for lm, hd in zip(results.multi_hand_landmarks, results.multi_handedness):
                if hd.classification[0].label == 'Left':
                    left = lm.landmark
                else:
                    right = lm.landmark
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

        # 특징 벡터 생성
        feats = extract_rel(left, W, H) + extract_rel(right, W, H) + calc_ang(left) + calc_ang(right)
        if self.collecting and any(abs(f) > 1e-6 for f in feats):
            self.sequence.append(feats)

        # 상태 표시
        frame = draw_text(
            frame,
            f"상태: {'수집 중' if self.collecting else '대기 중'} | {len(self.sequence)}/{WINDOW_SIZE}",
            color=(0, 102, 204) if self.collecting else (50, 50, 50)
        )
        img = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format_BGR888)
        self.image_label.setPixmap(QPixmap.fromImage(img))

    # 수집 / 중지
    def toggle_collect(self):
        self.collecting = not self.collecting
        self.sequence.clear()
        self.latest_text = ""
        self.status_label.setText("상태: 수집 중" if self.collecting else "상태: 대기 중")
        self.start_btn.setText("수집 중지" if self.collecting else "수집 시작")

    # 예측 버튼
    def predict_sign(self):
        if len(self.sequence) < WINDOW_SIZE:
            self.result_label.setText("❗ 시퀀스 부족")
            self.top3_label.setText("")
            return

        seq_arr = np.array(self.sequence, dtype=np.float32)
        n_windows = len(seq_arr) - WINDOW_SIZE + 1
        windows = np.stack([seq_arr[i:i+WINDOW_SIZE] for i in range(n_windows)], axis=0)
        normed = (windows - X_mean) / X_std
        preds = model.predict(normed, verbose=0)

        # 온도 스케일링
        logits = np.log(np.clip(preds, 1e-12, 1.0))
        scaled = np.exp(logits / T)
        preds_T = scaled / np.sum(scaled, axis=1, keepdims=True)

        # 각 창별 최고 점수
        window_scores = preds_T.max(axis=1)
        best_idx = window_scores.argmax()
        best_pred = preds_T[best_idx]

        # Top-3 출력
        top3_idx = best_pred.argsort()[-3:][::-1]
        top3_text = "\n".join([f"{id2label[idx]}: {best_pred[idx]:.2f}" for idx in top3_idx])

        # 문턱값 판정
        top1_conf = best_pred[top3_idx[0]]
        self.latest_text = f"{id2label[top3_idx[0]]} ({top1_conf:.2f})" if top1_conf > CONF_THRESH else "❗ 신뢰도 부족"

        # 결과 출력
        self.result_label.setText(f"결과: {self.latest_text}")
        self.top3_label.setText(top3_text)
        self.sequence.clear()

    def closeEvent(self, event):
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Sign2TextApp()
    window.show()
    sys.exit(app.exec_())