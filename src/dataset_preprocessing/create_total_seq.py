import numpy as np
import pandas as pd
from tqdm import tqdm

# 🔧 설정
CSV_PATH = "merged_with_angles.csv"
SEQ_LEN = 50
LABEL_COLUMN = "word_id"
OUTPUT_X_PATH = "X_selected_L50.npy"
OUTPUT_Y_PATH = "y_selected_pair_50.npy"

# ✅ 유지할 라벨 리스트 (50개)
selected_labels = [
    "밥솥", "커피", "라면", "병문안", "수면제", "입원", "퇴원", "경찰서", "독서", "지도",
    "콜라", "술", "치료", "보건소", "버스값",
    "출근", "퇴사", "싫어하다", "슬프다", "감기",
    "포켓", "개학", "여아", "학업", "여학교", "백수", "채팅", "신학", "뉴질랜드", "남아",
    "독서실", "유학", "식당", "국어학", "다과", "의학", "위스키", "울산", "월세", "구직",
    "학교연혁", "문학", "예습", "사직", "친아들", "벌꿀", "배드민턴", "독일어", "복습"
]

# 📌 1. 데이터 로드
print("[로드] CSV 로드 중...")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# 📌 2. feature 컬럼 추출
feature_cols = [col for col in df.columns if col.startswith(("lx", "ly", "rx", "ry", "angle_"))]

# 📌 3. word_id ↔ label 매핑
mapping_df = df[["word_id", "label"]].drop_duplicates()
id2label = dict(zip(mapping_df["word_id"], mapping_df["label"]))
label2id = {v: k for k, v in id2label.items()}

# 📌 4. 유지할 word_id 리스트 생성
selected_word_ids = [label2id[label] for label in selected_labels if label in label2id]

# 📌 5. 시퀀스 생성 및 필터링
X, y_wordid = [], []

for word_id, group in tqdm(df.groupby("word_id"), desc="시퀀스 생성 중"):
    if word_id not in selected_word_ids:
        continue

    group = group.reset_index(drop=True)

    for i in range(len(group) - SEQ_LEN + 1):
        window = group.iloc[i:i + SEQ_LEN]
        x_seq = window[feature_cols].values
        label = window[LABEL_COLUMN].iloc[-1]

        X.append(x_seq)
        y_wordid.append(label)

# 📌 6. 라벨 매핑 및 튜플 생성
X = np.array(X).astype(np.float32)  # ← 여기 float32로 명시
y_wordid = np.array(y_wordid)
y_label = np.array([id2label.get(wid, "UNKNOWN") for wid in y_wordid], dtype=object)
y_pair = np.array(list(zip(y_wordid, y_label)), dtype=object)

# 📌 7. 저장
np.save(OUTPUT_X_PATH, X)
np.save(OUTPUT_Y_PATH, y_pair)

print(f"✅ 저장 완료: X_selected.npy {X.shape}, y_selected_pair.npy {y_pair.shape}")
