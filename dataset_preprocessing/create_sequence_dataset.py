import pandas as pd
import numpy as np
from tqdm import tqdm

# 설정
CSV_PATH = "merged_with_angles.csv"
SEQ_LEN = 10
LABEL_COLUMN = "word_id"  # 또는 "label"

# CSV 불러오기 (한글 깨짐 방지)
print("[로드] CSV 로딩 중...")
df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")

# 사용할 feature 컬럼 추출
feature_cols = [col for col in df.columns if col.startswith(("lx", "ly", "rx", "ry", "angle_"))]
print(f"[정보] 총 피처 수: {len(feature_cols)}개")

X, y = [], []

# word_id 단위로 시퀀스 구성
for word_id, group in tqdm(df.groupby("word_id"), desc="시퀀스 생성 중"):
    group = group.reset_index(drop=True)

    for i in range(len(group) - SEQ_LEN + 1):
        window = group.iloc[i:i + SEQ_LEN]
        x_seq = window[feature_cols].values  # (10, 84)
        label = window[LABEL_COLUMN].iloc[-1]  # 마지막 프레임 기준

        X.append(x_seq)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"[완료] X.shape = {X.shape}, y.shape = {y.shape}")

# 저장
np.save("X.npy", X)
np.save("y.npy", y)
print("[저장] X.npy, y.npy 저장 완료")
