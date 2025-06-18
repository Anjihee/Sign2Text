import os
import pandas as pd

# === 설정 ===
root_dir = r"E:\user\SignLangData\01_real_word_keypoint\004.수어영상\1.Training\라벨링데이터\REAL\WORD\01"
merged_csv_path = os.path.join(root_dir, "merged_labeled_vectors.csv")

# === 병합 ===
merged_df = pd.DataFrame()


for word_folder in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, word_folder)
    if not os.path.isdir(folder_path):
        continue

    csv_path = os.path.join(folder_path, "labeled_vectors.csv")
    if not os.path.exists(csv_path):
        continue

    try:
        df = pd.read_csv(csv_path)
        df["word_id"] = word_folder  # 원본 폴더 이름 추가
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    except Exception as e:
        print(f"CSV 병합 실패: {csv_path} → {e}")

# === 저장 ===
merged_df.to_csv(merged_csv_path, index=False, encoding="utf-8-sig")
print(f"전체 병합 완료 → {merged_csv_path}")
