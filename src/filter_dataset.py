import pandas as pd

# 🔹 경로 설정: 기존 전체 데이터셋
INPUT_PATH = "../dataset/merged_labeled_vectors.csv"
OUTPUT_PATH = "../dataset/filtered_sentence_data.csv"

# 🔹 유지할 단어 리스트 (문장 구성에 적합한 20개)
selected_labels = [
    "밥솥", "커피", "라면", "병문안", "수면제", "입원", "퇴원", "경찰서", "독서", "지도",
    "콜라", "술", "치료", "보건소", "버스값",  # 명사
    "출근", "퇴사", "싫어하다", "슬프다", "감기"  # 동사 또는 동사적 표현
]

# 🔹 CSV 불러오기
df = pd.read_csv(INPUT_PATH)

# 🔹 필터링 수행
filtered_df = df[df["label"].isin(selected_labels)]

# 🔹 저장
filtered_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"✅ 저장 완료: {OUTPUT_PATH} ({filtered_df.shape[0]}행)")