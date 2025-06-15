import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_angle(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0  # norm이 0이면 각도는 0으로 처리

    v1 = v1 / norm1
    v2 = v2 / norm2
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot))

def extract_keypoints(row, hand_prefix):
    """예: ('l', 'l')이면 lx0, ly0 ~ lx20, ly20"""
    return np.array([[row[f'{hand_prefix[0]}x{i}'], row[f'{hand_prefix[1]}y{i}']] for i in range(21)])

def get_hand_angles(kps):
    angle_pairs = [
        (0, 1, 2), (1, 2, 3), (2, 3, 4),
        (0, 5, 6), (5, 6, 7), (6, 7, 8),
        (0, 9, 10), (9, 10, 11), (10, 11, 12),
        (0, 13, 14), (13, 14, 15), (14, 15, 16),
        (0, 17, 18), (17, 18, 19), (18, 19, 20)
    ]
    return [calculate_angle(kps[b] - kps[a], kps[c] - kps[b]) for a, b, c in angle_pairs]

def add_angles_to_dataframe(df):
    all_angles = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="각도 계산 중"):
        kp_left = extract_keypoints(row, ('l', 'l'))
        kp_right = extract_keypoints(row, ('r', 'r'))

        angle_left = get_hand_angles(kp_left)
        angle_right = get_hand_angles(kp_right)

        all_angles.append(angle_left + angle_right)

    angle_cols = [f'angle_l_{i}' for i in range(15)] + [f'angle_r_{i}' for i in range(15)]
    angle_df = pd.DataFrame(all_angles, columns=angle_cols)

    return pd.concat([df.reset_index(drop=True), angle_df], axis=1)

if __name__ == "__main__":
    input_path = "merged_labeled_vectors.csv"
    output_path = "merged_with_angles.csv"

    print("[시작] merged_labeled_vectors.csv 로드 중...")
    df = pd.read_csv(input_path)

    print("[진행] 양손 각도 계산 및 추가 중...")
    df_with_angles = add_angles_to_dataframe(df)

    print("[저장] merged_with_angles.csv 저장 중...")
    df_with_angles.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"[완료] 파일 저장됨: {output_path}")
