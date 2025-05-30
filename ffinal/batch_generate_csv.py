import os
import json
import csv
import re

keypoints_root = r"E:\user\SignLangData\01_real_word_keypoint\004.수어영상\1.Training\라벨링데이터\REAL\WORD\01"
morpheme_root = r"E:\user\SignLangData\01_real_word_morpheme\004.수어영상\1.Training\라벨링데이터\REAL\WORD\morpheme\01"

FPS = 30

def get_time(fname):  # 프레임 시간 추출
    match = re.search(r"_(\d+)_keypoints\.json", fname)
    return int(match.group(1)) / FPS if match else None

def extract_vector(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "people" not in data or not isinstance(data["people"], dict):
        return None
    person = data["people"]
    def norm(hand):  # x, y만 정규화
        coords = [hand[i] for i in range(len(hand)) if i % 3 != 2]
        if len(coords) != 42: return None
        x0, y0 = coords[0], coords[1]
        return [(x - x0) if i % 2 == 0 else (x - y0) for i, x in enumerate(coords)]
    left = norm(person.get("hand_left_keypoints_2d", []))
    right = norm(person.get("hand_right_keypoints_2d", []))
    if not left and not right: return None
    return (left or [0]*42) + (right or [0]*42)

def get_label(time, label_data, margin=0.1):
    for d in label_data:
        if (d["start"] - margin) <= time <= (d["end"] + margin):
            return d["attributes"][0]["name"]
    return None

for idx in range(1, 3001):
    word = f"NIA_SL_WORD{idx:04d}_REAL01"
    for suffix in ["D", "F", "L", "R", "U"]:
        base = f"{word}_{suffix}"
        kpt_dir = os.path.join(keypoints_root, base)
        mor_path = os.path.join(morpheme_root, base + "_morpheme.json")
        csv_path = os.path.join(kpt_dir, "labeled_vectors.csv")

        if not os.path.isdir(kpt_dir) or not os.path.exists(mor_path):
            continue
        if os.path.exists(csv_path):  # 이미 처리된 건 생략
            continue

        with open(mor_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)["data"]
        output = []
        for fname in os.listdir(kpt_dir):
            if not fname.endswith(".json"): continue
            time = get_time(fname)
            if time is None: continue
            label = get_label(time, label_data)
            if label is None: continue
            vec = extract_vector(os.path.join(kpt_dir, fname))
            if vec: output.append(vec + [label])

        if output:
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                header = [f"lx{i}" for i in range(21)] + [f"ly{i}" for i in range(21)] + \
                         [f"rx{i}" for i in range(21)] + [f"ry{i}" for i in range(21)] + ["label"]
                writer.writerow(header)
                writer.writerows(output)
            print(f"✅ CSV 생성 완료: {csv_path}")
