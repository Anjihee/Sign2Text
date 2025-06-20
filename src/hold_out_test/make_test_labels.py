#!/usr/bin/env python3
# src/hold_out_test/make_test_labels.py
# 테스트용 영상데이터 라벨 추출

import os
import csv

# 이 스크립트 경로 기준으로 프로젝트 루트 찾기
THIS_DIR    = os.path.dirname(__file__)               # .../src/hold_out_test
VIDEOS_DIR  = os.path.join(THIS_DIR, 'videos')

# 출력할 CSV 파일 경로 (hold_out_test/test_labels.csv)
OUT_CSV    = os.path.join(THIS_DIR, 'test_labels.csv')

rows = []
for fn in os.listdir(VIDEOS_DIR):
    name, ext = os.path.splitext(fn)
    if ext.lower() not in (".mp4", ".mkv"):
        continue
    rows.append({"filename": fn, "true_label": name})

# CSV 쓰기
with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=["filename", "true_label"])
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ '{OUT_CSV}' 생성 완료 – 총 {len(rows)}개 레코드")

