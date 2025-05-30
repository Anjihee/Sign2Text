# 📂 프로젝트 파일 구성

```
📁 dataset_preprocessing/
├── batch_generate_csv.py          # 모든 수어 keypoints + morpheme를 순회하며 CSV 파일 개별 생성            
├── merged_csv.py                  # 개별 CSV 파일들을 하나로 병합
├── zip_csv.py                     # **병합된 CSV를 zip 형식으로 압축** 
├── merged_labeled_vectors.zip     # 최종 압축된 데이터셋
├── labeles_list.txt               # 병합된 CSV에 포함된 고유 라벨 목록
│   
└── README.md
```

