
# 📂 프로젝트 파일 구성

```
📁 dataset_preprocessing/
├── batch_generate_csv.py          # 모든 수어 keypoints + morpheme를 순회하며 CSV 파일 개별 생성            
├── merged_csv.py                  # 개별 CSV 파일들을 하나로 병합
├── zip_csv.py                     # 병합된 CSV를 zip 형식으로 압축
├── merged_labeled_vectors.zip     # 최종 압축된 데이터셋
├── labeles_list.txt               # 병합된 CSV에 포함된 고유 라벨 목록
│   
└── README.md
```
<br>

---
## 📐 데이터 형태

### 1. X_selected.npy
- **Shape**: `(샘플 수, 시퀀스 길이=10, 특성 수=114)`
- **내용**:
  - 🔹 **좌표 84개**
    - 왼손: `lx0 ~ ly20` (21포인트 × 2좌표 = 42개)
    - 오른손: `rx0 ~ ry20` (21포인트 × 2좌표 = 42개)
  - 🔹 **관절 각도 30개**
    - 왼손: `angle_l_0 ~ angle_l_14`
    - 오른손: `angle_r_0 ~ angle_r_14`
  - 👉 총: `84 + 30 = 114`개 feature

<br>

### 2. y_selected_pair.npy
- **Shape**: `(샘플 수,)`
- **내용**: `(word_id, label)` 쌍을 튜플로 저장한 구조화 배열
  - 예: `("NIA_SL_WORD0032_REAL01_U", "감기")`
- ✅ word_id를 그대로 유지하면서 label도 함께 포함하여  
  자막 출력, 예측 분석, 혼동 행렬 시각화 등에 직접 활용 가능

<br>

### 3. labels.txt
- **내용**: 전체 라벨 리스트가 줄바꿈(`\n`)으로 정리된 `.txt` 파일  
  - 이 데이터셋에는 총 **20개 라벨**만 포함되어 있습니다.
  - 예:  
    ```
    감기
    커피
    라면
    ...
    ```
---

## 🧠 용도

- 이 데이터셋은 **시퀀스 기반 수어 단어 분류 모델 학습**을 위해 전처리된 자료입니다.
- Conv1D + LSTM 모델 구조에 최적화되어 있으며,  
  입력 시퀀스와 라벨(단어)을 정확히 매칭할 수 있도록 구성되어 있습니다.


## 🔍 추가 정보

- 시퀀스는 동일한 `word_id` 내에서 **프레임 10개 단위**로 생성됩니다.
- 각도는 MediaPipe의 21개 손 포인트 기준으로, **관절 간 벡터의 arccos**를 통해 계산됩니다.
- `.npy` 파일은 `numpy.load(..., allow_pickle=True)`로 바로 로드할 수 있습니다.

---
<br>

# 📊 사용한 데이터셋

**AI HUB 한글 수어 데이터셋**
🔗 [데이터셋 바로가기](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=264)

### 📁 사용 경로 및 수량

* **Keypoint JSON 파일**
  `SignLangData\01_real_word_keypoint\004.수어영상\1.Training\라벨링데이터\REAL\WORD\01`
  → 약 **172,922개**

* **Morpheme JSON 파일**
  `SignLangData\01_real_word_morpheme\004.수어영상\1.Training\라벨링데이터\REAL\WORD\morpheme\01`
  → 약 **15,000개**

<br>

---

# 🛠 전처리 과정

## 1. **Keypoint + Morpheme 매칭 및 벡터 추출**

* 각 JSON(`*_keypoints.json`) 파일에서 손 좌표 읽기
* `hand_left_keypoints_2d`, `hand_right_keypoints_2d` → 총 42개 좌표씩, 좌우 손 합쳐 **84차원 벡터**
* 좌표 누락된 손은 `[0.0]*42`로 패딩
* 프레임 번호 기반으로 **morpheme의 start \~ end 시간**과 매핑하여 라벨 부여
* **FPS(30)** 기준으로 프레임 → 초 단위 변환

> 예외: 손 좌표가 모두 없는 경우 제외, 라벨 없는 경우 `SKIP` 처리

<br>

## 2. **정규화 및 전처리 방식**

* **좌표 정규화:** 각 손의 첫 번째 landmark (`id=0`)를 원점으로 변환
  → `(xi, yi) → (xi - x0, yi - y0)` 방식
* 상대 좌표만 남기되, 모델 학습 시 normalize 여부는 선택적으로 처리

<br>

## 3. **Batch 처리 구현**

* `batch_generate_csv.py`를 통해 `WORD0001` \~ `WORD3000` 범위의 모든 단어에 대해
  `D/F/L/R/U` 유형의 keypoint 폴더 & morpheme 파일을 **자동 순회 처리**
* 각 단어-유형별로 `labeled_vectors.csv` 파일 생성

<br>

## 4. **CSV 병합 및 라벨 목록 정리**

* `merge_labeled_vectors.py`로 모든 CSV 파일 **하나로 통합**

  * 최종 파일: `merged_labeled_vectors.csv`
* `labels_list.txt`: 전체 라벨을 중복 없이 정리하여 저장

<br>

## 5. **CSV 압축 및 공유**

* `zip_csv.py`를 통해 **100MB 이하의 zip**으로 압축 (`merged_labeled_vectors.zip`)
* GitHub 업로드 시 용량 제한에 대응하기 위해 `.csv`는 무시하고 `.zip`만 저장소에 반영

<br>

## 전달용 파일 구성

* `merged_labeled_vectors.zip`: 모델 학습용 데이터
* `labels_list.txt`: 전체 라벨 목록
* `README.md`: 구성 및 사용법 정리




