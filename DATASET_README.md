# 📊 사용한 데이터셋

**AI HUB 한글 수어 데이터셋**
🔗 [데이터셋 바로가기](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=264)

### 📁 사용 경로 및 수량

- **Keypoint JSON 파일**`SignLangData\\01_real_word_keypoint\\004.수어영상\\1.Training\\라벨링데이터\\REAL\\WORD\\01`
→ 약 **172,922개**
- **Morpheme JSON 파일**`SignLangData\\01_real_word_morpheme\\004.수어영상\\1.Training\\라벨링데이터\\REAL\\WORD\\morpheme\\01`
→ 약 **15,000개**

<br>

---

# 🛠 전처리 과정

## 1. **Keypoint + Morpheme 매칭 및 벡터 추출**

- 각 JSON(`_keypoints.json`) 파일에서 손 좌표 읽기
- `hand_left_keypoints_2d`, `hand_right_keypoints_2d` → 총 42개 좌표씩, 좌우 손 합쳐 **84차원 벡터**
- 좌표 누락된 손은 `[0.0]*42`로 패딩
- 프레임 번호 기반으로 **morpheme의 start \~ end 시간**과 매핑하여 라벨 부여
- **FPS(30)** 기준으로 프레임 → 초 단위 변환

> 예외: 손 좌표가 모두 없는 경우 제외, 라벨 없는 경우 SKIP 처리
> 

<br>

## 2. **정규화 및 전처리 방식**

- **좌표 정규화:** 각 손의 첫 번째 landmark (`id=0`)를 원점으로 변환
→ `(xi, yi) → (xi - x0, yi - y0)` 방식
- 상대 좌표만 남기되, 모델 학습 시 normalize 여부는 선택적으로 처리

<br>

## 3. **Batch 처리 구현**

- `batch_generate_csv.py`를 통해 `WORD0001` \~ `WORD3000` 범위의 모든 단어에 대해
`D/F/L/R/U` 유형의 keypoint 폴더 & morpheme 파일을 **자동 순회 처리**
- 각 단어-유형별로 `labeled_vectors.csv` 파일 생성

<br>

## 4. **CSV 병합 및 라벨 목록 정리**

- `merge_labeled_vectors.py`로 모든 CSV 파일 **하나로 통합**

  <br>
  

# ✋ 수어 시퀀스용 시퀀스 데이터셋 생성 코드

수어 단어 인식 모델 학습을 위해, 좌표 기반 벡터 데이터를 **시퀀스 형태로 구성**하고

라벨 필터링, 매핑, 저장까지 자동화하는 일련의 파이썬 스크립트를 구성했습니다.

## ✅ 주요 스크립트 구성

| 파일명 | 역할 |
| --- | --- |
| `batch_generate_csv.py` | 각 단어 폴더 내 JSON을 읽어 `labeled_vectors.csv` 생성 |
| `merge_csv.py` | 위에서 생성한 모든 CSV를 하나로 병합 (`merged_labeled_vectors.csv`) |
| `zip_csv.py` | 병합된 CSV를 압축하여 백업 (`merged_labeled_vectors.zip`) |
| `total.py` | 시퀀스 생성, 라벨 필터링, `(word_id, label)` 매핑하여 `.npy` 파일 생성 |

---

## ⚙️ 시퀀스 생성 (`total.py`)

- 입력 파일: `merged_with_angles.csv`
- 출력 파일:
    - `X_selected_L50.npy`: 시퀀스 길이 50 기준, shape `(N, 50, 114)`, float32
    - `y_selected_pair_50.npy`: 각 시퀀스에 대응되는 `(word_id, label)` 튜플

```python
X = np.load("X_selected_L50.npy")
y = np.load("y_selected_pair_50.npy", allow_pickle=True)
print(X.shape)  # 예: (203, 50, 114)
print(y[0])     # 예: ('NIA_SL_WORD0123_REAL01_F', '술')

```

> allow_pickle=True는 구조화 배열을 로드할 때 반드시 필요합니다.
> 

---

## 📌 시퀀스 길이(L값)에 따른 구성

`total.py` 내부의 `SEQ_LEN` 값만 수정하면 `L10`, `L20`, ..., `L50` 버전 데이터셋을 생성할 수 있습니다.
이는 다양한 시퀀스 길이 실험에 활용되며,
📉 일부 짧은 영상은 L50 이상에서 시퀀스를 1개만 생성할 수 있어 `stratify=y` 사용 시 주의가 필요합니다.

---

## 🧠 전처리 자동화 요약

```
JSON (좌표) + JSON (라벨)
  ↓ batch_generate_csv.py
CSV (단일) + 병합
  ↓ merge_csv.py
CSV (모두 합친 것)
  ↓ total.py
X_selected_L*.npy + y_selected_pair_L*.npy 생성

```

---

## 📎 참고

- 전체 라벨은 50개 한글 단어 기준으로 필터링됨 (labels.txt 참고)
- 데이터는 Conv1D + LSTM 모델에 바로 입력 가능한 `.npy` 포맷으로 저장됨
- `y_selected_pair.npy`는 모델 결과 해석, 자막 출력 등에 직접 활용 가능
