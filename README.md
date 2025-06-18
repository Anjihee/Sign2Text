# ✋📈 보강 데이터 생성 & 실시간 인식 개선 

---

## 담당 파일 및 역할

| 파일                                      | 나의 주요 수정·구현 포인트                                                                                                                                                         |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`src/webcam/webcam_test.py`**         | *w* 키로 웹캠 프레임 → `raw_seq_*.npy`&`norm_seq_*.npy` 자동 저장.<br>  · 저장 위치 `dataset/augmented_samples/<label>/`<br>  · 1 세션 ≒ 20 프레임, **라벨당 30 샘플** 수집 권장                     |
| **`src/train_by_seq_aug.py`**           | 보강 시퀀스를 **원본 데이터와 통합 학습**.<br>  · `augmented_samples` 스캔 → shape 검증 → `np.stack`<br>  · 정규화 파라미터 저장(`X_mean.npy`, `X_std.npy`)<br>  · `EarlyStopping(patience=12)` 로 조정 |
| **`src/webcam/realtime_infer_test.py`** | **자동 예측** 로직 구현.<br>  · 손이 사라진 2 초 뒤 predict → 결과 4 초 표시 / 업데이트<br>  · 손 사라진 1 초 뒤에만 `sequence.clear()` → 깜빡임 무시                                                        |

---

## 데이터 보강 전체 흐름

```mermaid
graph TD
A[한국수어사전 원본 영상] -->|모션 따라 시연| B(webcam_test.py)
B -->|w 키 저장| C[raw_seq_<label>_n.npy \n (20×114)]
B --> D[norm_seq_<label>_n.npy \n ((raw-μ)/σ)]
C & D --> E[dataset/augmented_samples/<label>/]
E --> F(train_by_seq_aug.py)
```

> **수집 절차**
> 1\) `python webcam_test.py` 실행 → *s* 수집 → 수어 시연
> 2\) *w* 저장 (라벨당 ≈30 회)
> 3\) `train_by_seq_aug.py` 로 통합 학습
> 4\) `realtime_infer_test.py` 로 실시간 검증

---

## `webcam_test.py` 핵심 코드 (발췌)

```python
if key == ord('w'):
    save_dir = base_dir/"dataset"/"augmented_samples"/CURRENT_LABEL
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir/f"raw_seq_{CURRENT_LABEL}_{cnt}.npy", seq_arr)
    np.save(save_dir/f"norm_seq_{CURRENT_LABEL}_{cnt}.npy", (seq_arr-X_mean)/X_std)
```

---

## `train_by_seq_aug.py` 통합 로직

```python
aug_dir = DATASET_DIR/"augmented_samples"
for label in aug_dir.iterdir():
    for fn in label.glob("norm_seq_*.npy"):
        seq = np.load(fn).squeeze(0)     # (1,20,114) → (20,114)
        if seq.shape == expected_shape:
            X_aug_list.append(seq)
            y_aug_list.append(label.name)
X_normalized = np.concatenate([X_normalized, np.stack(X_aug_list)])
```

---

## 실시간 인식 개선 – 시간 기반 트리거

| 이벤트            | 동작                             |
| -------------- | ------------------------------ |
| 손이 사라짐 **2 s** | `model.predict()` 호출           |
| 결과 표시          | 4 초 동안 화면 유지, 이후 자동 클리어        |
| 손 사라짐 **1 s**  | `sequence.clear()` → 다음 제스처 대기 |

```python
hands_gone_at = None
if gesture_active:
    if not hand_detected:
        hands_gone_at = hands_gone_at or time.time()
    else:
        hands_gone_at = None

# 1 초 버퍼 리셋
if gesture_active and hands_gone_at and time.time()-hands_gone_at>=1:
    sequence.clear()
```

---


