# Hold‑out Test Suite for Sign2Text ✋➡️📝

This directory contains a **self‑contained evaluation harness** that runs a batch of sign‑language videos through the trained Sign2Text model and reports accuracy, confidence and detailed error analytics.

> **Folder:** `src/hold_out_test/`

---

## 📂 Directory layout

```text
src/
└─ hold_out_test/
   ├─ videos/                 # ▶️  test clips (.mp4 / .mkv)
   ├─ auto_infer.py           # single‑video inference utility
   ├─ holdout_test.py         # batch runner + CSV export + console summary
   └─ (generated) holdout_results.csv
models/
└─ L20/                       # pre‑trained model + stats for 20‑frame window
   ├─ sign_language_model_normalized.h5
   ├─ label_classes.npy
   ├─ X_mean.npy
   └─ X_std.npy
```

## ⚙️ Requirements

* Python ≥ 3.8
* TensorFlow ≥ 2.9
* OpenCV‑Python
* MediaPipe (hands)
* NumPy, Pandas, Matplotlib

```bash
pip install -r requirements.txt
```

`requirements.txt` example:

```text
tensorflow==2.15.0
mediapipe==0.10.14
opencv-python
numpy
pandas
matplotlib
```

## 🚀 Quick start

```bash
# 1️⃣ Activate your venv / conda env first
# 2️⃣ Navigate to the project root
cd src/hold_out_test

# 3️⃣ Run hold‑out test (default L20 model, threshold 0.3)
python holdout_test.py \
    --videos_dir videos \
    --seq L20 \
    --conf 0.3 \
    --temp 2.5
```

Output:

```
[OK] 꿀.mkv  -> predicted: 꿀 (conf=0.783)
[WRONG] 경찰서.mkv -> predicted: 월세 (conf=0.623)
⋮
=== Holdout Test Summary ===
Total videos: 25
Accuracy: 52.0%
Results saved to: holdout_results.csv
```

A CSV file `holdout_results.csv` appears in the same folder. It lists:

|  Column      |  Meaning                                  |
| ------------ | ----------------------------------------- |
| `video_name` | file analysed                             |
| `expected`   | label from file name                      |
| `predicted`  | model top‑1 prediction                    |
| `confidence` | softmax prob (after temperature scaling)  |
| `top3`       | three highest labels with probs           |
| `status`     | `OK`, `LOW_CONF_CORRECT`, `WRONG`, `FAIL` |

## 📊 Visualising the results

A convenience snippet is provided in the repo (`viz_holdout.py`) that produces:

* **Status pie chart** (`OK` vs `WRONG`)
* **Confidence histogram**
* **Confusion matrix** for wrong predictions

```bash
python viz_holdout.py --csv holdout_results.csv
```

## 🔍 Script details

### `auto_infer.py`

* Performs per‑frame hand‑landmark extraction with MediaPipe‑Hands.
* Creates sliding windows (default **20 frames**) → normalises with `X_mean`, `X_std`.
* Runs batch inference (`model.predict`) → temperature scaling (`--temp`).
* Returns **top‑1** label / confidence + **top‑3** list.

### `holdout_test.py`

* Scans a folder for `*.mp4` / `*.mkv` test clips.
* Derives expected label from file name (Unicode NFC‑normalised).
* Calls `infer_from_video` (via `auto_infer`) for each clip.
* Writes per‑clip row to CSV + prints live progress.
* Prints aggregate accuracy & confusion counts.

### Command‑line switches

| Flag           | Default               | Description                                          |
| -------------- | --------------------- | ---------------------------------------------------- |
| `--videos_dir` | `videos`              | folder containing test videos                        |
| `--output`     | `holdout_results.csv` | CSV save path                                        |
| `--seq`        | `L20`                 | choose `L10`, `L20`, … matching trained model folder |
| `--conf`       | `0.3`                 | confidence threshold for OK / LOW\_CONF\_CORRECT     |
| `--temp`       | `2.5`                 | temperature scaling `T`                              |

## 🛠️ Customising

* **New window size** – train a model in `models/Lx/` and call `--seq Lx`.
* **Different threshold** – tweak `--conf 0.xx`.
* **Alternative temperature** – adjust `--temp`.
* **Model ensemble / speed variants** – see `auto_infer.py`’s `speed` parameter and `ensemble_infer()` (experimental).

## ❓ Troubleshooting

| Issue                                 | Fix                                                                          |
| ------------------------------------- | ---------------------------------------------------------------------------- |
| `FileNotFoundError: … video`          | Verify clip is in `videos/` with correct extension                           |
| `ValueError: 프레임 부족`                  | Clip shorter than window size – use shorter window or longer clip            |
| Unicode‑looking same but status WRONG | NFC normalisation already applied; ensure file name truly matches label list |

## 📜 Licence

Apache‑2.0 (see `LICENSE`).
