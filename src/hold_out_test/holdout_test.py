# src/hold_out_test/holdout_test.py
#!/usr/bin/env python3

import os
import argparse
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm


def main():
    print("▶ Hold-out 평가 스크립트 시작", flush=True)

    parser = argparse.ArgumentParser(description="Hold-out 테스트 자동 실행 스크립트")
    parser.add_argument('--videos', '-v',
        default=os.path.join(os.path.dirname(__file__), 'videos'),
        help="테스트 영상 폴더 경로"
    )
    parser.add_argument('--labels', '-l',
        default=os.path.join(os.path.dirname(__file__), 'test_labels.csv'),
        help="레이블 CSV 파일 경로"
    )
    parser.add_argument('--seq', '-s', default='L20',
        help="윈도우 시퀀스 (예: L10, L20)"
    )
    parser.add_argument('--conf', '-c', type=float, default=0.3,
        help="신뢰도 문턱값"
    )
    parser.add_argument('--temp', '-t', type=float, default=2.5,
        help="온도 스케일링 파라미터 T"
    )
    args = parser.parse_args()

    videos_dir  = args.videos
    labels_csv  = args.labels
    seq_name    = args.seq
    conf_thresh = args.conf
    temp        = args.temp

    print(f"▶ videos_dir   = {videos_dir}", flush=True)
    print(f"▶ labels_csv   = {labels_csv}", flush=True)
    print(f"▶ sequence name= {seq_name}", flush=True)

    # 1) 레이블 CSV 로드
    df = pd.read_csv(labels_csv)
    print(f"▶ 레이블 CSV 로드 완료: {len(df)}개 항목", flush=True)
    if not {'filename','true_label'}.issubset(df.columns):
        raise ValueError("CSV에 'filename' 및 'true_label' 칼럼이 필요합니다.")

    # 2) 모델이 아는 클래스 목록 로드
    ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    MODEL_DIR   = os.path.join(ROOT_DIR, 'models', seq_name)
    _raw = np.load(os.path.join(MODEL_DIR, 'label_classes.npy'), allow_pickle=True)
    label_classes = [lbl.strip() for lbl in _raw.tolist()]
    print(f"▶ 모델 클래스 {len(label_classes)}개 로드: {label_classes}", flush=True)


    # 3) auto_infer 스크립트 경로
    infer_script = os.path.join(os.path.dirname(__file__), 'auto_infer.py')
    print("▶ infer_script =", infer_script, flush=True)

    preds = []
    trues = []

    # 4) 영상별 반복 처리
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="영상 처리"):
        fn       = row['filename']
        true_lbl = row['true_label']
        video_fp = os.path.join(videos_dir, fn)

        print(f"[{idx+1}/{len(df)}] 대상: {video_fp}", flush=True)
        if not os.path.exists(video_fp):
            print(f"❌ 영상 없음: {video_fp}", flush=True)
            preds.append(None)
            trues.append(true_lbl)
            continue

        cmd = [
            'python', infer_script, fn,
            '--seq', seq_name,
            '--conf', str(conf_thresh),
            '--temp', str(temp)
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=os.path.dirname(__file__),
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"⚠️ 처리 실패 ({fn}): returncode={e.returncode}", flush=True)
            print("stderr:", e.stderr.strip(), flush=True)
            preds.append(None)
            trues.append(true_lbl)
            continue

        # Top-1 예측 파싱
        pred_lbl = None
        for line in proc.stdout.splitlines():
            if line.startswith("Top-1:"):
                parts = line.split()
                if len(parts) >= 2:
                    pred_lbl = parts[1]
                break

        print(f"   → 예측: {pred_lbl}", flush=True)
        preds.append(pred_lbl)
        trues.append(true_lbl)

    # 5) DataFrame 생성
    results = pd.DataFrame({
        'filename': df['filename'],
        'true':    trues,
        'pred':    preds
    })

    res = results

    import unicodedata
    results['true'] = results['true'].str.strip().map(lambda s: unicodedata.normalize('NFC', s))
    results['pred'] = results['pred'].str.strip().map(lambda s: unicodedata.normalize('NFC', s))

    # 디버깅용 출력
    print("\n=== In-memory 결과 샘플 ===", flush=True)
    print(results.head(10).to_string(index=False), flush=True)

    # 6) 모델 클래스 기준 valid 필터링
    mask = results['true'].isin(label_classes) & results['pred'].isin(label_classes)
    valid = results[mask]
    if valid.empty:
        print("❗ 유효 예측 결과(모델 클래스 기준)가 없습니다.", flush=True)
        return

    import unicodedata

    # --- 공백 제거 및 Unicode 정규화 ---
    res['true'] = (
        res['true']
        .astype(str)
        .str.strip()
        .map(lambda s: unicodedata.normalize('NFC', s))
    )
    res['pred'] = (
        res['pred']
        .astype(str)
        .str.strip()
        .map(lambda s: unicodedata.normalize('NFC', s))
    )

    # --- 매칭 상태 직접 확인 ---
    matches = res['true'] == res['pred']
    print(f"✅ 맞은 개수: {matches.sum()} / 전체: {len(res)}")
    print("일치 샘플 예시:")
    print(res[matches].head(5).to_string(index=False))
    print("\n불일치 샘플 예시:")
    print(res[~matches].head(5).to_string(index=False))

    # --- 그 다음에 accuracy_score 계산 ---
    y_true = res['true']
    y_pred = res['pred']
    acc = accuracy_score(y_true, y_pred)
    print(f"\n=== Hold-out Accuracy: {acc*100:.2f}% ===")


    # confusion_matrix & classification_report 에 labels 명시
    cm = confusion_matrix(y_true, y_pred, labels=label_classes)
    print("Confusion Matrix:", flush=True)
    print(cm, flush=True)

    cr = classification_report(
        y_true, y_pred,
        labels=label_classes,
        target_names=label_classes,
        zero_division=0
    )
    print("\nClassification Report:", flush=True)
    print(cr, flush=True)

    # 8) 결과 CSV 저장
    out_csv = os.path.join(os.path.dirname(labels_csv), 'holdout_results.csv')
    results.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n▶ 최종 결과 저장: {out_csv}", flush=True)

if __name__ == '__main__':
    main()



