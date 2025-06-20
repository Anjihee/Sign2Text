#!/usr/bin/env python3
"""
holdout_test.py

holdout test를 실행하고 결과를 CSV와 터미널에 출력합니다.
predicted과 expected 비교 시, Unicode 정규화를 적용해 정확히 일치할 때 OK로 표시합니다.
"""
import os
import glob
import argparse
import csv
import unicodedata

from auto_infer import infer_from_video

def main(videos_dir, output_csv_name, seq, conf_thresh, temp):
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, output_csv_name)

    patterns = ["*.mp4", "*.mkv"]
    video_paths = []
    for p in patterns:
        video_paths.extend(glob.glob(os.path.join(videos_dir, p)))

    if not video_paths:
        print(f"[ERROR] No video files found in {videos_dir}")
        return

    results = []  # (name, expected, predicted, confidence, top3, status)

    with open(output_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(["video_name", "expected", "predicted", "confidence", "top3", "status"])

        for vp in sorted(video_paths):
            name = os.path.basename(vp)
            expected_raw = os.path.splitext(name)[0]
            # Unicode 정규화: NFC
            expected = unicodedata.normalize('NFC', expected_raw.strip())
            try:
                pred_label, conf_score, top3_list = infer_from_video(
                    vp, seq=seq, conf=conf_thresh, temp=temp
                )
                # Unicode 정규화: NFC
                predicted = unicodedata.normalize('NFC', pred_label.strip())

                top3_str = "; ".join(f"{lbl}:{sc:.3f}" for lbl, sc in top3_list)
                if predicted == expected:
                    status = 'OK' if conf_score >= conf_thresh else 'LOW_CONF_CORRECT'
                else:
                    status = 'WRONG'

                writer.writerow([name, expected, pred_label, f"{conf_score:.3f}", top3_str, status])
                print(f"[{status}] {name} -> predicted: {pred_label}, expected: {expected}, conf={conf_score:.3f}")
                results.append((name, expected, pred_label, conf_score, top3_str, status))
            except Exception as e:
                status = 'FAIL'
                writer.writerow([name, expected, '', '', '', status])
                print(f"[FAIL] {name} -> {e}")
                results.append((name, expected, None, None, None, status))

    total = len(results)
    ok_count = sum(1 for r in results if r[5] == 'OK')
    low_conf_correct = sum(1 for r in results if r[5] == 'LOW_CONF_CORRECT')
    wrong_count = sum(1 for r in results if r[5] == 'WRONG')
    fail_count = sum(1 for r in results if r[5] == 'FAIL')
    accuracy = (ok_count / total * 100) if total else 0.0

    print("\n=== Holdout Test Summary ===")
    print(f"Total videos: {total}")
    print(f"Correct & >=threshold: {ok_count}")
    print(f"Correct but <threshold: {low_conf_correct}")
    print(f"Wrong predictions: {wrong_count}")
    print(f"Failures: {fail_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"Results saved to: {output_path}")

    print("\nDetailed Results:")
    for name, expected, pred, conf_score, top3_str, status in results:
        if status == 'FAIL':
            print(f" - {name}: FAIL")
        else:
            print(f" - {name}: expected={expected}, predicted={pred}, conf={conf_score:.3f}, status={status}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Holdout Test Runner: 비디오 폴더 내 모든 파일 일괄 예측"
    )
    script_dir = os.path.dirname(__file__)
    default_videos = os.path.join(script_dir, "videos")
    parser.add_argument(
        "--videos_dir", default=default_videos,
        help="hold_out_test/videos 폴더 경로"
    )
    parser.add_argument(
        "--output", default="holdout_results.csv",
        help="출력할 CSV 파일명"
    )
    parser.add_argument(
        "--seq", default="L20",
        help="윈도우 시퀀스 (예: L10, L20)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="신뢰도 문턱값"
    )
    parser.add_argument(
        "--temp", type=float, default=2.5,
        help="온도 스케일링 T"
    )
    args = parser.parse_args()

    main(
        videos_dir=args.videos_dir,
        output_csv_name=args.output,
        seq=args.seq,
        conf_thresh=args.conf,
        temp=args.temp
    )

