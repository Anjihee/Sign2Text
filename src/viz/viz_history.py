#!/usr/bin/env python3
# viz_history.py

import os
import argparse
import pickle
import matplotlib.pyplot as plt

def plot_history(history: dict, seq_name: str):
    epochs = range(1, len(history.get('loss', [])) + 1)

    # Loss
    plt.figure()
    plt.plot(epochs, history.get('loss', []), label='Train Loss')
    plt.plot(epochs, history.get('val_loss', []), label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss History ({seq_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(epochs, history.get('accuracy', []), label='Train Acc')
    plt.plot(epochs, history.get('val_accuracy', []), label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy History ({seq_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Visualize training history (loss & accuracy).'
    )
    # 여기에 단축 옵션을 주지 말고, --history-path 하나만 등록합니다.
    parser.add_argument(
        '--history-path',
        required=True,
        help='Path to the pickle file containing history.history dict'
    )
    args = parser.parse_args()

    history_path = args.history_path
    if not os.path.isfile(history_path):
        print(f"❌ 파일을 찾을 수 없습니다: {history_path}")
        return

    seq_name = os.path.splitext(os.path.basename(history_path))[0].split('_')[-1]

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    plot_history(history, seq_name)

if __name__ == '__main__':
    main()
