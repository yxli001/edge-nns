import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from load_model import load_model

NUM_CLASSES = 10


def _list_model_files(ensemble_dir: str) -> List[str]:
    pattern = os.path.join(ensemble_dir, "model_*.h5")
    files = glob.glob(pattern)

    def sort_key(path: str) -> int:
        name = os.path.basename(path)
        match = re.match(r"model_(\d+)\.h5", name)
        return int(match.group(1)) if match else -1

    return sorted(files, key=sort_key)


def _load_models(ensemble_dir: str) -> Tuple[List[tf.keras.Model], str]:
    config_path = os.path.join(ensemble_dir, "config.yml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config.yml in {ensemble_dir}")

    model_files = _list_model_files(ensemble_dir)
    if not model_files:
        raise FileNotFoundError(f"No model_*.h5 files found in {ensemble_dir}")

    models = []
    for model_file in model_files:
        model = load_model(config_path, pretrained_model=model_file)
        models.append(model)

    return models, config_path


def _predict_models(models: List[tf.keras.Model], x: np.ndarray, batch_size: int) -> np.ndarray:
    preds = []
    for model in models:
        preds.append(model.predict(x, batch_size=batch_size, verbose=0))
    return np.stack(preds, axis=0)


def _accuracy_from_probs(probs: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.argmax(probs, axis=1)
    return float(np.mean(y_pred == y_true))


def _pairwise_disagreement(preds: np.ndarray) -> np.ndarray:
    num_models = preds.shape[0]
    labels = np.argmax(preds, axis=2)
    matrix = np.zeros((num_models, num_models), dtype=np.float32)
    for i in range(num_models):
        for j in range(num_models):
            matrix[i, j] = np.mean(labels[i] != labels[j])
    return matrix

def _summarize_ensemble(name: str, preds: np.ndarray, y_true: np.ndarray) -> Dict:
    per_model_acc = [_accuracy_from_probs(preds[i], y_true) for i in range(preds.shape[0])]
    ensemble_probs = np.mean(preds, axis=0)
    ensemble_acc = _accuracy_from_probs(ensemble_probs, y_true)

    disagreement = _pairwise_disagreement(preds)
    mean_disagreement = float(np.mean(disagreement[np.triu_indices(preds.shape[0], k=1)]))

    return {
        "name": name,
        "ensemble_acc": ensemble_acc,
        "mean_model_acc": float(np.mean(per_model_acc)),
        "mean_disagreement": mean_disagreement,
        "per_model_acc": per_model_acc,
    }


def _plot_accuracy(out_dir: str, stats_i: Dict[str, float], stats_q: Dict[str, float]) -> None:
    labels = ["independent", "dependent"]
    ensemble_accs = [stats_i["ensemble_acc"], stats_q["ensemble_acc"]]
    mean_model_accs = [stats_i["mean_model_acc"], stats_q["mean_model_acc"]]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(6, 4))
    plt.bar(x - width / 2, mean_model_accs, width, label="Mean member acc")
    plt.bar(x + width / 2, ensemble_accs, width, label="Ensemble acc")
    plt.xticks(x, labels)
    # plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Accuracy comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_comparison.png"))
    plt.close()


def _plot_disagreement(out_dir: str, name: str, matrix: np.ndarray) -> None:
    plt.figure(figsize=(5, 4))
    plt.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, origin="lower")
    plt.colorbar(label="Disagreement")
    plt.title(f"Pairwise disagreement: {name}")
    indices = np.arange(matrix.shape[0])
    plt.xticks(indices, [str(i) for i in indices])
    plt.yticks(indices, [str(i) for i in indices])
    plt.xlabel("Model index")
    plt.ylabel("Model index")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_disagreement.png"))
    plt.close()

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare i_ensemble vs q_ensemble on CIFAR-10")
    parser.add_argument("--i-ensemble-dir", required=True, help="Path to i_ensembles/ensemble_X")
    parser.add_argument("--q-ensemble-dir", required=True, help="Path to q_ensembles/ensemble_X")
    parser.add_argument("--output-dir", default="./ensemble_compare", help="Output directory for plots")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for prediction")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of test samples (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test / 255.0
    y_test = y_test.squeeze()

    if args.max_samples > 0:
        x_test = x_test[: args.max_samples]
        y_test = y_test[: args.max_samples]

    i_models, _ = _load_models(args.i_ensemble_dir)
    q_models, _ = _load_models(args.q_ensemble_dir)

    i_preds = _predict_models(i_models, x_test, args.batch_size)
    q_preds = _predict_models(q_models, x_test, args.batch_size)

    stats_i = _summarize_ensemble("independent", i_preds, y_test)
    stats_q = _summarize_ensemble("dependent", q_preds, y_test)

    i_disagreement = _pairwise_disagreement(i_preds)
    q_disagreement = _pairwise_disagreement(q_preds)

    _plot_accuracy(args.output_dir, stats_i, stats_q)
    _plot_disagreement(args.output_dir, "independent", i_disagreement)
    _plot_disagreement(args.output_dir, "dependent", q_disagreement)

    print("Saved plots and summary to:", args.output_dir)


if __name__ == "__main__":
    main()
