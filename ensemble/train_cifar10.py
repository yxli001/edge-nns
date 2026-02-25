import os
import sys
import argparse
import glob

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qensemble import QEnsemble

_cifar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cifar-10_cnn")
sys.path.insert(0, _cifar_dir)
_spec = importlib.util.spec_from_file_location("cifar10_load_model", os.path.join(_cifar_dir, "load_model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_model = _mod.load_model

NUM_CLASSES = 10

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cifar-10_cnn", "ensemble", "results"
)

# Load data once and reuse across all runs
_data_cache = None
def load_data():
    global _data_cache
    if _data_cache is None:
        print("Loading CIFAR-10 dataset...")
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train / 255.0
        x_test = x_test / 255.0
        y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
        _data_cache = (x_train, y_train), (x_test, y_test)
        print("Dataset loaded and cached.")
    return _data_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", type=str, nargs="+", required=True,
        help="Paths to config yml files, or a glob pattern e.g. '../cifar-10_cnn/ensemble/configs/*.yml'"
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[2, 3, 4, 5],
        help="Ensemble sizes to train (default: 2 3 4 5)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    # Expand any glob patterns
    configs = []
    for pattern in args.configs:
        matches = sorted(glob.glob(pattern))
        if matches:
            configs.extend(matches)
        else:
            configs.append(pattern)

    total = len(configs) * len(args.sizes)
    run = 0

    for config in configs:
        for size in args.sizes:
            run += 1
            print(f"\n{'='*60}")
            print(f"Run {run}/{total} | config={os.path.basename(config)} | size={size}")
            print(f"{'='*60}\n")

            compile_kwargs = dict(
                optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=[tf.keras.metrics.CategoricalAccuracy()],
            )

            try:
                ensemble = QEnsemble(
                    config_file=config,
                    load_model_fn=load_model,
                    load_data_fn=load_data,
                    compile_kwargs=compile_kwargs,
                    size=size,
                    seed=args.seed,
                    results_dir=RESULTS_DIR,
                )
                ensemble.train()
            except Exception as e:
                print(f"ERROR on config={config} size={size}: {e}")
                print("Skipping and continuing...\n")
                continue

    print(f"\nAll {total} runs complete. Results in: {RESULTS_DIR}")