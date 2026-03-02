import os
import sys
import argparse
import glob

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qensemble import QEnsemble

_jsc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jsc")
sys.path.insert(0, _jsc_dir)
_spec = importlib.util.spec_from_file_location("jsc_load_model", os.path.join(_jsc_dir, "load_model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_model = _mod.load_model

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "jsc", "ensemble", "results"
)

# Load data once and reuse across all runs
_data_cache = None
def load_data():
    global _data_cache
    if _data_cache is None:
        print("Loading JSC dataset...")
        data = fetch_openml("hls4ml_lhc_jets_hlf", as_frame=False, cache=True)
        X = data["data"]
        y = data["target"]
        le = LabelEncoder()
        y = le.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        _data_cache = (X_train, y_train), (X_test, y_test)
        print("Dataset loaded and cached.")
    return _data_cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", type=str, nargs="+", required=True,
        help="Paths to config yml files, or a glob pattern e.g. '../jsc/ensemble/configs/*.yml'"
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
            configs.append(pattern)  # let it fail naturally if file doesn't exist

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
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
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