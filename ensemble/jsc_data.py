import os
import sys
import argparse
import tensorflow as tf
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qensemble import QEnsemble

_jsc_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "jsc")
sys.path.insert(0, _jsc_dir)  # for models.py etc
_spec = importlib.util.spec_from_file_location("jsc_load_model", os.path.join(_jsc_dir, "load_model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_model = _mod.load_model

COMPILE_KWARGS = dict(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "jsc", "ensemble", "results"
)


def load_data():
    data = fetch_openml("hls4ml_lhc_jets_hlf", as_frame=False)
    X = data["data"]
    y = data["target"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yml")
    parser.add_argument("--size", type=int, default=4, help="Ensemble size (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    ensemble = QEnsemble(
        config_file=args.config,
        load_model_fn=load_model,
        load_data_fn=load_data,
        compile_kwargs=COMPILE_KWARGS,
        size=args.size,
        seed=args.seed,
        results_dir=RESULTS_DIR,
    )
    ensemble.train()
