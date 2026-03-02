import os
import sys
import argparse
import tensorflow as tf

import importlib.util

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qensemble import QEnsemble

_cifar_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cifar-10_cnn")
sys.path.insert(0, _cifar_dir)  # resnet_v1_eembc etc
_spec = importlib.util.spec_from_file_location("cifar10_load_model", os.path.join(_cifar_dir, "load_model.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_model = _mod.load_model

NUM_CLASSES = 10

COMPILE_KWARGS = dict(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
    metrics=[tf.keras.metrics.CategoricalAccuracy()],
)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "cifar-10_cnn", "ensemble", "results"
)


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    return (x_train, y_train), (x_test, y_test)


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
