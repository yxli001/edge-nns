import os
import sys
import shutil
import argparse
import yaml
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load_model import load_model

NUM_CLASSES = 10


class QEnsemble:
    def __init__(self, config_file, size=4, seed=42, model_dir=None):
        # trying random seeds for reproducibility, suggested by Olivia 
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()

        self.size = size
        self.seed = seed
        self.config_file = config_file
        self.model_dir = model_dir
        self.models = []

        if self.model_dir is None:
            results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
            os.makedirs(results_dir, exist_ok=True)
            existing = [d for d in os.listdir(results_dir) if d.startswith("ensemble_")]
            next_idx = max([int(d.split("_")[1]) for d in existing], default=-1) + 1
            self.model_dir = os.path.join(results_dir, f"ensemble_{next_idx}")

        os.makedirs(self.model_dir, exist_ok=True)
        shutil.copy(config_file, os.path.join(self.model_dir, "config.yml"))

        # averaged output
        for _ in range(self.size):
            self.models.append(load_model(config_file))

        input_layer = tf.keras.layers.Input(shape=(32, 32, 3), name="input")
        outputs = [model(input_layer) for model in self.models]
        avg_output = tf.keras.layers.Average()(outputs)
        self.ensemble_model = tf.keras.Model(inputs=input_layer, outputs=avg_output)

    def train(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        x_train = x_train / 255.
        x_test = x_test / 255.
        y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
        y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

        self.ensemble_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.categorical_crossentropy,
            metrics=[tf.keras.metrics.CategoricalAccuracy()],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.model_dir, "ensemble.h5"),
                monitor="val_loss",
                save_best_only=True,
            ),
        ]

        self.ensemble_model.fit(
            x_train, y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

        # Save individual member weights
        for i, model in enumerate(self.models):
            model.save(os.path.join(self.model_dir, f"model_{i}.h5"))

        loss, acc = self.ensemble_model.evaluate(x_test, y_test, verbose=0)

        with open(os.path.join(self.model_dir, "eval.txt"), "w") as f:
            f.write(f"Test loss: {loss}\n")
            f.write(f"Test accuracy: {acc}\n")
            f.write(f"Ensemble size: {self.size}\n")
            f.write(f"Seed: {self.seed}\n")

        print(f"Saved to: {self.model_dir}")
        print(f"Test loss: {loss:.4f} | Test accuracy: {acc:.4f}")
        return loss, acc

    def predict(self, x):
        return self.ensemble_model.predict(x)

    def eval(self, x, y):
        loss, acc = self.ensemble_model.evaluate(x, y, verbose=0)
        with open(os.path.join(self.model_dir, "eval.txt"), "w") as f:
            f.write(f"Test loss: {loss}\n")
            f.write(f"Test accuracy: {acc}\n")
            f.write(f"Ensemble size: {self.size}\n")
            f.write(f"Seed: {self.seed}\n")
        return loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config.yml")
    parser.add_argument("--size", type=int, default=4, help="Ensemble size (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    ensemble = QEnsemble(args.config, size=args.size, seed=args.seed)
    ensemble.train()
