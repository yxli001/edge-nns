import os
import sys
import shutil
import yaml
import numpy as np

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import tensorflow as tf
keras = tf.keras


class QEnsemble:
    def __init__(
        self,
        config_file,
        load_model_fn,
        load_data_fn,
        compile_kwargs,
        size=4,
        seed=42,
        model_dir=None,
        results_dir=None,
    ):
        keras.utils.set_random_seed(seed)

        self.size = size
        self.seed = seed
        self.config_file = config_file
        self.load_data_fn = load_data_fn
        self.compile_kwargs = compile_kwargs
        self.models = []

        with open(config_file) as f:
            config = yaml.safe_load(f)
        input_shape = tuple(config["data"]["input_shape"])

        if model_dir is None:
            if results_dir is None:
                results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
            os.makedirs(results_dir, exist_ok=True)
            existing = [d for d in os.listdir(results_dir) if d.startswith("ensemble_")]
            next_idx = max([int(d.split("_")[1]) for d in existing], default=-1) + 1
            model_dir = os.path.join(results_dir, f"ensemble_{next_idx}")

        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        shutil.copy(config_file, os.path.join(self.model_dir, "config.yml"))

        shared_input = keras.layers.Input(shape=input_shape, name="ensemble_input")

        outputs = []
        for i in range(self.size):
            m = load_model_fn(config_file)
            self.models.append(m)
            for layer in m.layers:
                layer._name = f"m{i}_{layer.name}"
            out = m(shared_input)
            outputs.append(out)

        avg_output = keras.layers.Average()(outputs) if len(outputs) > 1 else outputs[0]
        self.ensemble_model = keras.Model(inputs=shared_input, outputs=avg_output)

        n_trainable = len(self.ensemble_model.trainable_weights)
        print(f"[QEnsemble] Built ensemble of {size} models | trainable weights: {n_trainable}")
        if n_trainable == 0:
            raise RuntimeError("Ensemble model has 0 trainable weights.")

    def train(self):
        (x_train, y_train), (x_test, y_test) = self.load_data_fn()

        self.ensemble_model.compile(**self.compile_kwargs)

        best_weights_path = os.path.join(self.model_dir, "ensemble_best.weights.h5")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                best_weights_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,  # avoids serialization issues with qkeras
            ),
        ]

        self.ensemble_model.fit(
            x_train,
            y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            callbacks=callbacks,
            verbose=1,
        )

        # Save final weights
        self.ensemble_model.save_weights(
            os.path.join(self.model_dir, "ensemble_final.weights.h5")
        )

        results = self.ensemble_model.evaluate(x_test, y_test, verbose=1, return_dict=True)
        self._save_eval(results)

        print(f"\n[QEnsemble] Saved to: {self.model_dir}")
        for k, v in results.items():
            print(f"  {k}: {float(v):.4f}")
        return results

    def predict(self, x):
        return self.ensemble_model.predict(x)

    def eval(self, x, y):
        results = self.ensemble_model.evaluate(x, y, verbose=0, return_dict=True)
        self._save_eval(results)
        return results

    def _save_eval(self, results):
        payload = {k: float(v) for k, v in results.items()}
        payload["ensemble_size"] = self.size
        payload["seed"] = self.seed
        with open(os.path.join(self.model_dir, "eval.yml"), "w") as f:
            yaml.dump(payload, f)
