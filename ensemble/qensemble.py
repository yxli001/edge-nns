import os
import shutil
import yaml
import tensorflow as tf


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
        # reproducibility, suggested by Olivia
        tf.keras.utils.set_random_seed(seed)
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass

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

        for _ in range(self.size):
            self.models.append(load_model_fn(config_file))

        input_layer = tf.keras.layers.Input(shape=input_shape, name="input")
        outputs = [model(input_layer) for model in self.models]
        avg_output = tf.keras.layers.Average()(outputs)
        self.ensemble_model = tf.keras.Model(inputs=input_layer, outputs=avg_output)

    def train(self):
        (x_train, y_train), (x_test, y_test) = self.load_data_fn()

        self.ensemble_model.compile(**self.compile_kwargs)

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
            x_train,
            y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            shuffle=True,
            callbacks=callbacks,
            verbose=0,
        )

        for i, model in enumerate(self.models):
            model.save_weights(os.path.join(self.model_dir, f"model_{i}.weights.h5"))

        results = self.ensemble_model.evaluate(x_test, y_test, verbose=0, return_dict=True)
        self._save_eval(results)

        print(f"Saved to: {self.model_dir}")
        for k, v in results.items():
            print(f"{k}: {float(v):.4f}")
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
