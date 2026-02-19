import os
import shutil
import tensorflow as tf
import yaml
from load_model import load_model
from train import ensure_gpu_or_fail
import argparse

NUM_CLASSES = 10

# train ensemble dependently with averaged outputs
class QEnsemble:
  def __init__(self, config_file, size=4, model_dir=None):
    ensure_gpu_or_fail()
    
    self.size = size
    self.config_file = config_file
    self.model_dir = model_dir 
    self.models = []

    if self.model_dir is None:
      # find next index
      next_index = 0
      q_ensembles_dir = "./q_ensembles"
      
      if os.path.exists(q_ensembles_dir):
        existing_dirs = [d for d in os.listdir(q_ensembles_dir) if d.startswith("ensemble_")]
        if existing_dirs:
          existing_indices = [int(d.split("_")[1]) for d in existing_dirs]
          next_index = max(existing_indices) + 1

      self.model_dir = f"{q_ensembles_dir}/ensemble_{next_index}"

    # create directory and save config
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    
    config_dest = f"{self.model_dir}/config.yml"
    shutil.copy(self.config_file, config_dest)

    # load base models
    for i in range(self.size):
      model = load_model(self.config_file)
      self.models.append(model)

    input_layer = tf.keras.layers.Input(shape=(32, 32, 3), name='input')
    outputs = [model(input_layer) for model in self.models]
    avg_output = tf.keras.layers.Average()(outputs)
    
    self.ensemble_model = tf.keras.Model(inputs=input_layer, outputs=avg_output)

  def train(self):
    # load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    x_train = x_train / 255.
    x_test = x_test / 255.

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # compile ensemble model
    self.ensemble_model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=tf.keras.losses.categorical_crossentropy,
      metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    callbacks = [
      tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
      tf.keras.callbacks.ModelCheckpoint(
        f"{self.model_dir}/ensemble.h5", 
        monitor='val_loss', 
        save_best_only=True
      ),
    ]

    # train ensemble model (all base models train together)
    self.ensemble_model.fit(
      x_train,
      y_train,
      epochs=200,
      batch_size=32,
      validation_split=0.2,
      shuffle=True,
      callbacks=callbacks,
    )

    # save individual models
    for i, model in enumerate(self.models):
      model_file = f"{self.model_dir}/model_{i}.h5"
      model.save(model_file)

    # eval
    loss, acc = self.ensemble_model.evaluate(x_test, y_test)
    
    # save eval results
    with open(f"{self.model_dir}/eval.txt", "w") as f:
      f.write(f"Test loss: {loss}\n")
      f.write(f"Test accuracy: {acc}\n")

    return loss, acc

  def predict(self, x):
    return self.ensemble_model.predict(x)

  def eval(self, x, y):
    loss, acc = self.ensemble_model.evaluate(x, y)

    # save eval
    with open(f"{self.model_dir}/eval.txt", "w") as f:
      f.write(f"Test loss: {loss}\n")
      f.write(f"Test accuracy: {acc}\n")
    
    return loss, acc
  
if __name__ == "__main__":
  tf.keras.utils.set_random_seed(42)

  # config file as argument
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True, help="Path to config file")
  args = parser.parse_args()

  with open(args.config, "r") as f:
    config = yaml.safe_load(f)

  ensemble = QEnsemble(args.config)
  ensemble.train()

  # eval
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_test = x_test / 255.
  y_test = tf.keras.utils.to_categorical(y_test, 10)

  loss, acc = ensemble.eval(x_test, y_test)
  print(f"Ensemble test loss: {loss}, test accuracy: {acc}")
