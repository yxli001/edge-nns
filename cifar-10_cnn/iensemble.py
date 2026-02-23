import os
import shutil
import yaml
import tensorflow as tf
from train import train
import argparse

# train ensemble independently and average predictions
class IEnsemble:
  def __init__(self, config_file, size=4, model_dir=None):
    self.size = size
    self.config_file = config_file
    self.model_dir = model_dir 
    self.models = []

    if self.model_dir is None:
      # find next index
      next_index = 0
      i_ensembles_dir = "./i_ensembles"
      
      if os.path.exists(i_ensembles_dir):
        existing_dirs = [d for d in os.listdir(i_ensembles_dir) if d.startswith("ensemble_")]
        if existing_dirs:
          existing_indices = [int(d.split("_")[1]) for d in existing_dirs]
          next_index = max(existing_indices) + 1

      self.model_dir = f"{i_ensembles_dir}/ensemble_{next_index}"

  def train(self):
    # save config
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    
    config_file = f"{self.model_dir}/config.yml"
    shutil.copy(self.config_file, config_file)

    for i in range(self.size):
      model_file = f"{self.model_dir}/model_{i}.h5"

      model, _, _ = train(self.config_file, model_file)
      model.save(model_file)

      self.models.append(model)

  def predict(self, x):
    preds = [model.predict(x) for model in self.models]

    avg_preds = sum(preds) / len(preds)

    return avg_preds 

  def eval(self, x, y):
    avg_preds = self.predict(x)

    loss = tf.keras.losses.categorical_crossentropy(y, avg_preds)
    acc = tf.keras.metrics.categorical_accuracy(y, avg_preds)

    # save eval
    with open(f"{self.model_dir}/eval.txt", "w") as f:
      f.write(f"Test loss: {loss.numpy().mean()}\n")
      f.write(f"Test accuracy: {acc.numpy().mean()}\n")

    return loss.numpy().mean(), acc.numpy().mean()
  

if __name__ == "__main__":
  tf.keras.utils.set_random_seed(42)

  # config file as argument
  parser = argparse.ArgumentParser()
  parser.add_argument("--config", type=str, required=True, help="Path to config file")
  args = parser.parse_args()

  with open(args.config, "r") as f:
    config = yaml.safe_load(f)

  ensemble = IEnsemble(args.config)
  ensemble.train()

  # eval
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  x_test = x_test / 255.
  y_test = tf.keras.utils.to_categorical(y_test, 10)

  loss, acc = ensemble.eval(x_test, y_test)
  print(f"Ensemble test loss: {loss}, test accuracy: {acc}")
