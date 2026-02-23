import copy
import os
import itertools
import yaml
from train import train

NUM_CLASSES = 10

def gen_configs():
  stack_options = [
    {
      "filters": [8, 8],
      "kernels": [3, 3, 3],
      "strides": ["111"],
    },
    {
      "filters": [8, 8],
      "kernels": [3, 3, 3],
      "strides": ["411"],
    },
    {
      "filters": [12, 12],
      "kernels": [3, 3, 3],
      "strides": ["111"],
    },
    {
      "filters": [12, 12],
      "kernels": [3, 3, 3],
      "strides": ["411"],
    },
    {
      "filters": [16, 16],
      "kernels": [3, 3, 3],
      "strides": ["111"],
    },
    {
      "filters": [16, 16],
      "kernels": [3, 3, 3],
      "strides": ["411"],
    },
    {
      "filters": [24, 24],
      "kernels": [3, 3, 3],
      "strides": ["111"],
    },
    {
      "filters": [24, 24],
      "kernels": [3, 3, 3],
      "strides": ["411"],
    },
    {
      "filters": [8, 8, 16, 16],
      "kernels": [3, 3, 3, 3, 3, 1],
      "strides": ["111", "212"],
    },
    {
      "filters": [12, 12, 24, 24],
      "kernels": [3, 3, 3, 3, 3, 1],
      "strides": ["111", "212"],
    },
    {
      "filters": [16, 16, 32, 32],
      "kernels": [3, 3, 3, 3, 3, 1],
      "strides": ["111", "212"],
    },
    {
      "filters": [24, 24, 48, 48],
      "kernels": [3, 3, 3, 3, 3, 1],
      "strides": ["111", "212"],
    },
    {
      "filters": [8, 8, 16, 16, 32, 32],
      "kernels": [3, 3, 3, 3, 3, 1, 3, 3, 1],
      "strides": ["111", "212", "212"],
    },
    {
      "filters": [12, 12, 24, 24, 48, 48],
      "kernels": [3, 3, 3, 3, 3, 1, 3, 3, 1],
      "strides": ["111", "212", "212"],
    },
    {
      "filters": [16, 16, 32, 32, 64, 64],
      "kernels": [3, 3, 3, 3, 3, 1, 3, 3, 1],
      "strides": ["111", "212", "212"],
    },
    {
      "filters": [24, 24, 48, 48, 96, 96],
      "kernels": [3, 3, 3, 3, 3, 1, 3, 3, 1],
      "strides": ["111", "212", "212"],
    },
  ]

  activation_total_bits_options = [2, 4, 6, 8]
  logits_total_bits_options = [2, 4, 6, 8]

  base_config = {
    'data': {
      'name': 'cifar10',
      'input_shape': [32, 32, 3],
      'num_classes': 10
    },
    'pruning': {
      'sparsity': 1.0
    },
    'fit': {
      'compile': {
        'initial_lr': 0.001,
        'lr_decay': 0.99,
        'optimizer': 'Adam',
        'loss': 'categorical_crossentropy',
      },
      'epochs': 200,
      'patience': 40,
      'batch_size': 32,
      'verbose': 1
    }
  }

  configs = []

  for stack, activation_total_bits, logits_total_bits in itertools.product(
    stack_options,
    activation_total_bits_options,
    logits_total_bits_options,
  ):
    config = copy.deepcopy(base_config)
    filters = stack["filters"]
    kernels = stack["kernels"]
    strides = stack["strides"]

    config['model'] = {
      "name": "resnet_v1_eembc_quantized",
      "filters": filters,
      "l1": 0,
      "l2": 1e-4,
      "kernels": kernels,
      "strides": strides,
      "skip": 1,
      "avg_pooling": 0,
      "final_activation": 1
    }

    config['quantization'] = {
      "logit_total_bits": logits_total_bits,
      "logit_int_bits": 2 if logits_total_bits > 2 else 1,
      "logit_quantizer": "quantized_bits",
      "activation_total_bits": activation_total_bits,
      "activation_int_bits": 2 if activation_total_bits > 2 else 1,
      "activation_quantizer": "quantized_relu",
      "alpha": 1,
      "use_stochastic_rounding": 0
    }

    configs.append(config)

  return configs


def train_all():
  configs = gen_configs()

  if not os.path.exists("./models"):
    os.makedirs("./models")

  for i, config in enumerate(configs):
    model_dir = f"./models/model_{i}"
    config_file = f"{model_dir}/config.yml"
    model_file = f"{model_dir}/model.h5"
    model_log_file = f"{model_dir}/eval.txt"

    # if eval.txt exists, skip training
    if os.path.exists(os.path.join(model_dir, "eval.txt")):
      print(f"Model {i} already trained, skipping...")
      continue

    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
      with open(config_file, "w") as f:
        yaml.dump(config, f)

    model, loss, acc = train(config_file, model_file)

    model.save(model_file)

    # save model and accuracies
    with open(model_log_file, "w") as f:
      f.write(f"Test loss: {loss}\n")
      f.write(f"Test accuracy: {acc}\n")

if __name__ == "__main__":
  train_all()