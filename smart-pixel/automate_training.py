#!/usr/bin/env python3
"""
Automated training for multiple model configs.

Usage:
    python automate_training.py                     # trains everything
    python automate_training.py --config path.yml   # trains one specific config
"""

import random
import argparse
import os
import yaml

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from train import load_data, load_model

# Enable eager execution for QKeras compatibility
tf.config.run_functions_eagerly(True)

random.seed(42)

def generate_configs():
    """
    Generate a bunch of different model configurations to test.
    Returns list of config dictionaries.
    """
    
    width_options = [16, 32, 58, 64, 128, 256, 512]
    
    activation_bits_options = [2, 4, 6, 8]
    logit_bits_options = [2, 4, 6, 8]
    
    configs = []
    
    #50 configs. 
    for _ in range(50):

        num_layers = random.randint(1, 3)
        
        widths = [random.choice(width_options) for _ in range(num_layers)]
        
        act_bits = random.choice(activation_bits_options)
        logit_bits = random.choice(logit_bits_options)
        
        config = {
            'model': {
                'name': 'qkeras_dense_model',
                'input_shape': 13,
                'dense_widths': widths,
                'logit_total_bits': logit_bits,
                'logit_int_bits': 0,
                'activation_total_bits': act_bits,
                'activation_int_bits': 0,
            }
        }
        configs.append(config)
    
    print(f"Generated {len(configs)} different configurations to train")
    return configs

def load_config(config_path):
    """Load yaml config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_output_directory(config, base_output_dir="model_configs"):
    """
    Make a folder for this model's outputs.
    Uses config params 
    Format: model_configs/qkeras_dense_model_58_8_0_4_0/
    """
    model_name = config['model']['name']
    widths = '_'.join(map(str, config['model']['dense_widths']))
    act_total = config['model']['activation_total_bits']
    act_int = config['model']['activation_int_bits']
    logit_total = config['model']['logit_total_bits']
    logit_int = config['model']['logit_int_bits']
    
    dir_name = f"{model_name}_{widths}_{act_total}_{act_int}_{logit_total}_{logit_int}"
    output_path = os.path.join(base_output_dir, dir_name)
    
    os.makedirs(output_path, exist_ok=True)
    return output_path


def train_single_config(config, base_output_dir="model_configs"):
    """
    Train one model from a config dict.
    Saves weights, accuracy, and the config yaml.
    """
    # create a string representation for logging
    widths_str = '_'.join(map(str, config['model']['dense_widths']))
    config_str = f"{widths_str} (act:{config['model']['activation_total_bits']}, logit:{config['model']['logit_total_bits']})"
    
    print(f"\n{'='*60}")
    print(f"Training config: {config_str}")
    print(f"{'='*60}\n")
    
    # make output folder
    output_dir = create_output_directory(config, base_output_dir)
    print(f"Output directory: {output_dir}")
    
    # load dataset
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    
    # build the model
    print("Building model...")
    model = load_model(config)
    
    # compile
    model.compile(
        optimizer=Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    
    print(model.summary())
    
    # train
    print("\nTraining...")
    weights_path = os.path.join(output_dir, "weights.h5")
    
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=150,
        batch_size=1024,
        callbacks=[checkpoint],
        shuffle=True,
        verbose=1
    )
    
    # load best weights for eval
    model.load_weights(weights_path)
    
    # evaluate on test set
    print("\nEvaluating...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # save accuracy to text file
    accuracy_path = os.path.join(output_dir, "accuracy.txt")
    with open(accuracy_path, 'w') as f:
        f.write(f"Test loss: {loss}\n")
        f.write(f"Test accuracy: {accuracy}\n")
    
    # save the config to yaml inside the output folder
    config_copy_path = os.path.join(output_dir, "config.yaml")
    with open(config_copy_path, 'w') as f:
        yaml_content = yaml.dump(config, default_flow_style=False, indent=2)
        f.write(yaml_content)
    
    print(f"\n✓ Done! Results saved to: {output_dir}")
    print("  - Weights: weights.h5")
    print("  - Accuracy: accuracy.txt")
    print("  - Config: config.yaml")
    
    return {
        'config': config,
        'output_dir': output_dir,
        'loss': loss,
        'accuracy': accuracy
    }

def train_generated_configs(base_output_dir="model_configs"):
    """
    Generate and train a bunch of different configs.
    """
    configs = generate_configs()
    
    # train each one
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Processing configuration {i}")
        try:
            result = train_single_config(config, base_output_dir)
            results.append(result)
        except Exception as e:
            print(f"ERROR training config {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for result in results:
        widths = '_'.join(map(str, result['config']['model']['dense_widths']))
        act_bits = result['config']['model']['activation_total_bits']
        logit_bits = result['config']['model']['logit_total_bits']
        config_name = f"{widths}_a{act_bits}_l{logit_bits}"
        print(f"{config_name:20s} | Acc: {result['accuracy']:.4f} | Loss: {result['loss']:.4f}")
    print("="*60)

def train_from_config_file(config_path, base_output_dir="model_configs"):
    """
    Train from a specific config yaml file.
    """
    print(f"Training from config file: {config_path}")
    config = load_config(config_path)
    train_single_config(config, base_output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="Automated training for Smart Pixel QKeras models"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to specific config file. If not provided, trains all configs in configs/ directory.'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='model_configs',
        help='Base directory for output folders (default: model_configs)'
    )
    
    args = parser.parse_args()
    
    if args.config:
        # train from a specific config file
        if not os.path.exists(args.config):
            print(f"ERROR: Config file not found: {args.config}")
            return
        train_from_config_file(args.config, args.output_dir)
    else:
        # generate and train a bunch of configs automatically
        train_generated_configs(base_output_dir=args.output_dir)


if __name__ == "__main__":
    main()