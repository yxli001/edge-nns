#!/usr/bin/env python3
"""
Automated training for multiple model configs.

Usage:
    python automate_training.py                     # trains everything
    python automate_training.py --config path.yml   # trains one specific config
"""

import argparse
import glob
import os
import yaml
import shutil
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Enable eager execution for QKeras compatibility
tf.config.run_functions_eagerly(True)

from train import load_data, load_model


def load_config(config_path):
    """Load yaml config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_output_directory(config, base_output_dir="model_configs"):
    """
    Make a folder for this model's outputs.
    Uses config params instead of timestamp so identical configs overwrite.
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


def train_single_config(config_path, base_output_dir="model_configs"):
    """
    Train one model from a config file.
    Saves weights, accuracy, and a copy of the config.
    """
    print(f"\n{'='*60}")
    print(f"Training: {config_path}")
    print(f"{'='*60}\n")
    
    # load the config
    config = load_config(config_path)
    
    # make output folder
    output_dir = create_output_directory(config, base_output_dir)
    print(f"Output directory: {output_dir}")
    
    # load dataset
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    
    # build the model
    print("Building model...")

    # fix input_shape format - needs to be tuple not int
    if isinstance(config['model']['input_shape'], int):
        config['model']['input_shape'] = (config['model']['input_shape'],)
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
    
    history = model.fit(
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
    
    # copy the config into the output folder so we know what settings were used
    config_copy_path = os.path.join(output_dir, "config.yaml")
    shutil.copy(config_path, config_copy_path)
    
    print(f"\n✓ Done! Results saved to: {output_dir}")
    print(f"  - Weights: weights.h5")
    print(f"  - Accuracy: accuracy.txt")
    print(f"  - Config: config.yaml")
    
    return {
        'config_path': config_path,
        'output_dir': output_dir,
        'loss': loss,
        'accuracy': accuracy
    }


def train_all_configs(config_dir="configs", base_output_dir="model_configs"):
    """
    Train all the config files in the configs directory.
    """
    # find all yml files
    config_pattern = os.path.join(config_dir, "*.yml")
    config_files = sorted(glob.glob(config_pattern))
    
    if not config_files:
        print(f"No config files found in {config_dir}")
        return
    
    print(f"Found {len(config_files)} config files to train:")
    for cf in config_files:
        print(f"  - {cf}")
    print()
    
    # train each one
    results = []
    for i, config_path in enumerate(config_files, 1):
        print(f"\n[{i}/{len(config_files)}] Processing {config_path}")
        try:
            result = train_single_config(config_path, base_output_dir)
            results.append(result)
        except Exception as e:
            print(f"ERROR training {config_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # print summary of all results
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for result in results:
        config_name = os.path.basename(result['config_path'])
        print(f"{config_name:30s} | Acc: {result['accuracy']:.4f} | Loss: {result['loss']:.4f}")
    print("="*60)


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
        # train just one config
        if not os.path.exists(args.config):
            print(f"ERROR: Config file not found: {args.config}")
            return
        train_single_config(args.config, args.output_dir)
    else:
        # train all configs in the configs folder
        train_all_configs(config_dir="configs", base_output_dir=args.output_dir)


if __name__ == "__main__":
    main()