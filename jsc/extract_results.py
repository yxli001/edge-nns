import os
import csv
import yaml
import argparse
import json

# Disable GPU before importing TensorFlow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from load_model import load_model

INPUT_SHAPE = 16
NUM_CLASSES = 5

def parse_eval_file(filepath):
    result = {}
    with open(filepath, "r") as f:
        for line in f:
            if "Test loss:" in line:
                result["test_loss"] = float(line.split(":")[1].strip())
            elif "Test accuracy:" in line:
                result["test_accuracy"] = float(line.split(":")[1].strip())
    return result

def parse_eval_yaml(filepath):
    """Parse ensemble eval.yml file"""
    result = {}
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    result["ensemble_size"] = data.get("ensemble_size", "")
    result["loss"] = data.get("loss", "")
    result["accuracy"] = data.get("sparse_categorical_accuracy", "")
    return result

def count_model_parameters(config_path, ensemble_size=1):
    """Load model architecture and count parameters (without loading weights)
    For ensembles, multiply by ensemble_size since each model is replicated"""
    try:
        model = load_model(config_path)  # Don't load weights, just build architecture
        params_per_model = model.count_params()
        return params_per_model * ensemble_size
    except Exception as e:
        print(f"  ERROR counting params for {config_path}: {e}")
        return ""

def extract_results(model_dir="models", ensemble_dir="ensemble/results", output_path="jsc_results.csv"):
    rows = []
    
    # Extract individual models
    if os.path.isdir(model_dir):
        print(f"Extracting individual models from {model_dir}...")
        for model_name in sorted(os.listdir(model_dir)):
            model_path = os.path.join(model_dir, model_name)
            if not os.path.isdir(model_path):
                continue

            config_path = os.path.join(model_path, "config.yml")
            eval_path = os.path.join(model_path, "eval.txt")

            if not os.path.exists(config_path) or not os.path.exists(eval_path):
                print(f"  ✗ {model_name} - missing config.yml or eval.txt")
                continue

            print(f"  Processing {model_name}...", end=" ")
            
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            eval_data = parse_eval_file(eval_path)

            model_cfg = config.get("model", {})
            quant_cfg = config.get("quantization", {})
            dense_widths = model_cfg.get("dense_widths", [])

            num_params = count_model_parameters(config_path, ensemble_size=1)
            
            rows.append({
                "ensemble_size": 1,
                "ensemble_dir": model_name,
                "dense_widths": json.dumps(dense_widths),
                "model_config": json.dumps(model_cfg, sort_keys=True),
                "logit_total_bits": quant_cfg.get("logit_total_bits", ""),
                "activation_total_bits": quant_cfg.get("activation_total_bits", ""),
                "val_loss": eval_data.get("test_loss", ""),
                "val_accuracy": eval_data.get("test_accuracy", ""),
                "num_parameters": num_params,
            })
            print(f"✓ (params: {num_params})")
    else:
        print(f"WARNING: Model directory not found: {model_dir}")
    
    # Extract ensembles
    if os.path.isdir(ensemble_dir):
        print(f"Extracting ensembles from {ensemble_dir}...")
        for ens_name in sorted(os.listdir(ensemble_dir)):
            ens_path = os.path.join(ensemble_dir, ens_name)
            if not os.path.isdir(ens_path):
                continue

            config_path = os.path.join(ens_path, "config.yml")
            eval_path = os.path.join(ens_path, "eval.yml")

            if not os.path.exists(config_path) or not os.path.exists(eval_path):
                print(f"  ✗ {ens_name} - missing config.yml or eval.yml")
                continue

            print(f"  Processing {ens_name}...", end=" ")
            
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            eval_data = parse_eval_yaml(eval_path)

            model_cfg = config.get("model", {})
            quant_cfg = config.get("quantization", {})
            dense_widths = model_cfg.get("dense_widths", [])
            ens_size = eval_data.get("ensemble_size", 1)

            num_params = count_model_parameters(config_path, ensemble_size=ens_size)
            
            rows.append({
                "ensemble_size": ens_size,
                "ensemble_dir": ens_name,
                "dense_widths": json.dumps(dense_widths),
                "model_config": json.dumps(model_cfg, sort_keys=True),
                "logit_total_bits": quant_cfg.get("logit_total_bits", ""),
                "activation_total_bits": quant_cfg.get("activation_total_bits", ""),
                "val_loss": eval_data.get("loss", ""),
                "val_accuracy": eval_data.get("accuracy", ""),
                "num_parameters": num_params,
            })
            print(f"✓ (size: {ens_size}, params: {num_params})")
    else:
        print(f"WARNING: Ensemble directory not found: {ensemble_dir}")

    if not rows:
        print("No results found to extract.")
        return

    fieldnames = [
        "ensemble_size",
        "ensemble_dir",
        "dense_widths",
        "model_config",
        "logit_total_bits",
        "activation_total_bits",
        "val_loss",
        "val_accuracy",
        "num_parameters",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nExtracted {len(rows)} JSC results -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract JSC model and ensemble results into CSV")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory containing trained model folders")
    parser.add_argument("--ensemble-dir", type=str, default="ensemble/results",
                        help="Directory containing trained ensemble folders")
    parser.add_argument("--output", type=str, default="jsc_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    extract_results(args.model_dir, args.ensemble_dir, args.output)

if __name__ == "__main__":
    main()