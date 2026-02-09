import os
import csv
import yaml
import argparse
import sys


def parse_eval_file(filepath):
    result = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("Test loss:"):
                result["test_loss"] = float(line.split(":")[1].strip())
            elif line.startswith("Test accuracy:"):
                result["test_accuracy"] = float(line.split(":")[1].strip())
    return result


def estimate_resnet_params(config):
    try:
        filters = config['model'].get('filters', [])
        input_shape = config['data']['input_shape']
        num_classes = config['data']['num_classes']
        params = 3 * 3 * 3 * filters[0] + filters[0]  # weights + bias

        for i in range(len(filters)):
            if i > 0:
                # conv layers in residual blocks
                params += 3 * 3 * filters[i-1] * filters[i] + filters[i]
                params += 3 * 3 * filters[i] * filters[i] + filters[i]

        # after global average pooling, we have filters[-1] features
        params += filters[-1] * num_classes + num_classes

        return params
    except Exception as e:
        print(f"Warning: Could not estimate parameters: {e}")
        return None


def extract_results(model_dir="models", output_path="results.csv"):
    """Extract results from all trained CIFAR-10 models."""
    if not os.path.isdir(model_dir):
        print(f"ERROR: Directory not found: {model_dir}")
        return

    rows = []
    for model_name in sorted(os.listdir(model_dir), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else 0):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        config_path = os.path.join(model_path, "config.yml")
        eval_path = os.path.join(model_path, "eval.txt")
        h5_path = os.path.join(model_path, "model.h5")

        if not os.path.exists(config_path) or not os.path.exists(eval_path):
            print(f"WARNING: Skipping {model_name} - missing config.yml or eval.txt")
            continue

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_cfg = config["model"]
        quant_cfg = config.get("quantization", {})

        filters = model_cfg.get("filters", [])
        filters_str = "-".join(map(str, filters))

        eval_data = parse_eval_file(eval_path)

        num_params = estimate_resnet_params(config)

        rows.append({
            "model_name": model_name,
            "filters": filters_str,
            "num_params": num_params if num_params is not None else "",
            "test_loss": eval_data.get("test_loss", ""),
            "test_accuracy": eval_data.get("test_accuracy", ""),
            "activation_bit_width": quant_cfg.get("activation_total_bits", ""),
            "logit_bit_width": quant_cfg.get("logit_total_bits", ""),
            "activation_int_bits": quant_cfg.get("activation_int_bits", ""),
            "logit_int_bits": quant_cfg.get("logit_int_bits", ""),
        })

    if not rows:
        print("No results found to extract.")
        return

    fieldnames = [
        "model_name",
        "filters",
        "num_params",
        "test_loss",
        "test_accuracy",
        "activation_bit_width",
        "logit_bit_width",
        "activation_int_bits",
        "logit_int_bits",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} model results -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract CIFAR-10 model results into CSV")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory containing trained model folders (default: models)")
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Output CSV path (default: results.csv)")
    args = parser.parse_args()
    extract_results(args.model_dir, args.output)


if __name__ == "__main__":
    main()
