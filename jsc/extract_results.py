import os
import csv
import yaml
import argparse
import sys 

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

def compute_num_params(dense_widths):
    prev = INPUT_SHAPE
    params = 0
    for w in dense_widths:
        params += prev * w + w  # QDense kernel + bias
        params += 4 * w         # BatchNormalization
        prev = w
    params += prev * NUM_CLASSES + NUM_CLASSES  # output QDense
    return params

def extract_results(model_dir="models", output_path="jsc_results.csv"):
    if not os.path.isdir(model_dir):
        print(f"ERROR: Directory not found: {model_dir}")
        return

    rows = []
    for model_name in sorted(os.listdir(model_dir)):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        config_path = os.path.join(model_path, "config.yml")
        eval_path = os.path.join(model_path, "eval.txt")

        if not os.path.exists(config_path) or not os.path.exists(eval_path):
            print(f"WARNING: Skipping {model_name} - missing config.yml or eval.txt")
            continue

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        eval_data = parse_eval_file(eval_path)

        model_cfg = config.get("model", {})
        quant_cfg = config.get("quantization", {})
        dense_widths = model_cfg.get("dense_widths", [])

        rows.append({
            "model_name": model_name,
            "dense_widths": "-".join(map(str, dense_widths)),
            "num_layers": len(dense_widths),
            "num_params": compute_num_params(dense_widths),
            "test_loss": eval_data.get("test_loss", ""),
            "test_accuracy": eval_data.get("test_accuracy", ""),
            "activation_bit_width": quant_cfg.get("activation_total_bits", ""),
            "logit_bit_width": quant_cfg.get("logit_total_bits", ""),
        })

    if not rows:
        print("No results found to extract.")
        return

    fieldnames = [
        "model_name",
        "dense_widths",
        "num_layers",
        "num_params",
        "test_loss",
        "test_accuracy",
        "activation_bit_width",
        "logit_bit_width",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} JSC model results -> {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract JSC model results into CSV")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory containing trained model folders")
    parser.add_argument("--output", type=str, default="jsc_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    extract_results(args.model_dir, args.output)

if __name__ == "__main__":
    main()