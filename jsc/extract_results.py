import os
import csv
import yaml
import argparse
import sys 

def parse_eval_file(filepath):
    result = {}
    with open(filepath, "r") as f:
        for line in f:
            if "Test loss:" in line:
                result["test_loss"] = float(line.split(":")[1].strip())
            elif "Test accuracy:" in line:
                result["test_accuracy"] = float(line.split(":")[1].strip())
    return result

def estimate_model_params(config):
    return None

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

        rows.append({
            "model_name": model_name,
            "test_loss": eval_data.get("test_loss", ""),
            "test_accuracy": eval_data.get("test_accuracy", ""),
        })

    if not rows:
        print("No results found to extract.")
        return

    fieldnames = ["model_name", "test_loss", "test_accuracy"]

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