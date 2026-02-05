import os
import csv
import yaml
import argparse

NUM_CLASSES = 3
INPUT_SHAPE = 13


def compute_num_params(dense_widths):

    prev = INPUT_SHAPE
    params = 0
    for w in dense_widths:
        params += prev * w + w  # QDense kernel + bias
        params += 4 * w         # BatchNormalization
        prev = w
    params += prev * NUM_CLASSES + NUM_CLASSES  # output QDense
    return params


def parse_accuracy_file(filepath):
    result = {}
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith("Test loss:"):
                result["test_loss"] = float(line.split(":")[1].strip())
            elif line.startswith("Test accuracy:"):
                result["test_accuracy"] = float(line.split(":")[1].strip())
    return result


def extract_results(model_dir="model_configs", output_path="results.csv"):
    if not os.path.isdir(model_dir):
        print(f"ERROR: Directory not found: {model_dir}")
        return

    rows = []
    for model_name in sorted(os.listdir(model_dir)):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        config_path = os.path.join(model_path, "config.yaml")
        accuracy_path = os.path.join(model_path, "accuracy.txt")

        if not os.path.exists(config_path) or not os.path.exists(accuracy_path):
            print(f"WARNING: Skipping {model_name} - missing config.yaml or accuracy.txt")
            continue

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        model_cfg = config["model"]
        dense_widths = model_cfg["dense_widths"]
        acc_data = parse_accuracy_file(accuracy_path)

        rows.append({
            "model_name": model_name,
            "dense_widths": "-".join(map(str, dense_widths)),
            "num_params": compute_num_params(dense_widths),
            "test_loss": acc_data.get("test_loss", ""),
            "test_accuracy": acc_data.get("test_accuracy", ""),
            "activation_bit_width": model_cfg["activation_total_bits"],
            "logit_bit_width": model_cfg["logit_total_bits"],
        })

    if not rows:
        print("No results found to extract.")
        return

    fieldnames = [
        "model_name",
        "dense_widths",
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

    print(f"Extracted {len(rows)} model results -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract model results into CSV")
    parser.add_argument("--model-dir", type=str, default="model_configs",
                        help="Directory containing trained model folders (default: model_configs)")
    parser.add_argument("--output", type=str, default="results.csv",
                        help="Output CSV path (default: results.csv)")
    args = parser.parse_args()
    extract_results(args.model_dir, args.output)


if __name__ == "__main__":
    main()
