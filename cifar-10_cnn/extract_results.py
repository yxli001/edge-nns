import argparse
import csv
import json
import re
from pathlib import Path
import os

# Disable GPU to save memory during model parameter counting
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import yaml

from load_model import load_model


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Extract architecture, quantization bits, and validation metrics "
			"from ensemble result folders and individual models into a CSV file."
		)
	)
	parser.add_argument(
		"--results-dir",
		type=Path,
		default=Path("ensemble/results"),
		help="Directory containing ensemble_* subdirectories.",
	)
	parser.add_argument(
		"--models-dir",
		type=Path,
		default=Path("models"),
		help="Directory containing model_* subdirectories.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("cifar_results.csv"),
		help="Path for output CSV file.",
	)
	return parser.parse_args()


def load_yaml(path: Path) -> dict:
	with path.open("r", encoding="utf-8") as file:
		data = yaml.safe_load(file)
	return data or {}


def ensemble_sort_key(path: Path) -> tuple:
	match = re.search(r"(\d+)$", path.name)
	if match:
		return (0, int(match.group(1)))
	return (1, path.name)


def parse_eval_txt(path: Path) -> dict:
	"""Parse eval.txt file (plain text format)"""
	data = {}
	if not path.exists():
		return data
	
	with path.open("r", encoding="utf-8") as file:
		for line in file:
			line = line.strip()
			if "accuracy:" in line.lower():
				try:
					data["accuracy"] = float(line.split(":")[-1].strip())
				except ValueError:
					pass
			elif "loss:" in line.lower():
				try:
					data["loss"] = float(line.split(":")[-1].strip())
				except ValueError:
					pass
	return data


def get_accuracy(eval_data: dict):
	return eval_data.get("categorical_accuracy", eval_data.get("accuracy"))


def count_model_parameters(config_path: Path, ensemble_size=1) -> int:
	"""Load model architecture from config and count total parameters (without loading weights)"""
	if not config_path.exists():
		return None
	
	try:
		model = load_model(str(config_path), pretrained_model=None)
		return model.count_params() * ensemble_size
	except Exception as e:
		print(f"Warning: Could not build model from {config_path}: {e}")
		return None


def main() -> None:
	args = parse_args()
	results_dir = args.results_dir
	models_dir = args.models_dir

	if not results_dir.exists() or not results_dir.is_dir():
		raise FileNotFoundError(f"Results directory not found: {results_dir}")

	rows = []

	# Extract from ensemble results
	ensemble_dirs = sorted(
		[directory for directory in results_dir.iterdir() if directory.is_dir()],
		key=ensemble_sort_key,
	)

	for ensemble_dir in ensemble_dirs:
		config_path = ensemble_dir / "config.yml"
		eval_path = ensemble_dir / "eval.yml"

		if not config_path.exists() or not eval_path.exists():
			continue

		config_data = load_yaml(config_path)
		eval_data = load_yaml(eval_path)

		model_data = config_data.get("model", {})
		quant_data = config_data.get("quantization", {})
		ensemble_size = eval_data.get("ensemble_size", 1)
		num_parameters = count_model_parameters(config_path, ensemble_size)

		rows.append(
			{
				"ensemble_size": ensemble_size,
				"ensemble_dir": ensemble_dir.name,
				"filters": json.dumps(model_data.get("filters")),
				"kernels": json.dumps(model_data.get("kernels")),
				"strides": json.dumps(model_data.get("strides")),
				"model_config": json.dumps(model_data, sort_keys=True),
				"logit_total_bits": quant_data.get("logit_total_bits"),
				"activation_total_bits": quant_data.get("activation_total_bits"),
				"val_loss": eval_data.get("loss"),
				"val_accuracy": get_accuracy(eval_data),
				"num_parameters": num_parameters,
			}
		)

	# Extract from individual models
	if models_dir.exists() and models_dir.is_dir():
		model_dirs = sorted(
			[directory for directory in models_dir.iterdir() if directory.is_dir()],
			key=ensemble_sort_key,
		)

		for model_dir in model_dirs:
			config_path = model_dir / "config.yml"
			eval_path = model_dir / "eval.txt"

			if not config_path.exists():
				continue

			config_data = load_yaml(config_path)
			eval_data = parse_eval_txt(eval_path) if eval_path.exists() else {}

			model_data = config_data.get("model", {})
			quant_data = config_data.get("quantization", {})
			num_parameters = count_model_parameters(config_path)

			rows.append(
				{
					"ensemble_size": 1,
					"ensemble_dir": model_dir.name,
					"filters": json.dumps(model_data.get("filters")),
					"kernels": json.dumps(model_data.get("kernels")),
					"strides": json.dumps(model_data.get("strides")),
					"model_config": json.dumps(model_data, sort_keys=True),
					"logit_total_bits": quant_data.get("logit_total_bits"),
					"activation_total_bits": quant_data.get("activation_total_bits"),
					"val_loss": eval_data.get("loss"),
					"val_accuracy": get_accuracy(eval_data),
					"num_parameters": num_parameters,
				}
			)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	fieldnames = [
		"ensemble_size",
		"ensemble_dir",
		"filters",
		"kernels",
		"strides",
		"model_config",
		"logit_total_bits",
		"activation_total_bits",
		"val_loss",
		"val_accuracy",
		"num_parameters",
	]

	with args.output.open("w", newline="", encoding="utf-8") as csv_file:
		writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
	main()
