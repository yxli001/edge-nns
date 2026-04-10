import os
import csv
import yaml
import glob

NUM_CLASSES = 3
INPUT_SHAPE = 13


def calc_num_params(dense_widths):
    params = 0
    prev_size = INPUT_SHAPE

    for width in dense_widths:
        params += prev_size * width + width
        params += 4 * width
        prev_size = width

    params += prev_size * NUM_CLASSES + NUM_CLASSES

    return params


def parse_accuracy_file(filepath):
    result = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('Test loss:'):
                result['test_loss'] = float(line.split(':')[1].strip())
            elif line.startswith('Test accuracy:'):
                result['test_accuracy'] = float(line.split(':')[1].strip())
    return result


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_configs_dir = os.path.join(script_dir, 'model_configs')

    rows = []

    for model_dir in glob.glob(os.path.join(model_configs_dir, '*')):
        if not os.path.isdir(model_dir):
            continue

        model_name = os.path.basename(model_dir)

        accuracy_file = os.path.join(model_dir, 'accuracy.txt')
        config_file = os.path.join(model_dir, 'config.yaml')

        if not os.path.exists(accuracy_file):
            print(f"Warning: No accuracy.txt in {model_dir}")
            continue

        if not os.path.exists(config_file):
            print(f"Warning: No config.yaml in {model_dir}")
            continue

        eval_data = parse_accuracy_file(accuracy_file)

        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)

        model_cfg = config['model']
        dense_widths = model_cfg.get('dense_widths', [58])
        num_params = calc_num_params(dense_widths)

        row = {
            'model_name': model_name,
            'num_params': num_params,
            'test_accuracy': eval_data.get('test_accuracy', ''),
            'test_loss': eval_data.get('test_loss', ''),
            'activation_bit_width': model_cfg.get('activation_total_bits', ''),
            'logit_bit_width': model_cfg.get('logit_total_bits', ''),
        }
        rows.append(row)

    rows.sort(key=lambda x: x['model_name'])

    output_path = os.path.join(script_dir, 'results.csv')
    fieldnames = ['model_name', 'num_params', 'test_accuracy', 'test_loss',
                  'activation_bit_width', 'logit_bit_width']

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == '__main__':
    main()
