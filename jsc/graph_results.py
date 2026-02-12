import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml


def find_pareto_frontier(df, x_col='num_params', y_col='test_accuracy'):
    points = df[[x_col, y_col]].values
    pareto_points = []

    for i, point in enumerate(points):
        is_pareto = True
        for j, other_point in enumerate(points):
            if i != j:
                dominates = (other_point[0] <= point[0] and other_point[1] >= point[1]) and \
                           (other_point[0] < point[0] or other_point[1] > point[1])
                if dominates:
                    is_pareto = False
                    break
        if is_pareto:
            pareto_points.append(i)

    return df.iloc[pareto_points]


def estimate_jsc_params(config):
    try:
        dense_widths = config['model'].get('dense_widths', [])
        input_shape = config['data']['input_shape'][0]
        num_classes = config['data']['num_classes']
        
        params = 0
        prev_width = input_shape
        for width in dense_widths:
            params += prev_width * width + width
            prev_width = width
        params += prev_width * num_classes + num_classes
        
        return params
    except Exception as e:
        print(f"Warning: Could not estimate parameters: {e}")
        return None


def extract_additional_info(model_dir="../jsc/models"):
    results = []
    for model_name in sorted(os.listdir(model_dir)):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isdir(model_path):
            continue
            
        config_path = os.path.join(model_path, "config.yml")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            model_cfg = config.get("model", {})
            quant_cfg = config.get("quantization", {})
            data_cfg = config.get("data", {})
            
            dense_widths = model_cfg.get("dense_widths", [])
            widths_str = "-".join(map(str, dense_widths)) if dense_widths else ""
            
            num_params = estimate_jsc_params(config)
            
            results.append({
                "model_name": model_name,
                "dense_widths": widths_str,
                "num_params": num_params if num_params is not None else "",
                "activation_bit_width": quant_cfg.get("activation_total_bits", ""),
                "logit_bit_width": quant_cfg.get("logit_total_bits", ""),
                "activation_int_bits": quant_cfg.get("activation_int_bits", ""),
                "logit_int_bits": quant_cfg.get("logit_int_bits", ""),
                "input_shape": data_cfg.get("input_shape", [""])[0] if data_cfg.get("input_shape") else "",
                "num_classes": data_cfg.get("num_classes", ""),
            })
    return pd.DataFrame(results)


def main():
    df_results = pd.read_csv('jsc_results.csv')
    df_configs = extract_additional_info()
    
    df = pd.merge(df_results, df_configs, on='model_name')
    print(f"Loaded {len(df)} JSC models with config data")

    os.makedirs('graphs', exist_ok=True)

    pareto_df = find_pareto_frontier(df, 'num_params', 'test_accuracy')
    pareto_df = pareto_df.sort_values('num_params')

    plt.figure(figsize=(14, 8))

    x_min, x_max = df['num_params'].min(), df['num_params'].max()
    y_min, y_max = df['test_accuracy'].min(), df['test_accuracy'].max()

    unique_bits = sorted(df['activation_bit_width'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(unique_bits)))

    for i, bit_val in enumerate(unique_bits):
        subset = df[df['activation_bit_width'] == bit_val]
        plt.scatter(subset['num_params'], subset['test_accuracy'],
                   s=60, color=colors[i], alpha=0.7, edgecolors='black', linewidth=0.5,
                   label=f'{bit_val}-bit')

    plt.plot(pareto_df['num_params'], pareto_df['test_accuracy'],
            'r--', linewidth=2, alpha=0.8, label='Pareto Frontier')

    plt.title('JSC: Model Accuracy vs Size')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Accuracy')

    if df['num_params'].max() > 10000:
        plt.xscale('log')
        tick_values = [1e3, 1e4, 1e5]
        tick_labels = ['1,000', '10,000', '100,000']
        plt.xticks(tick_values, tick_labels)

    plt.ylim(y_min - 0.01, y_max + 0.005)

    plt.legend(title='Activation Bits')
    plt.grid(True, alpha=0.3, which='both')
    plt.tight_layout()

    output_path = 'graphs/accuracy_vs_params.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Graph saved: {output_path}")

    print(f"\nPareto optimal models ({len(pareto_df)}):")
    pareto_sorted = pareto_df.sort_values('test_accuracy', ascending=False)
    for idx, (_, model) in enumerate(pareto_sorted.iterrows()):
        print(f"  {idx+1}. {model['model_name'][:12]:12s} | widths: {model['dense_widths']:8s} | "
              f"Acc: {model['test_accuracy']:.4f} | Params: {model['num_params']:6,} | "
              f"Bits: {model['activation_bit_width']}/{model['logit_bit_width']}")

    best = df.loc[df['test_accuracy'].idxmax()]
    print(f"\nBest Model: {best['model_name']}")
    print(f"  Accuracy: {best['test_accuracy']:.4f}")
    print(f"  Test loss: {best['test_loss']:.4f}")
    print(f"  Parameters: {best['num_params']:,}")
    print(f"  Dense widths: {best['dense_widths']}")
    print(f"  Bits: {best['activation_bit_width']}/{best['logit_bit_width']}")
    print(f"  Input shape: {best['input_shape']}, Classes: {best['num_classes']}")


if __name__ == "__main__":
    main()