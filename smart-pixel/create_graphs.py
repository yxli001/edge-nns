import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def find_pareto_frontier(df, x_col='num_params', y_col='test_accuracy'):
    """Find Pareto optimal points."""
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

def main():
    
    # load data
    df = pd.read_csv('results.csv')
    print(f"Loaded {len(df)} models")
    
    # graph folder
    os.makedirs('graphs', exist_ok=True)
    
    # Find Pareto optimal models
    pareto_df = find_pareto_frontier(df, 'num_params', 'test_accuracy')
    pareto_df = pareto_df.sort_values('num_params')
    
    
    # plot
    plt.figure(figsize=(14, 8))
    
    # data ranges
    x_min, x_max = df['num_params'].min(), df['num_params'].max()
    y_min, y_max = df['test_accuracy'].min(), df['test_accuracy'].max()
    
    # Plot all models
    unique_bits = sorted(df['activation_bit_width'].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(unique_bits)))
    
    for i, bit_val in enumerate(unique_bits):
        subset = df[df['activation_bit_width'] == bit_val]
        plt.scatter(subset['num_params'], subset['test_accuracy'],
                   s=60, color=colors[i], alpha=0.7, edgecolors='black', linewidth=0.5,
                   label=f'{bit_val}-bit')
    
    # pareto frontier line
    plt.plot(pareto_df['num_params'], pareto_df['test_accuracy'],
            'r--', linewidth=2, alpha=0.8, label='Pareto Frontier')
    
    # Style
    plt.title('Model Accuracy vs Size')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Accuracy')
    
    # scale
    plt.xscale('log')
    tick_values = [1e2, 1e3, 1e4, 1e5, 1e6]
    tick_labels = ['100', '1,000', '10,000', '100,000', '1,000,000']
    plt.xticks(tick_values, tick_labels)
    
    # Y axis range
    plt.ylim(y_min - 0.01, y_max + 0.005)
    
    plt.legend(title='Activation Bits')
    plt.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Save 
    output_path = 'graphs/graph_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Graph saved: {output_path}")
    
    # top 4 pareto optimal
    pareto_sorted = pareto_df.sort_values('test_accuracy', ascending=False)
    for idx, (_, model) in enumerate(pareto_sorted.iterrows()):
        parts = model['model_name'].split('_')
        arch = '-'.join(parts[3:6]) if len(parts) >= 6 else model['dense_widths']
        print(f"{idx+1}. {arch:12s} | Acc: {model['test_accuracy']:.3f} | "
              f"Params: {model['num_params']:6,} | "
              f"Bits: {model['activation_bit_width']}/{model['logit_bit_width']}")
    
    
    # Print best model
    best = df.loc[df['test_accuracy'].idxmax()]
    print(f"\nBest Model: {best['model_name']}")
    print(f"Accuracy: {best['test_accuracy']:.3f}")
    print(f"Test loss: {best['test_loss']:.3f}")
    print(f"Parameters: {best['num_params']:,}")

if __name__ == "__main__":
    main()