import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

def visualize_results(json_path=None):
    """
    Load model training results from JSON and generate comparison visualizations.
    """
    # 1. Locate the results file
    if json_path is None:
        # Find the most recent results_v2_*.json file if no path is provided
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
        files = glob(os.path.join(results_dir, 'results_v2_*.json'))
        if not files:
            print(f"[ERR] No results files found in {results_dir}")
            return
        json_path = max(files, key=os.path.getctime) # Get newest file
    
    if not os.path.exists(json_path):
        print(f"[ERR] File not found: {json_path}")
        return

    print(f"[LOAD] Reading results from: {os.path.basename(json_path)}")
    
    # 2. Load and parse data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = []
    for model_name, metrics in data.items():
        metrics['Model'] = model_name
        rows.append(metrics)
    
    df = pd.DataFrame(rows).sort_values('accuracy', ascending=True)

    # 3. Create Visualizations
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Accuracy
    sns.barplot(data=df, x='accuracy', y='Model', palette='viridis', ax=axes[0])
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Test Accuracy')
    axes[0].set_xlim(0.8, 1.05) # Zoom in to see differences
    
    # Add labels on bars
    for i, v in enumerate(df['accuracy']):
        axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')

    # Plot 2: Weighted F1-Score
    sns.barplot(data=df, x='f1_weighted', y='Model', palette='magma', ax=axes[1])
    axes[1].set_title('Model F1-Score (Weighted) Comparison', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Weighted F1-Score')
    axes[1].set_xlim(0.8, 1.05)
    
    # Add labels on bars
    for i, v in enumerate(df['f1_weighted']):
        axes[1].text(v + 0.005, i, f'{v:.4f}', va='center', fontweight='bold')

    plt.suptitle(f'Visualizing Metrics from: {os.path.basename(json_path)}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 4. Save visualization
    vis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    output_filename = f'comparison_from_{os.path.splitext(os.path.basename(json_path))[0]}.png'
    output_path = os.path.join(vis_dir, output_filename)
    
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"[OK] Visualization saved to: {output_path}")

if __name__ == "__main__":
    # You can specify a manual path here if you want
    # Example: visualize_results(r'D:\...\results_v2_20260324_134148.json')
    visualize_results()
