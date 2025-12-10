#!/usr/bin/env python3
"""
Compile hyperparameter tuning results into a formatted table for the report
"""

import os
import glob
import pandas as pd
import sys

def compile_results(base_dir="outputs_hyperparams"):
    """Compile all hyperparameter tuning results"""
    
    print("="*70)
    print("HYPERPARAMETER TUNING RESULTS - SECTION 3")
    print("="*70)
    print()
    
    # Find all LR directories
    lr_dirs = glob.glob(os.path.join(base_dir, "lr_*"))
    
    if not lr_dirs:
        print(f"Error: No results found in {base_dir}/")
        print("Run hyperparameter_tuning_all.sh first!")
        return
    
    results = {}
    
    for lr_dir in sorted(lr_dirs):
        lr_value = os.path.basename(lr_dir).replace("lr_", "")
        
        # Read all CSV files in this directory
        csv_files = glob.glob(os.path.join(lr_dir, "*.csv"))
        
        dataset_results = {}
        for csv_file in csv_files:
            dataset_name = os.path.basename(csv_file).replace('.csv', '')
            try:
                df = pd.read_csv(csv_file)
                # Get K=16 result
                k16_results = df[df['num_shots'] == 16]['acc'].values
                if len(k16_results) > 0:
                    dataset_results[dataset_name] = k16_results[0]
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
        
        if dataset_results:
            avg_acc = sum(dataset_results.values()) / len(dataset_results)
            results[float(lr_value)] = {
                'avg': avg_acc,
                'datasets': dataset_results
            }
    
    if not results:
        print("Error: No valid results found!")
        return
    
    # Find baseline (LR=0.005)
    baseline_acc = results.get(0.005, {}).get('avg', 0)
    
    # Print summary table
    print("Table 1: Learning Rate Hyperparameter Tuning (K=16 shots)")
    print("-"*70)
    print(f"{'Learning Rate':<15} {'Avg Accuracy':<20} {'Change from Baseline':<20}")
    print("-"*70)
    
    best_lr = None
    best_acc = 0
    
    for lr in sorted(results.keys()):
        acc = results[lr]['avg']
        diff = acc - baseline_acc
        diff_str = f"{diff:+.2f}%" if diff != 0 else "0.00%"
        
        is_baseline = " (baseline)" if lr == 0.005 else ""
        
        print(f"{lr:<15} {acc:.2f}%{is_baseline:<15} {diff_str:<20}")
        
        if acc > best_acc:
            best_acc = acc
            best_lr = lr
    
    print("-"*70)
    print(f"\nBaseline (paper default): LR=0.005, Accuracy={baseline_acc:.2f}%")
    print(f"Best result: LR={best_lr}, Accuracy={best_acc:.2f}%")
    
    improvement = best_acc - baseline_acc
    if improvement > 0:
        print(f"Improvement: +{improvement:.2f}%")
    elif improvement < 0:
        print(f"Change: {improvement:.2f}% (baseline is best)")
    else:
        print("No improvement (baseline is optimal)")
    
    print()
    print("="*70)
    
    # Save to CSV for easy import to report
    output_file = os.path.join(base_dir, "hyperparameter_summary.csv")
    summary_data = []
    for lr in sorted(results.keys()):
        acc = results[lr]['avg']
        diff = acc - baseline_acc
        summary_data.append({
            'learning_rate': lr,
            'avg_accuracy': round(acc, 2),
            'change_from_baseline': round(diff, 2)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"Summary saved to: {output_file}")
    print()
    
    # Also print detailed per-dataset results for best LR
    if best_lr != 0.005:
        print("\nDetailed Results for Best LR ({}):".format(best_lr))
        print("-"*70)
        print(f"{'Dataset':<15} {'Baseline (0.005)':<20} {'Best LR ({})'.format(best_lr):<20} {'Change':<10}")
        print("-"*70)
        
        baseline_datasets = results[0.005]['datasets']
        best_datasets = results[best_lr]['datasets']
        
        for dataset in sorted(baseline_datasets.keys()):
            if dataset in best_datasets:
                baseline_val = baseline_datasets[dataset]
                best_val = best_datasets[dataset]
                diff = best_val - baseline_val
                diff_str = f"{diff:+.2f}%"
                
                print(f"{dataset:<15} {baseline_val:.2f}%{'':<15} {best_val:.2f}%{'':<15} {diff_str:<10}")
        print("-"*70)

if __name__ == "__main__":
    base_dir = "outputs_hyperparams"
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    
    compile_results(base_dir)

