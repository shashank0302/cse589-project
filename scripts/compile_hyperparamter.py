#!/usr/bin/env python3
"""
Compile hyperparameter tuning results for improvements section
Matches the LaTeX document format
"""

import os
import glob
import pandas as pd
import sys

def compile_results(base_dir="outputs_improvements"):
    """Compile all hyperparameter tuning results"""
    
    print("="*70)
    print("HYPERPARAMETER TUNING RESULTS - IMPROVEMENTS SECTION")
    print("="*70)
    print()
    
    results = {}
    baseline_acc = 0
    
    # Read results for each LR with WD=0.0
    learning_rates = ['0.001', '0.0045', '0.005', '0.006', '0.010']
    
    for lr_str in learning_rates:
        output_dir = os.path.join(base_dir, f"lr_{lr_str}_wd_0.0")
        csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
        
        if csv_files:
            dataset_results = {}
            for csv_file in csv_files:
                dataset_name = os.path.basename(csv_file).replace('.csv', '')
                try:
                    df = pd.read_csv(csv_file)
                    k16_results = df[df['num_shots'] == 16]['acc'].values
                    if len(k16_results) > 0:
                        dataset_results[dataset_name] = k16_results[0]
                except Exception as e:
                    print(f"Warning: Could not read {csv_file}: {e}")
            
            if dataset_results:
                avg_acc = sum(dataset_results.values()) / len(dataset_results)
                lr_float = float(lr_str)
                results[(lr_float, 0.0)] = {
                    'avg': avg_acc,
                    'datasets': dataset_results
                }
                
                if lr_float == 0.005:
                    baseline_acc = avg_acc
    
    # Read improved combination (LR=0.006, WD=0.02)
    output_dir = os.path.join(base_dir, "lr_0.006_wd_0.02")
    csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
    
    if csv_files:
        dataset_results = {}
        for csv_file in csv_files:
            dataset_name = os.path.basename(csv_file).replace('.csv', '')
            try:
                df = pd.read_csv(csv_file)
                k16_results = df[df['num_shots'] == 16]['acc'].values
                if len(k16_results) > 0:
                    dataset_results[dataset_name] = k16_results[0]
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
        
        if dataset_results:
            avg_acc = sum(dataset_results.values()) / len(dataset_results)
            results[(0.006, 0.02)] = {
                'avg': avg_acc,
                'datasets': dataset_results
            }
    
    if not results:
        print("Error: No valid results found!")
        print(f"Check if results exist in {base_dir}/")
        return
    
    # Print summary table (matching LaTeX format)
    print("Table: Hyperparameter Tuning Results (Average Accuracy on 10 Biomedical Datasets) for k=16")
    print("-"*70)
    print(f"{'Learning Rate':<20} {'Avg Acc (%) [k=16]':<25} {'Change':<15}")
    print("-"*70)
    
    best_lr = None
    best_acc = 0
    
    # Sort by learning rate
    for (lr, wd) in sorted(results.keys(), key=lambda x: x[0]):
        acc = results[(lr, wd)]['avg']
        diff = acc - baseline_acc
        diff_str = f"{diff:+.2f}" if diff != 0 else "0.00"
        
        lr_str = f"{lr}"
        if lr == 0.005 and wd == 0.0:
            lr_str += " (Baseline)"
        elif lr == 0.006 and wd == 0.02:
            lr_str += " (Ours)"
        
        print(f"{lr_str:<20} {acc:.2f}%{'':<20} {diff_str:<15}")
        
        if acc > best_acc:
            best_acc = acc
            best_lr = (lr, wd)
    
    print("-"*70)
    print(f"\nBaseline (paper default): LR=0.005, WD=0.0, Accuracy={baseline_acc:.2f}%")
    
    if best_lr:
        best_lr_val, best_wd_val = best_lr
        print(f"Best result: LR={best_lr_val}, WD={best_wd_val}, Accuracy={best_acc:.2f}%")
        
        improvement = best_acc - baseline_acc
        if improvement > 0:
            print(f"Improvement: +{improvement:.2f}%")
        elif improvement < 0:
            print(f"Change: {improvement:.2f}% (baseline is best)")
        else:
            print("No improvement (baseline is optimal)")
    
    print()
    print("="*70)
    
    # Save to CSV
    output_file = os.path.join(base_dir, "improvements_summary.csv")
    summary_data = []
    for (lr, wd) in sorted(results.keys(), key=lambda x: x[0]):
        acc = results[(lr, wd)]['avg']
        diff = acc - baseline_acc
        summary_data.append({
            'learning_rate': lr,
            'weight_decay': wd,
            'avg_accuracy': round(acc, 2),
            'change_from_baseline': round(diff, 2)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_file, index=False)
    print(f"Summary saved to: {output_file}")
    print()

if __name__ == "__main__":
    base_dir = "outputs_improvements"
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    
    compile_results(base_dir)

