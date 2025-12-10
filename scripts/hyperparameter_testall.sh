#!/bin/bash

# Comprehensive Hyperparameter Tuning for All 10 Biomedical Datasets
# Strategy: Create temporary config files with different learning rates

# Activate conda environment (for nohup/background execution)
export PATH="/scratch/sba6069/miniconda3/envs/cse589/bin:$PATH"

# Verify python is available
if ! command -v python &> /dev/null; then
    echo "Error: python not found. Trying to activate conda..."
    source /scratch/sba6069/miniconda3/etc/profile.d/conda.sh
    conda activate cse589
    if ! command -v python &> /dev/null; then
        echo "Error: Still cannot find python. Exiting."
        exit 1
    fi
fi

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

OUTPUT_BASE="outputs_hyperparams"
CONFIG_BASE="configs/few_shot"
TEMP_CONFIG_DIR="configs/hyperparam_temp"

# All 10 biomedical datasets
BIOMEDICAL_DATASETS=(
    "busi"
    "btmri"
    "chmnist"
    "covid"
    "ctkidney"
    "kneexray"
    "kvasir"
    "lungcolon"
    "octmnist"
    "retina"
)

# Test these learning rates (default is 0.005)
LEARNING_RATES=(
    "0.001"
    "0.003"
    "0.005"  # baseline
    "0.007"
    "0.010"
)

SHOTS=16  # Use 16 shots for comparison

echo "=========================================="
echo "Hyperparameter Tuning - All Datasets"
echo "=========================================="
echo "Datasets: 10 biomedical datasets"
echo "Shots: ${SHOTS}"
echo "Learning rates to test: ${LEARNING_RATES[@]}"
echo "Seeds: 1 (for speed)"
echo ""
echo "Estimated time: 3-5 hours"
echo "=========================================="
echo ""

# Create temp config directory
mkdir -p ${TEMP_CONFIG_DIR}

# Test each learning rate
for LR in "${LEARNING_RATES[@]}"
do
    OUTPUT_DIR="${OUTPUT_BASE}/lr_${LR}"
    mkdir -p ${OUTPUT_DIR}
    
    echo ""
    echo "=========================================="
    echo "Testing Learning Rate: ${LR}"
    echo "=========================================="
    
    # Create modified config files for this learning rate
    for DATASET in "${BIOMEDICAL_DATASETS[@]}"
    do
        ORIG_CONFIG="${CONFIG_BASE}/${DATASET}.yaml"
        TEMP_CONFIG="${TEMP_CONFIG_DIR}/${DATASET}_lr${LR}.yaml"
        
        # Copy and modify the learning rate
        sed "s/^lr: .*/lr: ${LR}/" ${ORIG_CONFIG} > ${TEMP_CONFIG}
    done
    
    # Run on all datasets with modified configs
    CURRENT=0
    TOTAL=${#BIOMEDICAL_DATASETS[@]}
    
    for DATASET in "${BIOMEDICAL_DATASETS[@]}"
    do
        CURRENT=$((CURRENT + 1))
        echo ""
        echo "[${CURRENT}/${TOTAL}] Running ${DATASET} with LR=${LR}..."
        
        TEMP_CONFIG="${TEMP_CONFIG_DIR}/${DATASET}_lr${LR}.yaml"
        
        python main.py --root_path data --dataset ${DATASET} \
                        --shots ${SHOTS} \
                        --output_dir ${OUTPUT_DIR} \
                        --config ${TEMP_CONFIG} \
                        --tasks 1
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Completed ${DATASET}"
        else
            echo "  ✗ Failed for ${DATASET}"
        fi
    done
    
    echo ""
    echo "✓ Completed all datasets for LR=${LR}"
    
    # Calculate average for this learning rate
    echo "Calculating average results for LR=${LR}..."
    python -c "
import pandas as pd
import glob
import os

output_dir = '${OUTPUT_DIR}'
csv_files = glob.glob(os.path.join(output_dir, '*.csv'))

if csv_files:
    all_results = {}
    for csv_file in csv_files:
        dataset_name = os.path.basename(csv_file).replace('.csv', '')
        df = pd.read_csv(csv_file)
        # Get the result for K=16
        result = df[df['num_shots'] == 16]['acc'].values
        if len(result) > 0:
            all_results[dataset_name] = result[0]
    
    if all_results:
        avg_result = sum(all_results.values()) / len(all_results)
        print(f'LR={LR} Average (K=16): {avg_result:.2f}%')
        
        # Save summary
        with open(f'{output_dir}/summary.txt', 'w') as f:
            f.write(f'Learning Rate: ${LR}\n')
            f.write(f'Average Accuracy (K=16): {avg_result:.2f}%\n')
            f.write(f'\nIndividual Results:\n')
            for dataset, acc in sorted(all_results.items()):
                f.write(f'  {dataset}: {acc:.2f}%\n')
else:
    print('No results found!')
"
done

echo ""
echo "=========================================="
echo "All Hyperparameter Experiments Complete!"
echo "=========================================="
echo ""
echo "Generating final comparison report..."

# Generate final comparison report
python -c "
import os

print('\n' + '='*60)
print('HYPERPARAMETER TUNING RESULTS (K=16)')
print('='*60)
print(f'{'Learning Rate':<15} {'Avg Accuracy':<15} {'vs Baseline':<15}')
print('-'*60)

learning_rates = [${LEARNING_RATES[@]//,/}]
results = {}

for lr in learning_rates:
    lr_str = str(lr).strip(\"'\")
    summary_file = f'${OUTPUT_BASE}/lr_{lr_str}/summary.txt'
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'Average Accuracy' in line:
                    acc_str = line.split(':')[1].strip().replace('%', '')
                    acc = float(acc_str)
                    results[float(lr_str)] = acc
                    break

# Baseline is LR=0.005
baseline = results.get(0.005, 0)

for lr in sorted(results.keys()):
    acc = results[lr]
    diff = acc - baseline
    diff_str = f'{diff:+.2f}%' if diff != 0 else '0.00%'
    baseline_str = ' (baseline)' if float(lr) == 0.005 else ''
    
    print(f'{lr:<15} {acc:.2f}%{baseline_str:<10} {diff_str:<15}')

if results:
    print('='*60)
    best_lr, best_acc = max(results.items(), key=lambda x: x[1])
    print(f'\nBest learning rate: {best_lr}')
    print(f'Best accuracy: {best_acc:.2f}%')
    if best_acc > baseline:
        print(f'Improvement over baseline: +{best_acc - baseline:.2f}%')
print()
"

# Cleanup temp configs
echo "Cleaning up temporary config files..."
rm -rf ${TEMP_CONFIG_DIR}

echo ""
echo "Results saved in: ${OUTPUT_BASE}/"
echo "Each LR has its own directory with:"
echo "  - Individual dataset CSVs"
echo "  - summary.txt with averaged results"
echo ""
echo "Run 'python scripts/compile_hyperparam_results.py' to get formatted tables"

