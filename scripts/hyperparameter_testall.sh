#!/bin/bash

# Hyperparameter Tuning for Improvements Section
# Tests Learning Rate and Weight Decay combinations as described in paper

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

OUTPUT_BASE="outputs_improvements"
CONFIG_BASE="configs/few_shot"
TEMP_CONFIG_DIR="configs/improvements_temp"

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

# Learning rates to test (matching LaTeX document)
LEARNING_RATES=(
    "0.001"
    "0.0045"
    "0.005"   # baseline
    "0.006"   # our improvement
    "0.010"
)

SHOTS=16  # Use 16 shots for comparison

echo "=========================================="
echo "Hyperparameter Tuning - Improvements"
echo "=========================================="
echo "Datasets: 10 biomedical datasets"
echo "Shots: ${SHOTS}"
echo "Learning rates: ${LEARNING_RATES[@]}"
echo "Weight decay: Testing 0.0 (baseline) and 0.02 (improved)"
echo "Seeds: 1 (for speed)"
echo ""
echo "Estimated time: 3-5 hours"
echo "=========================================="
echo ""

# Create temp config directory
mkdir -p ${TEMP_CONFIG_DIR}

# Test each learning rate with baseline weight decay (0.0)
for LR in "${LEARNING_RATES[@]}"
do
    OUTPUT_DIR="${OUTPUT_BASE}/lr_${LR}_wd_0.0"
    mkdir -p ${OUTPUT_DIR}
    
    echo ""
    echo "=========================================="
    echo "Testing LR=${LR} with Weight Decay=0.0"
    echo "=========================================="
    
    # Create modified config files for this learning rate
    for DATASET in "${BIOMEDICAL_DATASETS[@]}"
    do
        ORIG_CONFIG="${CONFIG_BASE}/${DATASET}.yaml"
        TEMP_CONFIG="${TEMP_CONFIG_DIR}/${DATASET}_lr${LR}_wd0.0.yaml"
        
        # Copy and modify the learning rate (weight decay stays 0.0 from config)
        sed "s/^lr: .*/lr: ${LR}/" ${ORIG_CONFIG} > ${TEMP_CONFIG}
    done
    
    # Run on all datasets
    CURRENT=0
    TOTAL=${#BIOMEDICAL_DATASETS[@]}
    
    for DATASET in "${BIOMEDICAL_DATASETS[@]}"
    do
        CURRENT=$((CURRENT + 1))
        echo "[${CURRENT}/${TOTAL}] Running ${DATASET} with LR=${LR}, WD=0.0..."
        
        TEMP_CONFIG="${TEMP_CONFIG_DIR}/${DATASET}_lr${LR}_wd0.0.yaml"
        
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
    
    echo "✓ Completed all datasets for LR=${LR}, WD=0.0"
done

# Test improved combination: LR=0.006 with Weight Decay=0.02
echo ""
echo "=========================================="
echo "Testing Improved Combination: LR=0.006, Weight Decay=0.02"
echo "=========================================="

LR="0.006"
WD="0.02"
OUTPUT_DIR="${OUTPUT_BASE}/lr_${LR}_wd_${WD}"
mkdir -p ${OUTPUT_DIR}

for DATASET in "${BIOMEDICAL_DATASETS[@]}"
do
    ORIG_CONFIG="${CONFIG_BASE}/${DATASET}.yaml"
    TEMP_CONFIG="${TEMP_CONFIG_DIR}/${DATASET}_lr${LR}_wd${WD}.yaml"
    
    # Modify both learning rate and weight decay
    sed -e "s/^lr: .*/lr: ${LR}/" -e "s/^weight_decay: .*/weight_decay: ${WD}/" ${ORIG_CONFIG} > ${TEMP_CONFIG}
done

CURRENT=0
TOTAL=${#BIOMEDICAL_DATASETS[@]}

for DATASET in "${BIOMEDICAL_DATASETS[@]}"
do
    CURRENT=$((CURRENT + 1))
    echo "[${CURRENT}/${TOTAL}] Running ${DATASET} with LR=${LR}, WD=${WD}..."
    
    TEMP_CONFIG="${TEMP_CONFIG_DIR}/${DATASET}_lr${LR}_wd${WD}.yaml"
    
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
echo "=========================================="
echo "All Hyperparameter Experiments Complete!"
echo "=========================================="
echo ""

# Generate final comparison report
python -c "
import os
import glob
import pandas as pd

print('\n' + '='*70)
print('HYPERPARAMETER TUNING RESULTS (K=16)')
print('='*70)
print(f'{'Learning Rate':<15} {'Weight Decay':<15} {'Avg Accuracy':<15} {'Change':<15}')
print('-'*70)

results = {}
baseline_acc = 0

# Read results for each LR with WD=0.0
for lr in ['0.001', '0.0045', '0.005', '0.006', '0.010']:
    output_dir = f'${OUTPUT_BASE}/lr_{lr}_wd_0.0'
    csv_files = glob.glob(os.path.join(output_dir, '*.csv'))
    
    if csv_files:
        all_results = {}
        for csv_file in csv_files:
            dataset_name = os.path.basename(csv_file).replace('.csv', '')
            df = pd.read_csv(csv_file)
            result = df[df['num_shots'] == 16]['acc'].values
            if len(result) > 0:
                all_results[dataset_name] = result[0]
        
        if all_results:
            avg_result = sum(all_results.values()) / len(all_results)
            results[(float(lr), 0.0)] = avg_result
            if float(lr) == 0.005:
                baseline_acc = avg_result

# Read improved combination (LR=0.006, WD=0.02)
output_dir = '${OUTPUT_BASE}/lr_0.006_wd_0.02'
csv_files = glob.glob(os.path.join(output_dir, '*.csv'))

if csv_files:
    all_results = {}
    for csv_file in csv_files:
        dataset_name = os.path.basename(csv_file).replace('.csv', '')
        df = pd.read_csv(csv_file)
        result = df[df['num_shots'] == 16]['acc'].values
        if len(result) > 0:
            all_results[dataset_name] = result[0]
    
    if all_results:
        avg_result = sum(all_results.values()) / len(all_results)
        results[(0.006, 0.02)] = avg_result

# Print results
for (lr, wd) in sorted(results.keys(), key=lambda x: (x[0], x[1])):
    acc = results[(lr, wd)]
    diff = acc - baseline_acc
    diff_str = f'{diff:+.2f}%' if diff != 0 else '0.00%'
    baseline_str = ' (baseline)' if lr == 0.005 and wd == 0.0 else ''
    improved_str = ' (improved)' if lr == 0.006 and wd == 0.02 else ''
    
    print(f'{lr:<15} {wd:<15} {acc:.2f}%{baseline_str}{improved_str:<10} {diff_str:<15}')

if results:
    print('='*70)
    best_combo, best_acc = max(results.items(), key=lambda x: x[1])
    print(f'\nBest combination: LR={best_combo[0]}, WD={best_combo[1]}')
    print(f'Best accuracy: {best_acc:.2f}%')
    if best_acc > baseline_acc:
        print(f'Improvement over baseline: +{best_acc - baseline_acc:.2f}%')
print()
"

# Cleanup temp configs
echo "Cleaning up temporary config files..."
rm -rf ${TEMP_CONFIG_DIR}

echo ""
echo "Results saved in: ${OUTPUT_BASE}/"
echo "Run 'python scripts/compile_improvements_results.py' to get formatted tables"

