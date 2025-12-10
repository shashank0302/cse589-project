#!/bin/bash

# Full replication script for ALL biomedical datasets
# Runs all 10 datasets × 5 shot values (1, 2, 4, 8, 16) × 3 seeds each
# This will give you averaged results comparable to the paper

OUTPUT_DIR="outputs_fewshot"

# All 10 biomedical datasets (matching paper)
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

echo "=========================================="
echo "Full Biomedical Dataset Replication"
echo "=========================================="
echo "This will run:"
echo "  - 10 datasets"
echo "  - 5 shot values each (1, 2, 4, 8, 16)"
echo "  - 3 seeds per run"
echo "  - Total: ~150 training runs (10 datasets × 5 shots × 3 seeds)"
echo ""
echo "Estimated time: 8-15 hours (depending on GPU)"
echo "=========================================="
echo ""

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Track progress
TOTAL_DATASETS=${#BIOMEDICAL_DATASETS[@]}
CURRENT=0

# Run for each dataset
for DATASET in "${BIOMEDICAL_DATASETS[@]}"
do
    CURRENT=$((CURRENT + 1))
    echo ""
    echo "=========================================="
    echo "[${CURRENT}/${TOTAL_DATASETS}] Processing: ${DATASET}"
    echo "=========================================="
    
    # Run for each shot value
    for SHOTS in 1 2 4 8 16
    do
        echo "  Running ${DATASET} with ${SHOTS} shots..."
        
        python main.py --root_path data --dataset ${DATASET} --tasks 3 --shots ${SHOTS} \
                        --output_dir ${OUTPUT_DIR} --config configs/few_shot/${DATASET}.yaml
        
        if [ $? -eq 0 ]; then
            echo "  ✓ Completed ${DATASET} - ${SHOTS} shots"
        else
            echo "  ✗ Failed for ${DATASET} - ${SHOTS} shots"
            echo "  Continuing with next dataset..."
        fi
    done
    
    echo "✓ Completed all shots for ${DATASET}"
done

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo ""
echo "Generating summary results..."
python read_fewshot_results.py --dataset_type biomedical --output_name biomedical_fewshot_results.csv --output_dir ${OUTPUT_DIR}

echo ""
echo "Summary saved to: ${OUTPUT_DIR}/biomedical_fewshot_results.csv"
echo ""
echo "You can now compare your averaged results with the paper's results!"

