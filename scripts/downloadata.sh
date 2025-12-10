#!/bin/bash

# Download script for all 10 biomedical datasets
# Downloads from HuggingFace (faster and more reliable)

DATA_DIR="data"
HUGGINGFACE_BASE="https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/resolve/main"

echo "=========================================="
echo "Downloading All Biomedical Datasets"
echo "=========================================="
echo ""

# Create data directory if it doesn't exist
mkdir -p ${DATA_DIR}
cd ${DATA_DIR}

# List of all 10 biomedical datasets
declare -A DATASETS=(
    # ["BUSI"]="BUSI.zip"
    # ["BTMRI"]="BTMRI.zip"
    # ["CHMNIST"]="CHMNIST.zip"
    # ["COVID_19"]="COVID_19.zip"
    # ["CTKidney"]="CTKidney.zip"
    # ["KneeXray"]="KneeXray.zip"
    # ["Kvasir"]="Kvasir.zip"
    # ["LungColon"]="LungColon.zip"
    # ["OCTMNIST"]="OCTMNIST.zip"
    # ["RETINA"]="RETINA.zip"
    ["DermaMNIST"]="DermaMNIST.zip"
)

TOTAL=${#DATASETS[@]}
CURRENT=0

for DATASET_NAME in "${!DATASETS[@]}"
do
    CURRENT=$((CURRENT + 1))
    ZIP_FILE="${DATASETS[$DATASET_NAME]}"
    DATASET_DIR="${DATASET_NAME}"
    
    echo "[${CURRENT}/${TOTAL}] Downloading ${DATASET_NAME}..."
    
    # # Check if already extracted
    # if [ -d "${DATASET_DIR}" ] && [ -f "${DATASET_DIR}/split_${DATASET_NAME}.json" ]; then
    #     echo "  ✓ ${DATASET_NAME} already exists, skipping..."
    #     continue
    # fi
    
    # Download
    echo "  Downloading from HuggingFace..."
    wget -q --show-progress "${HUGGINGFACE_BASE}/${ZIP_FILE}" -O "${ZIP_FILE}"
    
    if [ $? -eq 0 ]; then
        echo "  Extracting ${ZIP_FILE}..."
        unzip -q "${ZIP_FILE}" -d "${DATASET_DIR}_temp"
        
        # Move contents to correct location
        if [ -d "${DATASET_DIR}_temp/${DATASET_NAME}" ]; then
            mv "${DATASET_DIR}_temp/${DATASET_NAME}" "${DATASET_DIR}"
        else
            # Some datasets might extract differently
            mv "${DATASET_DIR}_temp"/* "${DATASET_DIR}" 2>/dev/null || true
        fi
        
        # Cleanup
        rm -rf "${DATASET_DIR}_temp"
        rm -f "${ZIP_FILE}"
        
        # Verify structure
        if [ -f "${DATASET_DIR}/split_${DATASET_NAME}.json" ]; then
            echo "  ✓ ${DATASET_NAME} downloaded and extracted successfully"
        else
            echo "  ⚠ ${DATASET_NAME} extracted but structure may be incorrect"
            echo "     Expected: ${DATASET_DIR}/split_${DATASET_NAME}.json"
        fi
    else
        echo "  ✗ Failed to download ${DATASET_NAME}"
        echo "     Try manual download from: ${HUGGINGFACE_BASE}/${ZIP_FILE}"
    fi
    
    echo ""
done

cd ..

echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Verify datasets:"
echo "  ls -la data/"
echo ""
echo "Each dataset should have:"
echo "  - A directory with the dataset name"
echo "  - split_<DATASET_NAME>.json file"
echo ""

