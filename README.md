# CLIP-SVD: Singular Value Few-shot Adaptation of Vision-Language Models

## Introduction

This repository contains the replication and improvement of **CLIP-SVD**, a state-of-the-art method for few-shot adaptation of Vision-Language Models (VLMs) to biomedical image classification tasks. The method achieves parameter-efficient adaptation by fine-tuning only **0.04%** of the model's parameters through Singular Value Decomposition (SVD).

The key features of this implementation include:

- **SVD-Based Adaptation**: Fine-tunes only singular values of weight matrices while freezing singular vectors
- **Biomedical Domain Support**: Tested on 10 diverse biomedical datasets (MRI, X-Ray, Histopathology, etc.)
- **Hyperparameter Optimization**: Includes learning rate tuning experiments showing improvements over baseline
- **Full Replication**: Successfully replicates paper results within ±0.4% accuracy

## Dependencies

Ensure the following packages are installed. You can install them directly using the provided `requirements.txt` file.

- Python 3.10
- PyTorch 2.0.1
- torchvision 0.15.2
- open-clip-torch==2.23.0
- transformers==4.35.2
- numpy, pandas, scikit-learn
- Other dependencies listed in `requirements.txt`

To install dependencies, run:

```bash
# Create conda environment
conda create -n clip_svd python=3.10 -y
conda activate clip_svd

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

## Steps to Run the Code

### 1. Clone the Repository

```bash
git clone https://github.com/shashank0302/cse589-project.git
cd cse589-project
```

### 2. Prepare the Dataset

The project uses 10 biomedical datasets from the CLIP-SVD benchmark. Download datasets from [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/tree/main) or use the download script:

```bash
bash scripts/download_all_biomedical.sh
```

Ensure the dataset follows the structure below:

```
data/
├── BTMRI/
│   ├── BTMRI/
│   └── split_BTMRI.json
├── Kvasir/
│   ├── Kvasir/
│   └── split_Kvasir.json
├── ... (other datasets)
```

### 3. Run the Code

#### Replication on All Datasets

Run full replication on all 10 biomedical datasets:

```bash
bash scripts/replicate_all_biomedical.sh
```

#### Single Dataset Example

Run on a specific dataset with specified shots:

```bash
bash scripts/fewshot.sh kvasir 16 outputs_test
```

#### Hyperparameter Tuning

Test different learning rates on all datasets:

```bash
bash scripts/hyperparameter_tuning_all_v2.sh

# Compile results
python scripts/compile_hyperparam_results.py
```

### Command Line Arguments

```bash
python main.py --root_path data --dataset <dataset_name> --shots <K> \
               --output_dir <output_dir> --config configs/few_shot/<dataset>.yaml
```

- `--dataset`: Dataset name (e.g., kvasir, btmri, covid)
- `--shots`: Number of shots per class (1, 2, 4, 8, 16)
- `--output_dir`: Directory to save results
- `--config`: Path to dataset config file
- `--tasks`: Number of random seeds (default: 3)

### 4. Monitor Training

Logs during training will display:
- Training accuracy and loss for each epoch
- Final test accuracy
- Results saved to CSV files in the output directory

Results are saved to:
- `outputs_fewshot/<dataset>.csv`: Individual dataset results
- `outputs_fewshot/biomedical_fewshot_results.csv`: All results
- `outputs_hyperparams/hyperparameter_summary.csv`: Hyperparameter tuning results

## Results

### Replication Results

Our replication successfully matches the paper's results:

| Shots (K) | Paper | Replicated | Difference |
|-----------|-------|------------|------------|
| 1 | 56.35% | 55.95% | -0.40% |
| 2 | 62.63% | 62.41% | -0.22% |
| 4 | 68.02% | 67.67% | -0.35% |
| 8 | 73.26% | 73.33% | +0.07% |
| 16 | 76.46% | 76.81% | +0.35% |

### Hyperparameter Tuning Results

Learning rate optimization on K=16 shots:

| Learning Rate | Avg Accuracy | Change from Baseline |
|---------------|--------------|----------------------|
| 0.001 | 71.37% | -4.98% |
| 0.0045 | 75.91% | -0.44% |
| 0.005 (Baseline) | 76.35% | 0.00% |
| **0.006 (Improved)** | **76.81%** | **+0.46%** |
| 0.010 | 75.58% | -0.77% |

**Improvement:** Increasing learning rate to 0.006 achieved **+0.46%** improvement over baseline.

## Repository Structure

```
CLIP-SVD/
├── configs/              # Configuration files for each dataset
│   ├── few_shot/         # Few-shot learning configs
│   └── base2new/         # Base-to-novel configs
├── datasets/             # Dataset loaders
├── svf_utils/            # SVD implementation utilities
├── scripts/              # Training and evaluation scripts
│   ├── replicate_all_biomedical.sh
│   ├── hyperparameter_tuning_all_v2.sh
│   └── fewshot.sh
├── outputs_fewshot/      # Replication results
├── outputs_hyperparams/  # Hyperparameter tuning results
├── data/                 # Dataset directory
├── main.py               # Main training script
├── train.py              # Training functions
└── requirements.txt      # Dependencies
```



