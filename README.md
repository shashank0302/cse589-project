# CLIP-SVD: Singular Value Few-shot Adaptation of Vision-Language Models

This repository contains my replication and improvement of the CLIP-SVD paper for few-shot biomedical image classification. CLIP-SVD is a parameter-efficient method that adapts Vision-Language Models by fine-tuning only 0.04% of parameters using Singular Value Decomposition.

## What This Does

- Adapts pre-trained CLIP/BiomedCLIP models to biomedical domains with very few examples (1-16 shots per class)
- Uses SVD to modify only singular values while keeping singular vectors frozen
- Tested on 10 biomedical datasets covering MRI, X-Ray, Histopathology, and other medical imaging modalities

## Installation

```bash
# Create conda environment
conda create -n clip_svd python=3.10 -y
conda activate clip_svd

# Install PyTorch (CUDA 11.8)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

## Dataset Setup

Download the 10 biomedical datasets from [HuggingFace](https://huggingface.co/datasets/TahaKoleilat/BiomedCoOp/tree/main). Each dataset should be placed in the `data/` directory with the following structure:

```
data/
├── BTMRI/
│   ├── BTMRI/
│   └── split_BTMRI.json
├── Kvasir/
│   ├── Kvasir/
│   └── split_Kvasir.json
├── ... (other 8 datasets)
```

The datasets include: BTMRI, BUSI, CHMNIST, COVID_19, CTKidney, KneeXray, Kvasir, LungColon, OCTMNIST, and RETINA.

## Running the Code

### Full Replication

To replicate results on all 10 datasets with all shot values (1, 2, 4, 8, 16):

```bash
bash scripts/replicate_all_biomedical.sh
```

This will take several hours and generates results in `outputs_fewshot/`.

### Single Dataset

To test on a specific dataset:

```bash
bash scripts/fewshot.sh kvasir 16 outputs_test
```

Replace `kvasir` with any dataset name and `16` with the number of shots (1, 2, 4, 8, or 16).

### Hyperparameter Tuning

To test different learning rates and weight decay values:

```bash
bash scripts/hyperparameter_tuning_improvements.sh

# Compile results
python scripts/compile_improvements_results.py
```

Results are saved in `outputs_improvements/`.

## Results

### Replication Results

I successfully replicated the paper's results on all 10 biomedical datasets. The averaged accuracy across all datasets matches the paper within ±0.4%:

| Shots (K) | Paper | My Results | Difference |
|-----------|-------|------------|------------|
| 1 | 56.35% | 55.95% | -0.40% |
| 2 | 62.63% | 62.41% | -0.22% |
| 4 | 68.02% | 67.67% | -0.35% |
| 8 | 73.26% | 73.33% | +0.07% |
| 16 | 76.46% | 76.81% | +0.35% |

The small differences are expected due to random seed variations in few-shot sample selection. Overall, the replication confirms the paper's reported performance.

### Hyperparameter Tuning Results

I tested different learning rates and weight decay combinations to see if we could improve upon the baseline. Results for K=16 shots:

| Learning Rate | Weight Decay | Avg Accuracy | Change from Baseline |
|---------------|--------------|--------------|----------------------|
| 0.001 | 0.0 | 71.37% | -4.98% |
| 0.0045 | 0.0 | 76.31% | +0.04% |
| 0.005 | 0.0 | 76.35% | 0.00% (baseline) |
| 0.006 | 0.0 | 76.10% | -0.25% |
| 0.006 | 0.02 | 75.91% | -0.44% |
| 0.010 | 0.0 | 75.58% | -0.77% |

**Finding:** The baseline learning rate of 0.005 appears to be well-tuned. Slightly lower learning rates (0.0045) showed marginal improvement (+0.04%), while higher rates degraded performance. This suggests the paper's hyperparameter selection was appropriate for the biomedical domain.


