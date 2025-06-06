# ALFIA

Predicting Everything Using Adaptive Transformer Layer Fusion

This is a Transformer-based model using adaptive layer fusion technology and optional LoRA fine-tuning, specifically designed for medical data.

## Table of Contents
1. [Environment Setup](#environment-setup)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Model Inference](#model-inference)
5. [Results Analysis](#results-analysis)
6. [Advanced Usage](#advanced-usage)
7. [Troubleshooting](#troubleshooting)

## Environment Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- At least 16GB RAM
- 10GB+ free disk space

### Installation

1. **Clone or download the scripts**
```bash
# Save the training script as train_model.py
# Save the inference script as inference_model.py
```

2. **Install required packages**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
pip install pandas numpy scikit-learn matplotlib seaborn
pip install tqdm pathlib
pip install peft  # For LoRA support (optional)
```

3. **Verify installation**
```python
import torch
import transformers
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Transformers version: {transformers.__version__}")
```

## Data Preparation

### Input Data Format

Your CSV file should contain at least two columns:
- **Text column**: Contains the text data to be classified
- **Target column**: Contains binary labels (0 or 1)

Example CSV structure:
```csv
patient_description,hospital_expire_flag,subject_id
"Patient admitted with chest pain and shortness of breath...",0,12345
"Elderly patient with multiple comorbidities...",1,12346
"Young adult with acute symptoms...",0,12347
```

### Data Requirements
- **Minimum samples**: 100+ (recommended 1000+)
- **Text length**: Variable (will be truncated/padded to max_seq_length)
- **Label distribution**: Can handle imbalanced datasets
- **Missing values**: Rows with missing text or labels will be dropped

## Model Training

### Basic Training

1. **Prepare your data**
```bash
# Ensure your CSV file is ready
ls data/mimiciv_text.csv
```

2. **Run basic training**
```bash
python train_model.py \
    --input-csv data/mimiciv_text.csv \
    --text-col patient_description \
    --target-col hospital_expire_flag \
    --epochs 10 \
    --lr 0.0001
```

### Training with Custom Parameters

```bash
python train_model.py \
    --input-csv data/your_data.csv \
    --text-col your_text_column \
    --target-col your_target_column \
    --embed-model dmis-lab/biobert-base-cased-v1.2 \
    --num-layers 4 \
    --epochs 15 \
    --lr 0.0001 \
    --train-batch-size 8 \
    --max-seq-length 512 \
    --validation-split-ratio 0.15 \
    --early-stopping-patience 5 \
    --global-seed 42
```

### Training with LoRA (Recommended for large models)

```bash
python train_model.py \
    --input-csv data/mimiciv_text.csv \
    --text-col patient_description \
    --target-col hospital_expire_flag \
    --use-lora \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-dropout 0.05 \
    --lora-target-modules "query,key,value" \
    --epochs 10
```

### Training Output

After training, you'll find the following in the results directory:
```
results/
└── dmis_lab_biobert_base_cased_v1_2_20241201_143022/
    ├── hyperparameters.json                    # Training configuration
    ├── trained_model_weights.pth               # Model weights
    ├── lora_adapters/                          # LoRA adapters (if used)
    ├── embedded_data.csv                       # Final dataset with embeddings
    ├── best_validation_set_predictions.csv     # Best model predictions on validation
    ├── best_validation_metrics_with_ci.csv     # Validation metrics with confidence intervals
    ├── best_model_test_set_predictions.csv     # Best model predictions on test set
    ├── best_model_test_metrics_with_ci.csv     # Test metrics with confidence intervals
    ├── run_log.log                             # Detailed training log
    ├── figures/
    │   ├── hospital_expire_flag_dist.png       # Target distribution
    │   ├── hospital_expire_flag_pie.png        # Target pie chart
    │   ├── missing_values_top20.png            # Missing values analysis
    │   ├── confusion_matrix_best_val_epoch_current.png
    │   ├── confusion_matrix_current_best_model_test.png
    │   └── all_epoch_metrics.json              # Metrics for each epoch
    └── inference/                              # Inference results (created during inference)
```

## Model Inference

### Prepare Inference Configuration

Create a JSON file specifying your input data:

**inference_config.json**
```json
[
    {
        "file_path": "data/new_patients.csv",
        "text_col": "patient_description",
        "target_col": "hospital_expire_flag"
    },
    {
        "file_path": "data/validation_set.csv",
        "text_col": "clinical_notes",
        "target_col": null
    }
]
```

### Run Inference

1. **Using specific results directory**
```bash
python inference_model.py \
    --results-dir results/dmis_lab_biobert_base_cased_v1_2_20241201_143022 \
    --input-config-json inference_config.json \
    --batch-size 16
```

2. **Using latest training results (lazy inference)**
```bash
python inference_model.py \
    --lazy-inference \
    --base-results-path results \
    --input-config-json inference_config.json \
    --batch-size 16
```

3. **With custom parameters**
```bash
python inference_model.py \
    --results-dir results/your_model_directory \
    --input-config-json inference_config.json \
    --batch-size 32 \
    --seed 42 \
    --n-bootstraps 1000
```

### Inference Output

Results will be saved in the `inference/` subdirectory:
```
results/your_model/inference/
├── inference_results_new_patients_20241201_150322.csv
├── inference_metrics_new_patients_20241201_150322.json
├── inference_results_validation_set_20241201_150322.csv
└── inference_metrics_validation_set_20241201_150322.json
```

**inference_results_*.csv** contains:
- `text`: Original text
- `predicted_probability`: Model confidence (0-1)
- `predicted_label`: Binary prediction (0 or 1)
- `true_label`: Actual label (if provided)

**inference_metrics_*.json** contains:
- Performance metrics (accuracy, precision, recall, F1, AUC, etc.)
- 95% confidence intervals for each metric
- Confusion matrix components
- Performance timing information

## Results Analysis

### Understanding Metrics

The system provides comprehensive evaluation metrics:

1. **Classification Metrics**
   - **Accuracy**: Overall correctness
   - **Precision**: True positives / (True positives + False positives)
   - **Recall (Sensitivity)**: True positives / (True positives + False negatives)
   - **F1-score**: Harmonic mean of precision and recall
   - **Specificity**: True negatives / (True negatives + False positives)

2. **Probabilistic Metrics**
   - **ROC AUC**: Area under ROC curve
   - **AUPRC**: Area under Precision-Recall curve (better for imbalanced data)

3. **Confidence Intervals**
   - All metrics include 95% confidence intervals via bootstrapping
   - Useful for assessing statistical significance

### Interpreting Results

**Example metrics output:**
```json
{
    "accuracy": {
        "value": 0.8542,
        "confidence_interval_95": [0.8234, 0.8821]
    },
    "f1_score": {
        "value": 0.7891,
        "confidence_interval_95": [0.7456, 0.8298]
    },
    "auprc": {
        "value": 0.8123,
        "confidence_interval_95": [0.7789, 0.8445]
    }
}
```

**Performance timing:**
```json
{
    "performance": {
        "total_inference_time_seconds": 45.23,
        "samples_processed": 1000,
        "avg_time_per_sample_ms": 45.23,
        "peak_gpu_memory_mb": 2048.5
    }
}
```

## Advanced Usage

### Custom Model Configuration

1. **Different base models**
```bash
python train_model.py \
    --embed-model bert-base-uncased \
    --input-csv data/general_text.csv
```

2. **Adjust fusion layers**
```bash
python train_model.py \
    --num-layers 6 \
    --input-csv data/mimiciv_text.csv
```

3. **Memory optimization**
```bash
python train_model.py \
    --train-batch-size 2 \
    --max-seq-length 256 \
    --input-csv data/large_dataset.csv
```

### Hyperparameter Tuning

**For small datasets (< 5000 samples):**
```bash
python train_model.py \
    --use-lora \
    --lora-r 4 \
    --train-batch-size 4 \
    --lr 0.0002 \
    --epochs 20
```

**For large datasets (> 10000 samples):**
```bash
python train_model.py \
    --train-batch-size 16 \
    --lr 0.00005 \
    --epochs 5 \
    --validation-split-ratio 0.1
```

**For imbalanced datasets:**
```bash
python train_model.py \
    --early-stopping-patience 10 \
    --epochs 25 \
    --validation-split-ratio 0.2
```

### Batch Processing Multiple Datasets

Create multiple inference configurations:

**batch_inference_config.json**
```json
[
    {
        "file_path": "data/hospital_a.csv",
        "text_col": "notes",
        "target_col": "mortality"
    },
    {
        "file_path": "data/hospital_b.csv", 
        "text_col": "clinical_text",
        "target_col": "outcome"
    },
    {
        "file_path": "data/hospital_c.csv",
        "text_col": "patient_summary",
        "target_col": null
    }
]
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
```bash
# Reduce batch size
python train_model.py --train-batch-size 2

# Reduce sequence length
python train_model.py --max-seq-length 256

# Use gradient checkpointing (add to model if needed)
```

2. **Poor Performance**
```bash
# Increase training epochs
python train_model.py --epochs 20

# Adjust learning rate
python train_model.py --lr 0.00005

# Use LoRA for better fine-tuning
python train_model.py --use-lora
```

3. **Data Loading Errors**
```bash
# Check CSV format
head -5 data/your_file.csv

# Verify column names
python -c "import pandas as pd; print(pd.read_csv('data/your_file.csv').columns.tolist())"
```

4. **Missing Dependencies**
```bash
# Install PEFT for LoRA
pip install peft

# Update transformers
pip install --upgrade transformers
```

### Performance Optimization

1. **For faster training:**
   - Use smaller batch sizes if memory limited
   - Reduce max_seq_length for shorter texts
   - Use LoRA instead of full fine-tuning
   - Enable mixed precision (if supported)

2. **For better accuracy:**
   - Increase number of epochs
   - Use larger validation split
   - Try different learning rates
   - Experiment with different base models

3. **For production inference:**
   - Use larger batch sizes
   - Pre-load models once
   - Consider model quantization
   - Use GPU for inference

### Monitoring Training

Watch the training progress:
```bash
# Monitor log file in real-time
tail -f results/your_model_directory/run_log.log

# Check GPU usage
nvidia-smi

# Monitor training metrics
cat results/your_model_directory/figures/all_epoch_metrics.json
```

### Model Validation

Before deploying your model:

1. **Check validation metrics**
2. **Examine confusion matrices**
3. **Review confidence intervals**
4. **Test on held-out data**
5. **Verify inference speed**

This tutorial should help you get started with training and using the text classification system. For specific medical applications, ensure you follow appropriate data privacy and validation protocols.
