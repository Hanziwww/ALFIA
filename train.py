# ───── Version and SCRIPT_VERSION ───────────────────────────────────────────
SCRIPT_VERSION = "1.4.3" # Incremented version
# CHANGELOG removed as per user request

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional, Union, Any
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, classification_report, average_precision_score, log_loss
from sklearn.utils import resample # Added for bootstrapping
import json
import random
from datetime import datetime # Added for timestamp

# Import PEFT libraries
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT library not found. LoRA functionality will be disabled.")

# ───── logging setup (initial console setup) ───────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler()])
logging.info(f"Script Version: {SCRIPT_VERSION}")
# logging.info("Changelog:\n" + CHANGELOG) # Changelog display removed


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ───── paths (placeholders, will be redefined in main) ────────────────
INPUT_PATH = Path("data/mimiciv_text.csv")
BASE_RESULTS_DIR = Path("results")

EMBED_PATH: Path = None
FIG_PATH: Path = None
TRAINED_FUSION_PATH: Path = None
TRAINED_LORA_ADAPTER_PATH: Path = None
EPOCH_METRICS_SAVE_PATH: Path = None
LOG_FILE_PATH: Path = None
HYPERPARAMS_FILE_PATH: Path = None
RUN_SPECIFIC_DIR: Path = None
BEST_VAL_PREDS_SAVE_PATH: Path = None
METRICS_CI_VAL_SAVE_PATH: Path = None
BEST_TEST_PREDS_SAVE_PATH: Path = None
METRICS_CI_TEST_SAVE_PATH: Path = None


# ───── Control Parameters ───────────────────────────────────────────
TARGET_COL = "hospital_expire_flag"
TEXT_COL = "patient_description"
EMBED_MODEL = "dmis-lab/biobert-base-cased-v1.2"
GLOBAL_SEED = 42

DROP_COLS = [
    "subject_id", "hadm_id", "stay_id", "dischtime", "admittime",
    "icu_intime", "icu_outtime", "admission_type", "admission_location",
    "discharge_location", "careunit", "los_hospital", "los_icu",
    "dod", "mortality_365", "icu_expire_flag", "follow_up_years",
    "survival_days", "mortality_28", "mortality_30", "mortality_90",
    "apsiii", "sapsii", "sofa", "gcs", "sirs", "lods", "charlson",
    "meld", "sepsis_time", "oasis", "InvasiveVent", "cpr", "crrt",
    "rrt", "lvef_min", "aki_score", "aki_time", "microbiology", "race", "diagnosis_long_title1"
]
ALWAYS_KEEP = {"lactate_max"}

# ───── Training Hyperparameters ─────────────────────────
NUM_EPOCHS = 10
TRAIN_BATCH_SIZE = 4
LEARNING_RATE = 0.0001
MAX_SEQ_LENGTH = 512
NUM_FUSION_LAYERS = 4
MIN_SAMPLES_FOR_TOP_LAYER_FINETUNING = 5000
EARLY_STOPPING_PATIENCE = 5
TEST_SPLIT_RATIO_CONST = 0.125
VALIDATION_SPLIT_RATIO_DEFAULT = 0.125 / (1.0 - TEST_SPLIT_RATIO_CONST)
N_BOOTSTRAPS_CI = 1000

# ───── LoRA Hyperparameters (Defaults) ───────────────────
USE_LORA_DEFAULT = False
LORA_R_DEFAULT = 8
LORA_ALPHA_DEFAULT = 16
LORA_DROPOUT_DEFAULT = 0.05
LORA_TARGET_MODULES_DEFAULT = "query,key,value"

# ───── Reproducibility Function ─────────────────────────
def set_seed(seed_value: int = GLOBAL_SEED):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    logging.info(f"Global seed set to {seed_value}")

# ───── Adaptive Layer Fusion Module (Modified - Uses Average Pooling) ──────────────────────────────────
class AdaptiveLayerFusion(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4, # Number of layers to fuse from the top of the base model
        num_heads: int = 4, # Number of attention heads for layer attention mechanism
        dropout: float = 0.1,
        use_layer_gating: bool = True # Whether to use a sigmoid gate on layer weights
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_layer_gating = use_layer_gating
        self.num_layers_to_fuse = num_layers # Stores the intended number of layers to fuse

        head_dim = 64 # Fixed dimension for each head in layer attention
        self.layer_query_projection = nn.Linear(hidden_size, num_heads * head_dim)
        self.layer_key_projection = nn.Linear(hidden_size, num_heads * head_dim)
        self.layer_value_projection = nn.Linear(hidden_size, num_heads * head_dim)
        self.layer_attention_dropout = nn.Dropout(dropout)
        self.layer_attention_output = nn.Linear(num_heads * head_dim, hidden_size)

        self.layer_interaction = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size)
        )

        if use_layer_gating:
            self.layer_gate = nn.Sequential(
                nn.Linear(hidden_size, self.num_layers_to_fuse), # Output dim matches number of layers to fuse
                nn.Sigmoid()
            )

        self.content_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        # Token-level attention mechanism to get a local context vector
        self.token_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout/2), # Reduced dropout for this part
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        self.token_attention_dropout = nn.Dropout(dropout)

        # Global context processing layer (operates on average pooled representation)
        self.global_context_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Fuses local (token-attended) and global (average-pooled) contexts
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # Input is concatenation of local and global
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.register_buffer('last_attention_mask', None, persistent=False) # Stores the attention mask from the last forward pass


    def _compute_layer_weights(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes attention weights over the input hidden_states (from different layers of a base model).
        Uses average-pooled representations of each layer as keys and values for an attention mechanism.
        The query is derived from the mean of these average-pooled representations.
        """
        batch_size = hidden_states[0].size(0)
        num_layers_being_fused = len(hidden_states) # Actual number of layers passed in
        head_dim = self.layer_query_projection.out_features // self.num_heads

        # Use the stored attention_mask to perform masked average pooling for each layer
        if self.last_attention_mask is not None:
            attention_mask = self.last_attention_mask
            mask_expanded = attention_mask.unsqueeze(-1).float() # (batch, seq_len, 1)
            avg_pooled_layers = []
            for layer in hidden_states:
                masked_layer = layer * mask_expanded # Apply mask
                seq_lengths = attention_mask.sum(dim=1, keepdim=True).float() # (batch, 1)
                avg_pooled = masked_layer.sum(dim=1) / seq_lengths.clamp(min=1) # (batch, hidden_size)
                avg_pooled_layers.append(avg_pooled)
            avg_reps = torch.stack(avg_pooled_layers, dim=1) # (batch, num_layers_being_fused, hidden_size)
        else:
            # Fallback if mask is not available (should not happen in normal operation)
            logging.warning("last_attention_mask not found in AdaptiveLayerFusion. Using simple mean pooling for layer weights.")
            avg_pooled_layers = [layer.mean(dim=1) for layer in hidden_states]
            avg_reps = torch.stack(avg_pooled_layers, dim=1)

        query_input = avg_reps.mean(dim=1) # (batch, hidden_size), query derived from average of all layer representations

        query = self.layer_query_projection(query_input).view(batch_size, self.num_heads, head_dim) # (batch, num_heads, head_dim)
        key = self.layer_key_projection(avg_reps).view(batch_size, num_layers_being_fused, self.num_heads, head_dim) # (batch, num_layers, num_heads, head_dim)
        value = self.layer_value_projection(avg_reps).view(batch_size, num_layers_being_fused, self.num_heads, head_dim) # (batch, num_layers, num_heads, head_dim)

        key = key.permute(0, 2, 1, 3) # (batch, num_heads, num_layers, head_dim)
        value = value.permute(0, 2, 1, 3) # (batch, num_heads, num_layers, head_dim)

        attention_scores = torch.matmul(query.unsqueeze(2), key.transpose(-1, -2)) / math.sqrt(head_dim) # (batch, num_heads, 1, num_layers)
        attention_weights = F.softmax(attention_scores, dim=-1) # (batch, num_heads, 1, num_layers)
        attention_weights = self.layer_attention_dropout(attention_weights)

        context = torch.matmul(attention_weights, value) # (batch, num_heads, 1, head_dim)
        context = context.squeeze(2).contiguous().view(batch_size, self.num_heads * head_dim) # (batch, num_heads * head_dim)
        context = self.layer_attention_output(context) # (batch, hidden_size)

        if self.use_layer_gating:
            # Dynamically adjust layer_gate's output dimension if it doesn't match num_layers_being_fused
            if self.layer_gate[0].out_features != num_layers_being_fused:
                logging.warning(
                    f"AdaptiveLayerFusion's layer_gate was initialized for {self.layer_gate[0].out_features} layers, "
                    f"but received {num_layers_being_fused} layers in forward pass. "
                    f"Re-initializing layer_gate's final linear layer for {num_layers_being_fused} outputs."
                )
                original_device = self.layer_gate[0].weight.device
                original_dtype = self.layer_gate[0].weight.dtype
                self.layer_gate[0] = nn.Linear(self.hidden_size, num_layers_being_fused).to(device=original_device, dtype=original_dtype)

            gate_weights = self.layer_gate(context) # (batch, num_layers_being_fused)
            # Sanity check after potential dynamic adjustment
            if gate_weights.shape[1] != num_layers_being_fused:
                logging.error(f"CRITICAL: Layer gate output features ({gate_weights.shape[1]}) "
                                      f"still does not match number of layers being fused ({num_layers_being_fused}) "
                                      "after attempted dynamic adjustment. Falling back to uniform weights.")
                return torch.ones(batch_size, num_layers_being_fused, device=context.device) / num_layers_being_fused
            return gate_weights # (batch_size, num_layers_being_fused)
        else:
            # If not using gating, return the averaged attention weights across heads
            return attention_weights.squeeze(2).mean(dim=1) # (batch_size, num_layers_being_fused)

    def forward(
        self,
        all_hidden_states: List[torch.Tensor], # List of (batch, seq_len, hidden_size) tensors
        attention_mask: torch.Tensor # (batch, seq_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]: # (pooled_output, layer_weights)
        if not all_hidden_states:
            raise ValueError("all_hidden_states list cannot be empty.")

        self.last_attention_mask = attention_mask # Store for _compute_layer_weights
        layer_weights = self._compute_layer_weights(all_hidden_states) # (batch, num_fused_layers)

        # Fallback if layer_weights shape is incorrect (e.g., due to dynamic adjustment failure)
        if layer_weights.shape[1] != len(all_hidden_states):
            logging.error(f"Shape mismatch after _compute_layer_weights: layer_weights has {layer_weights.shape[1]} weights, "
                          f"but {len(all_hidden_states)} hidden states were provided. Using uniform weights as fallback.")
            num_fused_layers = len(all_hidden_states)
            layer_weights = torch.ones(all_hidden_states[0].size(0), num_fused_layers, device=all_hidden_states[0].device) / num_fused_layers

        # Weighted sum of layer hidden states
        weighted_states = torch.zeros_like(all_hidden_states[0]) # (batch, seq_len, hidden_size)
        for i, layer_state in enumerate(all_hidden_states):
            current_layer_weight = layer_weights[:, i].unsqueeze(1).unsqueeze(2) # (batch, 1, 1)
            weighted_states += current_layer_weight * layer_state

        interaction_states = self.layer_interaction(weighted_states) # (batch, seq_len, hidden_size)
        projected_states = self.content_projection(weighted_states)
        enhanced_states = self.layer_norm1(weighted_states + projected_states + interaction_states) # (batch, seq_len, hidden_size)

        # Token-level attention for local context
        token_scores = self.token_attention(enhanced_states).squeeze(-1) # (batch, seq_len)
        token_scores = token_scores.masked_fill(attention_mask == 0, -1e9) # Apply attention mask
        token_weights = F.softmax(token_scores, dim=1) # (batch, seq_len)
        token_weights = self.token_attention_dropout(token_weights)
        local_context = torch.bmm(token_weights.unsqueeze(1), enhanced_states).squeeze(1) # (batch, hidden_size)

        # Global context (masked average pooling over enhanced_states)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        masked_enhanced_states = enhanced_states * mask_expanded
        seq_lengths = attention_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        global_context_unprocessed = masked_enhanced_states.sum(dim=1) / seq_lengths # (batch, hidden_size)
        global_context = self.global_context_layer(global_context_unprocessed) # (batch, hidden_size)

        # Combine local and global contexts
        combined_context = torch.cat([local_context, global_context], dim=1) # (batch, hidden_size * 2)
        pooled_output = self.context_fusion(combined_context) # (batch, hidden_size)

        projected_output = self.output_projection(pooled_output)
        pooled_output = self.layer_norm2(pooled_output + projected_output) # Final pooled output

        return pooled_output, layer_weights


# ───── Attentional Classifier Head ───────────────────────────────────
class AttentionalClassifierHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, num_attn_heads: int = 4, ffn_expansion: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        # Adjust num_attn_heads if hidden_size is not divisible by it
        if hidden_size % num_attn_heads != 0:
            valid_heads = [h for h in [16, 12, 8, 4, 2, 1] if hidden_size % h == 0] # Common head counts
            if not valid_heads: # Should not happen if hidden_size is typical (e.g., 768)
                raise ValueError(f"The hidden size ({hidden_size}) is not divisible by any common attention head counts (1,2,4,8,12,16).")
            original_num_attn_heads = num_attn_heads
            num_attn_heads = valid_heads[0] # Pick the largest valid head count
            logging.warning(f"AttentionalClassifierHead: num_attn_heads ({original_num_attn_heads}) caused hidden_size ({hidden_size}) not to be divisible. Adjusted to {num_attn_heads}.")

        self.hidden_size = hidden_size
        self.num_attn_heads = num_attn_heads
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=self.num_attn_heads, dropout=dropout_rate, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * ffn_expansion, hidden_size)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, num_classes)

    def forward(self, fused_embeddings: torch.Tensor) -> torch.Tensor: # fused_embeddings: (batch, hidden_size)
        x = fused_embeddings.unsqueeze(1) # (batch, 1, hidden_size) - MHA expects sequence
        attn_output, _ = self.attention(query=x, key=x, value=x) # Self-attention on the single fused vector
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        x_squeezed = x.squeeze(1) # (batch, hidden_size)
        return self.output_layer(x_squeezed)


# ───── Complete Classification Model ─────────────────────────────────────
class TextEmbedderWithClassifier(nn.Module):
    def __init__(self,
                 base_model_name: str,
                 num_fusion_layers: int, # How many top layers from base_model to fuse
                 fusion_dropout: float = 0.1,
                 num_classes: int = 1, # For binary classification with BCEWithLogitsLoss
                 use_lora: bool = False,
                 lora_config_dict: Optional[Dict[str, Any]] = None,
                 classifier_num_attn_heads: int = 4, # For AttentionalClassifierHead
                 classifier_ffn_expansion: int = 2): # For AttentionalClassifierHead
        super().__init__()
        self.num_fusion_layers = num_fusion_layers
        self.use_lora = use_lora and PEFT_AVAILABLE

        self.base_model_internal = AutoModel.from_pretrained(base_model_name, output_hidden_states=True)
        self.num_base_model_total_hidden_states = self.base_model_internal.config.num_hidden_layers + 1 # Includes embedding layer

        if self.use_lora:
            if lora_config_dict is None: lora_config_dict = {}
            raw_target_modules = lora_config_dict.get("target_modules", LORA_TARGET_MODULES_DEFAULT)
            if isinstance(raw_target_modules, str):
                target_modules_list = [m.strip() for m in raw_target_modules.split(',') if m.strip()]
            elif isinstance(raw_target_modules, list):
                target_modules_list = raw_target_modules
            else: # Fallback to default if type is unexpected
                target_modules_list = [m.strip() for m in LORA_TARGET_MODULES_DEFAULT.split(',') if m.strip()]

            lora_config = LoraConfig(
                r=lora_config_dict.get("r", LORA_R_DEFAULT),
                lora_alpha=lora_config_dict.get("lora_alpha", LORA_ALPHA_DEFAULT),
                target_modules=target_modules_list,
                lora_dropout=lora_config_dict.get("lora_dropout", LORA_DROPOUT_DEFAULT),
                bias="none", # Common LoRA setting
                task_type=TaskType.FEATURE_EXTRACTION # Or SEQ_CLS if LoRA is applied to a model with a classification head
            )
            self.base_model = get_peft_model(self.base_model_internal, lora_config)
            logging.info("LoRA applied to the base model.")
            self.base_model.print_trainable_parameters()
        else:
            self.base_model = self.base_model_internal
            if use_lora and not PEFT_AVAILABLE: # Log if LoRA was intended but not possible
                logging.warning("LoRA requested but PEFT library not available. Proceeding without LoRA.")

        self.fusion_module = AdaptiveLayerFusion(
            hidden_size=self.base_model.config.hidden_size,
            num_layers=num_fusion_layers, # This is the `num_layers_to_fuse` for the gate initialization
            num_heads=4, # Default from AdaptiveLayerFusion
            dropout=fusion_dropout
        )
        self.classifier = AttentionalClassifierHead(
            hidden_size=self.base_model.config.hidden_size, # Input to classifier is output of fusion
            num_classes=num_classes,
            num_attn_heads=classifier_num_attn_heads,
            ffn_expansion=classifier_ffn_expansion,
            dropout_rate=fusion_dropout # Use same dropout as fusion for consistency
        )
        logging.info(f"TextEmbedder: Using AttentionalClassifierHead with num_attn_heads={self.classifier.num_attn_heads}, ffn_expansion={classifier_ffn_expansion}, dropout={fusion_dropout}")


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        base_model_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        all_layers_hidden_states = list(base_model_outputs.hidden_states) # Tuple to list

        # Validate num_fusion_layers against actual available transformer layers
        actual_num_transformer_layers = len(all_layers_hidden_states) -1 # Exclude initial embedding layer output
        if not (0 < self.num_fusion_layers <= actual_num_transformer_layers):
            raise ValueError(
                f"num_fusion_layers ({self.num_fusion_layers}) is invalid. "
                f"The base model has {actual_num_transformer_layers} transformer layers. "
                f"num_fusion_layers must be > 0 and <= {actual_num_transformer_layers}."
            )

        # Select the top `num_fusion_layers` from the transformer layers (excluding embedding layer)
        # `all_layers_hidden_states` includes embedding layer as [0], then transformer layers [1]...[N]
        # So, to get the top K transformer layers, we take from the end.
        hidden_states_for_fusion = all_layers_hidden_states[-self.num_fusion_layers:]

        fused_embeddings, _ = self.fusion_module(hidden_states_for_fusion, attention_mask)
        return self.classifier(fused_embeddings)

# ───── Custom Dataset ───────────────────────────────────────
class MIMICIVTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item_idx):
        text = str(self.texts[item_idx])
        label = self.labels[item_idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False, # Not needed for most BERT-like models
            padding='max_length', # Pad to max_len
            truncation=True, # Truncate to max_len
            return_attention_mask=True,
            return_tensors='pt', # Return PyTorch tensors
        )
        return {
            'text': text, # Keep original text for reference if needed
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float) # For BCEWithLogitsLoss
        }

# ───── Compute Evaluation Metrics Function ─────────────────────────────────
def compute_metrics(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    preds_cpu = preds.detach().cpu().numpy()
    targets_cpu = targets.detach().cpu().numpy()

    # Ensure preds_cpu is 1D for binary classification probability scores
    if preds_cpu.ndim > 1 and preds_cpu.shape[1] == 1:
        preds_cpu = preds_cpu.flatten()

    binary_preds_np = (preds_cpu >= threshold).astype(int)

    accuracy = (binary_preds_np == targets_cpu).mean().item()
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets_cpu, binary_preds_np, average='binary', zero_division=0
    )

    cm = confusion_matrix(targets_cpu, binary_preds_np, labels=[0, 1]) # Ensure labels are [0,1]

    # Handle cases where confusion matrix might not be 2x2 (e.g., all preds/targets are same class)
    if cm.size == 4: # Standard 2x2 case
        tn, fp, fn, tp = cm.ravel()
    else: # Degenerate cases
        tn, fp, fn, tp = 0, 0, 0, 0
        if (targets_cpu == 0).all() and (binary_preds_np == 0).all(): tn = len(targets_cpu)
        elif (targets_cpu == 1).all() and (binary_preds_np == 1).all(): tp = len(targets_cpu)
        elif (targets_cpu == 0).all() and (binary_preds_np == 1).all(): fp = len(targets_cpu) # All actual negatives predicted positive
        elif (targets_cpu == 1).all() and (binary_preds_np == 0).all(): fn = len(targets_cpu) # All actual positives predicted negative
        # Other combinations (e.g. mixed actuals, all same predictions) are also possible but covered by initializing to 0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc_roc = float('nan')
    auprc = float('nan')

    # AUC-ROC and AUPRC require at least two classes in targets
    if len(np.unique(targets_cpu)) > 1:
        try:
            auc_roc = roc_auc_score(targets_cpu, preds_cpu) # Use probabilities for AUCs
        except ValueError as e:
            logging.debug(f"AUC-ROC calculation error: {e}. Targets unique: {np.unique(targets_cpu)}, Preds shape: {preds_cpu.shape}")
        try:
            auprc = average_precision_score(targets_cpu, preds_cpu) # Use probabilities for AUCs
        except ValueError as e:
            logging.debug(f"AUPRC calculation error: {e}. Targets unique: {np.unique(targets_cpu)}, Preds shape: {preds_cpu.shape}")
    else:
        logging.debug("Single class in targets, AUC-ROC and AUPRC are not defined (NaN).")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall, # Sensitivity
        'f1': f1,
        'specificity': specificity,
        'balanced_acc': (recall + specificity) / 2.0 if recall is not None and specificity is not None else 0.0,
        'auc_roc': auc_roc,
        'auprc': auprc,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0.0, # Positive Predictive Value (same as precision)
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0.0, # Negative Predictive Value
        'pos_ratio_actual': targets_cpu.mean().item(), # Actual positive rate
        'neg_ratio_actual': 1.0 - targets_cpu.mean().item(), # Actual negative rate
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }

# ───── Calculate Evaluation Metrics with 95% CI Function ─────────────────────────
def calculate_metrics_with_ci(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = N_BOOTSTRAPS_CI, # Global default
    alpha: float = 0.05 # For 95% CI
) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)

    # Ensure y_pred_proba is 1D
    if y_pred_proba.ndim > 1 and y_pred_proba.shape[1] == 1:
        y_pred_proba = y_pred_proba.flatten()

    metrics_results_list = []
    bootstrapped_metrics_values = {
        'roc_auc': [], 'prauc': [], 'logloss': [],
        'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'specificity': []
    }

    eps = 1e-15 # For clipping probabilities in log_loss
    y_pred_proba_clipped = np.clip(y_pred_proba, eps, 1 - eps)

    # Calculate point estimates first
    point_estimates = {}
    y_pred_binary_orig = (y_pred_proba >= threshold).astype(int)

    if len(np.unique(y_true)) > 1:
        try: point_estimates['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError: point_estimates['roc_auc'] = np.nan
        try: point_estimates['prauc'] = average_precision_score(y_true, y_pred_proba)
        except ValueError: point_estimates['prauc'] = np.nan
    else:
        point_estimates['roc_auc'] = np.nan
        point_estimates['prauc'] = np.nan

    try: point_estimates['logloss'] = log_loss(y_true, y_pred_proba_clipped, labels=[0,1]) # Ensure labels are specified for consistency
    except ValueError: point_estimates['logloss'] = np.nan

    precision_orig, recall_orig, f1_orig, _ = precision_recall_fscore_support(
        y_true, y_pred_binary_orig, average='binary', zero_division=0
    )
    cm_orig = confusion_matrix(y_true, y_pred_binary_orig, labels=[0,1])
    if cm_orig.size == 4: tn_orig, fp_orig, fn_orig, tp_orig = cm_orig.ravel()
    else: tn_orig, fp_orig, fn_orig, tp_orig = 0,0,0,0 # Handle degenerate cases

    point_estimates['accuracy'] = (y_pred_binary_orig == y_true).mean()
    point_estimates['f1'] = f1_orig
    point_estimates['precision'] = precision_orig
    point_estimates['recall'] = recall_orig
    point_estimates['specificity'] = tn_orig / (tn_orig + fp_orig) if (tn_orig + fp_orig) > 0 else 0.0

    # Bootstrapping
    rng = np.random.RandomState(GLOBAL_SEED) # Use global seed for reproducibility of bootstrapping
    for _ in tqdm(range(n_bootstraps), desc="Bootstrapping CI", leave=False):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(indices) == 0: continue # Should not happen if y_true is not empty

        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        y_pred_proba_boot_clipped = np.clip(y_pred_proba_boot, eps, 1 - eps)
        y_pred_binary_boot = (y_pred_proba_boot >= threshold).astype(int)

        if len(np.unique(y_true_boot)) > 1:
            try: bootstrapped_metrics_values['roc_auc'].append(roc_auc_score(y_true_boot, y_pred_proba_boot))
            except ValueError: bootstrapped_metrics_values['roc_auc'].append(np.nan)
            try: bootstrapped_metrics_values['prauc'].append(average_precision_score(y_true_boot, y_pred_proba_boot))
            except ValueError: bootstrapped_metrics_values['prauc'].append(np.nan)
        else:
            bootstrapped_metrics_values['roc_auc'].append(np.nan)
            bootstrapped_metrics_values['prauc'].append(np.nan)

        try: bootstrapped_metrics_values['logloss'].append(log_loss(y_true_boot, y_pred_proba_boot_clipped, labels=[0,1]))
        except ValueError: bootstrapped_metrics_values['logloss'].append(np.nan)

        precision_boot, recall_boot, f1_boot, _ = precision_recall_fscore_support(
            y_true_boot, y_pred_binary_boot, average='binary', zero_division=0
        )
        cm_boot = confusion_matrix(y_true_boot, y_pred_binary_boot, labels=[0,1])
        if cm_boot.size == 4: tn_boot, fp_boot, fn_boot, tp_boot = cm_boot.ravel()
        else: tn_boot, fp_boot, fn_boot, tp_boot = 0,0,0,0


        bootstrapped_metrics_values['accuracy'].append((y_pred_binary_boot == y_true_boot).mean())
        bootstrapped_metrics_values['f1'].append(f1_boot)
        bootstrapped_metrics_values['precision'].append(precision_boot)
        bootstrapped_metrics_values['recall'].append(recall_boot)
        bootstrapped_metrics_values['specificity'].append(tn_boot / (tn_boot + fp_boot) if (tn_boot + fp_boot) > 0 else 0.0)

    # Calculate CIs from bootstrapped values
    for metric_name, point_estimate_val in point_estimates.items():
        boot_values_for_metric = [v for v in bootstrapped_metrics_values[metric_name] if not np.isnan(v)] # Filter out NaNs

        ci_lower, ci_upper = np.nan, np.nan
        if len(boot_values_for_metric) > 1: # Need at least 2 values for percentile
            ci_lower = np.percentile(boot_values_for_metric, (alpha / 2) * 100)
            ci_upper = np.percentile(boot_values_for_metric, (1 - alpha / 2) * 100)

        metrics_results_list.append({
            'metric': metric_name,
            'value': point_estimate_val,
            'ci_lower (95%)': ci_lower,
            'ci_upper (95%)': ci_upper
        })
    return pd.DataFrame(metrics_results_list)


# ───── DataLoader worker_init_fn for reproducibility ─────────────────
def seed_worker(worker_id):
    # Ensures that each worker in DataLoader has a different, but reproducible, seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# ───── Training Function ──────────────────────────────────
def train_and_save_fusion_model(
    texts: List[str], labels: List[int], base_model_name: str,
    num_fusion_layers_to_train: int, fusion_model_save_path: Path,
    lora_adapter_save_path: Path, use_lora: bool, lora_config_dict: Dict[str, Any],
    validation_split_ratio: float, # Proportion of (train+val) data to use for validation
    tokenizer_max_len: int = MAX_SEQ_LENGTH,
    train_batch_size: int = TRAIN_BATCH_SIZE, num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE, early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
    seed: int = GLOBAL_SEED,
    test_dataloader_main: Optional[DataLoader] = None # Kept for structure, but test_dataloader is now created inside
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Training on {device} for base model: {base_model_name}, fusing top {num_fusion_layers_to_train} layers.")
    if use_lora and PEFT_AVAILABLE: logging.info(f"LoRA ENABLED. Config: {lora_config_dict}. Adapters to: {lora_adapter_save_path.parent}")
    else: logging.info("LoRA DISABLED.")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    texts_np_all, labels_np_all = np.array(texts), np.array(labels)
    set_seed(seed) # Set seed before any data splitting or model initialization

    # Split into (Train+Val) and Test first
    stratify_main_split = labels_np_all if len(np.unique(labels_np_all)) > 1 else None
    train_val_texts_np, test_texts_np, train_val_labels_np, test_labels_np = train_test_split(
        texts_np_all, labels_np_all, test_size=TEST_SPLIT_RATIO_CONST, random_state=seed, stratify=stratify_main_split
    )

    train_texts_list: List[str]
    train_labels_list: List[int]
    val_texts_list: List[str] = []
    val_labels_list: List[int] = []

    # Split (Train+Val) into Train and Val if validation_split_ratio > 0
    if validation_split_ratio > 0 and len(train_val_texts_np) > 0 :
        stratify_val_split = train_val_labels_np if len(np.unique(train_val_labels_np)) > 1 else None
        train_texts_np_final, val_texts_np_final, train_labels_np_final, val_labels_np_final = train_test_split(
            train_val_texts_np, train_val_labels_np, test_size=validation_split_ratio, random_state=seed, stratify=stratify_val_split
        )
        train_texts_list = train_texts_np_final.tolist()
        train_labels_list = train_labels_np_final.tolist()
        val_texts_list = val_texts_np_final.tolist()
        val_labels_list = val_labels_np_final.tolist()
    else: # No validation split, all train_val data goes to training
        train_texts_list = train_val_texts_np.tolist()
        train_labels_list = train_val_labels_np.tolist()

    test_texts_list = test_texts_np.tolist()
    test_labels_list = test_labels_np.tolist()

    logging.info(f"Data split: {len(train_texts_list)} Train, {len(val_texts_list)} Val, {len(test_texts_list)} Test.")

    train_dataset = MIMICIVTextDataset(train_texts_list, train_labels_list, tokenizer, tokenizer_max_len)
    g = torch.Generator(); g.manual_seed(seed) # Generator for DataLoader shuffling reproducibility
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                                  num_workers=min(2, os.cpu_count()//2 if os.cpu_count() else 1), # Sensible num_workers
                                  pin_memory=(device=="cuda"), worker_init_fn=seed_worker, generator=g)

    val_dataloader = None
    if val_texts_list:
        val_dataset = MIMICIVTextDataset(val_texts_list, val_labels_list, tokenizer, tokenizer_max_len)
        val_dataloader = DataLoader(val_dataset, batch_size=train_batch_size, # Use train_batch_size for val for consistency
                                    num_workers=min(2, os.cpu_count()//2 if os.cpu_count() else 1),
                                    pin_memory=(device=="cuda"), worker_init_fn=seed_worker) # No shuffle for val

    test_dataloader = None # Initialize test_dataloader
    if test_texts_list:
        test_dataset = MIMICIVTextDataset(test_texts_list, test_labels_list, tokenizer, tokenizer_max_len)
        test_dataloader = DataLoader(test_dataset, batch_size=train_batch_size,
                                     num_workers=min(2, os.cpu_count()//2 if os.cpu_count() else 1),
                                     pin_memory=(device=="cuda"), worker_init_fn=seed_worker) # No shuffle for test


    # Calculate positive class weight for imbalanced datasets for BCEWithLogitsLoss
    num_pos_train = np.sum(train_labels_list); num_neg_train = len(train_labels_list) - num_pos_train
    pos_weight_train = num_neg_train / num_pos_train if num_pos_train > 0 and num_neg_train > 0 else 1.0
    logging.info(f"Train Stats: Samples={len(train_labels_list)}, Pos={num_pos_train} ({num_pos_train/len(train_labels_list)*100:.2f}% if len(train_labels_list) > 0 else 0), Neg={num_neg_train}. Pos Weight: {pos_weight_train:.4f}")
    if val_dataloader: logging.info(f"Val Stats: Samples={len(val_labels_list)}, Pos={np.sum(val_labels_list)} ({np.sum(val_labels_list)/len(val_labels_list)*100:.2f}% if len(val_labels_list) > 0 else 0)")
    if test_dataloader: logging.info(f"Test Stats: Samples={len(test_labels_list)}, Pos={np.sum(test_labels_list)} ({np.sum(test_labels_list)/len(test_labels_list)*100:.2f}% if len(test_labels_list) > 0 else 0)")


    model = TextEmbedderWithClassifier(base_model_name, num_fusion_layers_to_train, use_lora=use_lora, lora_config_dict=lora_config_dict).to(device)

    # Parameter freezing logic if not using LoRA
    if not use_lora or not PEFT_AVAILABLE:
        dataset_size = len(texts) # Using the original full dataset size for this decision
        if dataset_size >= MIN_SAMPLES_FOR_TOP_LAYER_FINETUNING:
            logging.info(f"(No LoRA) Fine-tuning top layers of base model ({dataset_size} samples).")
            if hasattr(model.base_model, 'encoder') and hasattr(model.base_model.encoder, 'layer'):
                num_actual_layers = len(model.base_model.encoder.layer)
                # Unfreeze top 2 transformer layers and pooler (if exists)
                layers_to_unfreeze_prefixes = [f"encoder.layer.{num_actual_layers-1}.", f"encoder.layer.{num_actual_layers-2}."]
                if hasattr(model.base_model, 'pooler'): layers_to_unfreeze_prefixes.append("pooler.")

                for name, param in model.base_model.named_parameters():
                    param.requires_grad = any(name.startswith(p_prefix) for p_prefix in layers_to_unfreeze_prefixes)
                    if param.requires_grad: logging.debug(f"Unfreezing (No LoRA): {name}")
            else:
                logging.warning("(No LoRA) Could not identify encoder layers for partial unfreezing. Base model might be fully frozen or fully trainable.")
        else:
            logging.info(f"(No LoRA) Freezing entire base model due to small dataset size ({dataset_size} samples). Only fusion/classifier trainable.")
            for param in model.base_model.parameters():
                param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_dataloader) * num_epochs * 0.1), num_training_steps=len(train_dataloader) * num_epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_train], device=device))

    best_val_auprc = 0.0
    # These store the state of the best model found so far *in memory* during training.
    # Model weights (.pth, LoRA) are saved/overwritten immediately when a new best is found.
    # CSVs for metrics/preds are also saved immediately.
    best_model_state_in_memory: Optional[Dict[str, Any]] = None # For fusion/classifier weights
    best_lora_model_peft_instance_in_memory: Optional[PeftModel] = None # For LoRA adapters

    epochs_no_improve = 0
    all_epoch_metrics_run = []

    logging.info("Starting training loop...")
    for epoch in range(num_epochs):
        model.train(); total_train_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch in progress_bar:
            ids, mask, tgts = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(input_ids=ids, attention_mask=mask)
            loss = criterion(outputs, tgts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0) # Gradient clipping
            optimizer.step(); scheduler.step()
            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})

        avg_train_loss = total_train_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        logging.info(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        current_epoch_metrics = {'epoch': epoch + 1, 'avg_train_loss': avg_train_loss}

        if val_dataloader:
            model.eval(); total_val_loss = 0; all_val_preds_epoch_tensors, all_val_labels_epoch_tensors = [], []
            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} [Val]", leave=False)
            with torch.no_grad():
                for batch in val_progress_bar:
                    ids, mask, tgts = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device).unsqueeze(1)
                    outputs = model(input_ids=ids, attention_mask=mask)
                    loss = criterion(outputs, tgts) # Use same criterion for val loss
                    total_val_loss += loss.item()
                    all_val_preds_epoch_tensors.append(torch.sigmoid(outputs.detach())) # Store probabilities
                    all_val_labels_epoch_tensors.append(tgts.detach())
                    val_progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})

            avg_val_loss = total_val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
            current_val_preds_tensor = torch.cat(all_val_preds_epoch_tensors)
            current_val_labels_tensor = torch.cat(all_val_labels_epoch_tensors)

            val_metrics_epoch = compute_metrics(current_val_preds_tensor, current_val_labels_tensor)
            current_epoch_metrics.update({'avg_val_loss': avg_val_loss, **{f'val_{k}': v for k,v in val_metrics_epoch.items()}})
            auprc_val = val_metrics_epoch.get('auprc', 0.0); auprc_val = 0.0 if np.isnan(auprc_val) else auprc_val # Handle NaN AUPRC
            logging.info(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}, Val F1: {val_metrics_epoch.get('f1',0):.4f}, Val AUC: {val_metrics_epoch.get('auc_roc',0):.4f}, Val AUPRC: {auprc_val:.4f}")

            if auprc_val > best_val_auprc:
                best_val_auprc = auprc_val
                # Store the model state IN MEMORY first (for potential end-of-training save if logic changes)
                best_model_state_in_memory = {
                    'fusion_module_state_dict': model.fusion_module.state_dict(),
                    'classifier_head_state_dict': model.classifier.state_dict()
                }
                if use_lora and PEFT_AVAILABLE:
                    best_lora_model_peft_instance_in_memory = model.base_model # This is a PeftModel instance

                epochs_no_improve = 0
                logging.info(f"*** New best Val AUPRC: {best_val_auprc:.4f} at Epoch {epoch+1}. Saving weights and results... ***")

                # --- Immediately Save/Overwrite Model Weights for this new best model ---
                current_best_model_weights_to_save = {
                    'fusion_module_state_dict': model.fusion_module.state_dict(),
                    'classifier_head_state_dict': model.classifier.state_dict()
                }
                torch.save(current_best_model_weights_to_save, fusion_model_save_path) # Overwrites
                logging.info(f"Best fusion/classifier weights (Epoch {epoch+1}) saved to {fusion_model_save_path}")
                if use_lora and PEFT_AVAILABLE:
                    model.base_model.save_pretrained(str(lora_adapter_save_path)) # Overwrites
                    logging.info(f"Best LoRA adapters (Epoch {epoch+1}) saved to {lora_adapter_save_path}")


                # --- Save/Overwrite Validation Results for this new best model ---
                current_best_val_preds_np = current_val_preds_tensor.cpu().numpy().flatten()
                current_best_val_labels_np = current_val_labels_tensor.cpu().numpy().flatten()

                val_preds_df = pd.DataFrame({'true_labels': current_best_val_labels_np, 'pred_probabilities': current_best_val_preds_np})
                val_preds_df.to_csv(BEST_VAL_PREDS_SAVE_PATH, index=False) # Overwrites
                logging.info(f"Best validation set predictions (Epoch {epoch+1}) saved to {BEST_VAL_PREDS_SAVE_PATH}")

                metrics_ci_val_df = calculate_metrics_with_ci(current_best_val_labels_np, current_best_val_preds_np)
                metrics_ci_val_df.to_csv(METRICS_CI_VAL_SAVE_PATH, index=False) # Overwrites
                logging.info(f"Metrics with 95% CI for best validation set (Epoch {epoch+1}) saved to {METRICS_CI_VAL_SAVE_PATH}")
                logging.info(f"Best Validation Set Metrics (Epoch {epoch+1}, with 95% CI):\n" + metrics_ci_val_df.to_string())

                cm_data_val = np.array([[val_metrics_epoch.get('tn',0), val_metrics_epoch.get('fp',0)],
                                        [val_metrics_epoch.get('fn',0), val_metrics_epoch.get('tp',0)]])
                plt.figure(figsize=(8,6)); sns.heatmap(cm_data_val, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Neg', 'Pred Pos'], yticklabels=['Actual Neg', 'Actual Pos'])
                plt.title(f'Confusion Matrix (Best Val Epoch {epoch+1} - AUPRC: {best_val_auprc:.4f})'); plt.tight_layout()
                plt.savefig(FIG_PATH/f"confusion_matrix_best_val_epoch_current.png"); plt.close() # Overwrites
                logging.info(f"Confusion matrix for best validation epoch (Epoch {epoch+1}) saved to {FIG_PATH/'confusion_matrix_best_val_epoch_current.png'}")

                # --- Evaluate this new best model on Test Set and Save/Overwrite ---
                if test_dataloader:
                    set_seed(seed) # Re-set seed for deterministic test pass
                    model.eval() # Ensure model is in eval mode
                    current_best_model_test_preds_list, current_best_model_test_labels_list = [], []
                    test_eval_bar = tqdm(test_dataloader, desc=f"Epoch {epoch+1} [Best Model Test Eval]", leave=False)
                    with torch.no_grad():
                        for batch_test in test_eval_bar:
                            ids_test, mask_test, tgts_test = batch_test['input_ids'].to(device), batch_test['attention_mask'].to(device), batch_test['labels'].to(device).unsqueeze(1)
                            outputs_test = model(input_ids=ids_test, attention_mask=mask_test)
                            current_best_model_test_preds_list.append(torch.sigmoid(outputs_test.detach()))
                            current_best_model_test_labels_list.append(tgts_test.detach())

                    current_best_model_test_preds_tensor = torch.cat(current_best_model_test_preds_list)
                    current_best_model_test_labels_tensor = torch.cat(current_best_model_test_labels_list)
                    current_best_model_test_preds_np = current_best_model_test_preds_tensor.cpu().numpy().flatten()
                    current_best_model_test_labels_np = current_best_model_test_labels_tensor.cpu().numpy().flatten()

                    test_preds_df_current_best = pd.DataFrame({'true_labels': current_best_model_test_labels_np, 'pred_probabilities': current_best_model_test_preds_np})
                    test_preds_df_current_best.to_csv(BEST_TEST_PREDS_SAVE_PATH, index=False) # Overwrites
                    logging.info(f"Test set predictions for current best model (from Val Epoch {epoch+1}) saved to {BEST_TEST_PREDS_SAVE_PATH}")

                    metrics_ci_test_df_current_best = calculate_metrics_with_ci(current_best_model_test_labels_np, current_best_model_test_preds_np)
                    metrics_ci_test_df_current_best.to_csv(METRICS_CI_TEST_SAVE_PATH, index=False) # Overwrites
                    logging.info(f"Metrics with 95% CI for test set (current best model from Val Epoch {epoch+1}) saved to {METRICS_CI_TEST_SAVE_PATH}")
                    logging.info(f"Current Best Model (from Val Epoch {epoch+1}) Test Set Metrics (with 95% CI):\n" + metrics_ci_test_df_current_best.to_string())

                    current_best_model_test_metrics_basic = compute_metrics(current_best_model_test_preds_tensor, current_best_model_test_labels_tensor)
                    cm_data_test_current_best = np.array([[current_best_model_test_metrics_basic.get('tn',0), current_best_model_test_metrics_basic.get('fp',0)],
                                               [current_best_model_test_metrics_basic.get('fn',0), current_best_model_test_metrics_basic.get('tp',0)]])
                    plt.figure(figsize=(8,6)); sns.heatmap(cm_data_test_current_best, annot=True, fmt='d', cmap='Greens', xticklabels=['Pred Neg', 'Pred Pos'], yticklabels=['Actual Neg', 'Actual Pos'])
                    auprc_test_display_current_best = current_best_model_test_metrics_basic.get('auprc', float('nan'))
                    plt.title(f'CM Test (Best Val Ep {epoch+1} - AUPRC: {auprc_test_display_current_best:.4f})'); plt.tight_layout()
                    plt.savefig(FIG_PATH/f"confusion_matrix_current_best_model_test.png"); plt.close() # Overwrites
                    logging.info(f"Confusion matrix for current best model on test set saved to {FIG_PATH/'confusion_matrix_current_best_model_test.png'}")
            else: # AUPRC did not improve
                epochs_no_improve += 1

            if early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
                logging.info(f"Early stopping at Epoch {epoch+1} (Val AUPRC no improvement for {early_stopping_patience} epochs).")
        else: # No validation dataloader
            logging.info(f"Epoch {epoch+1} completed. No validation set for early stopping.")

        # End-of-Epoch Test Set Evaluation (always performed if test_dataloader exists)
        if test_dataloader:
            model.eval(); total_test_loss_epoch = 0; all_test_preds_epoch_eoe, all_test_labels_epoch_eoe = [], []
            test_progress_bar_eoe = tqdm(test_dataloader, desc=f"Epoch {epoch+1} [EOE Test]", leave=False) # EOE = End Of Epoch
            with torch.no_grad():
                for batch in test_progress_bar_eoe:
                    ids, mask, tgts = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device).unsqueeze(1)
                    outputs = model(input_ids=ids, attention_mask=mask)
                    loss = criterion(outputs, tgts) # Use same criterion for test loss
                    total_test_loss_epoch += loss.item()
                    all_test_preds_epoch_eoe.append(torch.sigmoid(outputs.detach()))
                    all_test_labels_epoch_eoe.append(tgts.detach())
                    test_progress_bar_eoe.set_postfix({'test_loss': f'{loss.item():.4f}'})

            avg_test_loss_epoch = total_test_loss_epoch / len(test_dataloader) if len(test_dataloader) > 0 else 0
            test_metrics_epoch_eoe = compute_metrics(torch.cat(all_test_preds_epoch_eoe), torch.cat(all_test_labels_epoch_eoe))
            current_epoch_metrics.update({'avg_test_loss': avg_test_loss_epoch, **{f'test_{k}': v for k,v in test_metrics_epoch_eoe.items()}})
            logging.info(f"Epoch {epoch+1} (End of Epoch Eval) - Test Loss: {avg_test_loss_epoch:.4f}, Test F1: {test_metrics_epoch_eoe.get('f1',0):.4f}, Test AUPRC: {test_metrics_epoch_eoe.get('auprc',0):.4f}")

        all_epoch_metrics_run.append(current_epoch_metrics)
        if val_dataloader and early_stopping_patience > 0 and epochs_no_improve >= early_stopping_patience:
            break # Break from training loop due to early stopping

    # --- After Training Loop ---
    # If no validation was done, save the model from the last trained epoch.
    # If validation was done, the best model (weights & LoRA) was already saved/overwritten during the loop.
    # This block handles the case where no validation is performed.
    if not val_dataloader and model: # Only save last epoch if no validation was performed at all
        logging.info("No validation set was used. Saving model from the last trained epoch.")
        last_epoch_state_to_save = {'fusion_module_state_dict': model.fusion_module.state_dict(), 'classifier_head_state_dict': model.classifier.state_dict()}
        torch.save(last_epoch_state_to_save, fusion_model_save_path)
        logging.info(f"Model from last trained epoch saved to {fusion_model_save_path}")
        if use_lora and PEFT_AVAILABLE:
            model.base_model.save_pretrained(str(lora_adapter_save_path))
            logging.info(f"LoRA adapters from last trained epoch saved to {lora_adapter_save_path}")
    elif not best_model_state_in_memory and model : # Should not happen if val_dataloader exists and runs for at least one epoch
         logging.warning("A best model was not identified during validation, but training completed. Saving model from last epoch as fallback.")
         last_epoch_state_to_save = {'fusion_module_state_dict': model.fusion_module.state_dict(), 'classifier_head_state_dict': model.classifier.state_dict()}
         torch.save(last_epoch_state_to_save, fusion_model_save_path)
         if use_lora and PEFT_AVAILABLE: model.base_model.save_pretrained(str(lora_adapter_save_path))
    elif not model:
        logging.warning("No model object available after training loop. Nothing to save.")


    if all_epoch_metrics_run:
        try:
            def convert_nan(o): return None if isinstance(o, float) and np.isnan(o) else o # JSON cannot serialize NaN
            with open(EPOCH_METRICS_SAVE_PATH, 'w') as f:
                json.dump(all_epoch_metrics_run, f, indent=4, default=convert_nan)
            logging.info(f"All epoch metrics saved to {EPOCH_METRICS_SAVE_PATH}")
        except Exception as e: logging.error(f"Failed to save epoch metrics: {e}")


# ───── Text Embedding ───────────────────────────────────
def embed_text_column(series: pd.Series, model_name: str = EMBED_MODEL, batch_size: int = 16,
                      max_length: int = MAX_SEQ_LENGTH, num_layers_to_fuse: int = NUM_FUSION_LAYERS,
                      trained_fusion_and_classifier_weights_path: Optional[Path] = None, # Path to .pth for fusion/classifier
                      trained_lora_adapter_weights_path: Optional[Path] = None # Path to LoRA adapter directory
                      ) -> pd.DataFrame:
    # Use global paths if specific paths are not provided
    actual_fusion_weights_path = TRAINED_FUSION_PATH if trained_fusion_and_classifier_weights_path is None else trained_fusion_and_classifier_weights_path
    actual_lora_adapter_path = TRAINED_LORA_ADAPTER_PATH if trained_lora_adapter_weights_path is None else trained_lora_adapter_weights_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Loading base model {model_name} to {device} for embedding generation...")
    base_model_for_embedding = AutoModel.from_pretrained(model_name, output_hidden_states=True)

    # Load LoRA adapters if available and path exists
    if actual_lora_adapter_path and actual_lora_adapter_path.exists() and PEFT_AVAILABLE:
        try:
            if actual_lora_adapter_path.is_dir() and any(actual_lora_adapter_path.iterdir()): # Check if dir and not empty
                base_model_for_embedding = PeftModel.from_pretrained(base_model_for_embedding, str(actual_lora_adapter_path))
                logging.info(f"Loaded LoRA adapters from {actual_lora_adapter_path} for embedding.")
            elif not actual_lora_adapter_path.is_dir():
                 logging.warning(f"LoRA path {actual_lora_adapter_path} for embedding is not a directory. Using base model without LoRA.")
            else: # Is a directory but empty
                 logging.warning(f"LoRA directory {actual_lora_adapter_path} for embedding is empty. Using base model without LoRA.")
        except Exception as e: logging.error(f"Error loading LoRA from {actual_lora_adapter_path} for embedding: {e}. Using base model.")
    elif actual_lora_adapter_path: logging.warning(f"LoRA path {actual_lora_adapter_path} for embedding not found, or PEFT unavailable. Using base model without LoRA.")
    base_model_for_embedding.to(device).eval()

    # Initialize and load fusion module weights
    layer_fusion_for_embedding = AdaptiveLayerFusion(base_model_for_embedding.config.hidden_size, num_layers_to_fuse, dropout=0.0).to(device) # Dropout 0 for inference
    if actual_fusion_weights_path and actual_fusion_weights_path.exists():
        try:
            checkpoint = torch.load(actual_fusion_weights_path, map_location=device)
            if 'fusion_module_state_dict' in checkpoint: # Check if it's the combined dict
                layer_fusion_for_embedding.load_state_dict(checkpoint['fusion_module_state_dict'])
                logging.info(f"Loaded fusion weights from 'fusion_module_state_dict' in {actual_fusion_weights_path} for embedding.")
            else: # Assume it's just the fusion module state dict
                layer_fusion_for_embedding.load_state_dict(checkpoint)
                logging.info(f"Loaded fusion weights directly from {actual_fusion_weights_path} for embedding (assumed no classifier dict).")
        except Exception as e: logging.error(f"Error loading fusion weights from {actual_fusion_weights_path} for embedding: {e}. Using initialized fusion module.")
    else: logging.warning(f"Fusion weights path {actual_fusion_weights_path} for embedding not found or None. Using initialized fusion module.")
    layer_fusion_for_embedding.eval()


    logging.info(f"Generating embeddings for {len(series)} texts, fusing top {num_layers_to_fuse} layers using loaded model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    texts_list = series.fillna("unknown").astype(str).tolist() # Handle NaNs and ensure string type
    all_embs, all_weights = [], []

    for i in tqdm(range(0, len(texts_list), batch_size), desc="Generating Embeddings", leave=False):
        batch_texts = texts_list[i:i+batch_size]
        try:
            encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = base_model_for_embedding(**encoded_input, output_hidden_states=True)
                hidden_states_all = list(outputs.hidden_states)

                # Validate num_layers_to_fuse for embedding model
                actual_num_tf_layers_emb = len(hidden_states_all) - 1 # Exclude initial embedding layer
                if not (0 < num_layers_to_fuse <= actual_num_tf_layers_emb):
                    logging.warning(f"num_layers_to_fuse ({num_layers_to_fuse}) is invalid for embedding model with {actual_num_tf_layers_emb} TF layers. Clamping to [1, {actual_num_tf_layers_emb}]")
                    fuse_n_emb = max(1, min(num_layers_to_fuse, actual_num_tf_layers_emb))
                    if actual_num_tf_layers_emb == 0 : fuse_n_emb = 0 # Edge case: no transformer layers
                else:
                    fuse_n_emb = num_layers_to_fuse

                if fuse_n_emb == 0: # If no layers to fuse (e.g. model has no transformer layers)
                    logging.error("Embedding: No transformer layers to fuse from base model. Returning zeros.")
                    # Create zero embeddings and uniform weights as fallback
                    pooled_embs_batch = torch.zeros((len(batch_texts), base_model_for_embedding.config.hidden_size), device=device)
                    weights_tensor_batch = torch.ones((len(batch_texts),1), device=device) # Dummy weights
                else:
                    selected_hidden_states_emb = hidden_states_all[-fuse_n_emb:]
                    pooled_embs_batch, weights_tensor_batch = layer_fusion_for_embedding(selected_hidden_states_emb, encoded_input['attention_mask'])

            all_embs.append(pooled_embs_batch.cpu().numpy())
            all_weights.append(weights_tensor_batch.cpu().numpy())
        except Exception as e:
            logging.error(f"Error in embedding generation batch {i//batch_size + 1}: {e}", exc_info=True)
            # Fallback for this batch: zero embeddings and uniform weights
            emb_dim_fallback = base_model_for_embedding.config.hidden_size
            fallback_num_layers = fuse_n_emb if 'fuse_n_emb' in locals() and fuse_n_emb > 0 else 1
            if fallback_num_layers == 0: fallback_num_layers = 1 # Ensure at least 1 for division
            all_embs.append(np.zeros((len(batch_texts), emb_dim_fallback)))
            all_weights.append(np.ones((len(batch_texts), fallback_num_layers)) / fallback_num_layers)


    if not all_embs: logging.warning("No embeddings were generated."); return pd.DataFrame()
    all_embs_arr = np.vstack(all_embs); all_weights_arr = np.vstack(all_weights)
    if all_embs_arr.shape[0] == 0: logging.warning("Embeddings array is empty after vstack."); return pd.DataFrame()

    # Normalize embeddings (L2 norm)
    norms = np.linalg.norm(all_embs_arr, axis=1, keepdims=True)
    norm_embs = all_embs_arr / np.maximum(norms, 1e-10) # Avoid division by zero
    emb_df = pd.DataFrame(norm_embs, columns=[f"text_emb_{j}" for j in range(norm_embs.shape[1])], index=series.index)

    res_df = emb_df
    if all_weights_arr.size > 0 and all_weights_arr.shape[1] > 0 : # Check if weights were actually produced and have columns
        weights_df = pd.DataFrame(all_weights_arr, columns=[f"layer_weight_{k}" for k in range(all_weights_arr.shape[1])], index=series.index)
        res_df = pd.concat([emb_df, weights_df], axis=1)
        logging.info(f"Average layer weights from embedding: {all_weights_arr.mean(axis=0)}")
    else: logging.warning("Layer weights array for embeddings was empty or had no columns.")

    logging.info(f"Text embedding generation complete. Final shape: {res_df.shape}")
    return res_df

# ───── Visualization ────────────────────────────────────────────
def quick_plots(df: pd.DataFrame, target_col_name: str):
    if target_col_name not in df.columns:
        logging.warning(f"Target column '{target_col_name}' not found in DataFrame for plotting."); return
    try:
        # Target distribution bar plot
        plt.figure(figsize=(6,4)); sns.countplot(x=df[target_col_name])
        plt.title(f"Distribution of {target_col_name}"); plt.tight_layout()
        plt.savefig(FIG_PATH/f"{target_col_name.replace('/','_')}_dist.png"); plt.close()

        # Target distribution pie chart
        plt.figure(figsize=(8,6)); counts = df[target_col_name].value_counts()
        if not counts.empty:
            plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue','salmon'])
            plt.title(f'{target_col_name} Pie Chart'); plt.axis('equal'); plt.tight_layout()
            plt.savefig(FIG_PATH/f"{target_col_name.replace('/','_')}_pie.png"); plt.close()
        else:
            logging.warning(f"No data for pie chart of {target_col_name}")
        logging.info(f"Target distribution plots saved to {FIG_PATH}")
    except Exception as e: logging.error(f"Error during target column plotting for '{target_col_name}': {e}", exc_info=True)

    # Missing values plot (top 20 features)
    miss_percent = df.isna().mean() * 100
    if not miss_percent.empty:
        miss_head = miss_percent[miss_percent > 0].sort_values(ascending=False).head(20)
        if not miss_head.empty:
            try:
                plt.figure(figsize=(10,8)); miss_head.plot(kind="barh")
                plt.title("Top 20 Features with Missing Values (%)"); plt.xlabel("Percentage Missing (%)"); plt.tight_layout()
                plt.savefig(FIG_PATH/"missing_values_top20.png"); plt.close()
                logging.info(f"Missing values plot saved to {FIG_PATH/'missing_values_top20.png'}")
            except Exception as e: logging.error(f"Error during missing values plot: {e}", exc_info=True)
        else: logging.info("No features with missing values to plot.")
    else: logging.info("DataFrame has no columns or no missing values to plot statistics for.")


# ───── Main Process ────────────────────────────────────────────
def main(args): # `args` comes from argparse
    # Make global variables modifiable within this function scope if they are reassigned
    global EMBED_MODEL, TEXT_COL, NUM_FUSION_LAYERS, INPUT_PATH, EMBED_PATH, TARGET_COL, NUM_EPOCHS, LEARNING_RATE, TRAIN_BATCH_SIZE, MIN_SAMPLES_FOR_TOP_LAYER_FINETUNING, EARLY_STOPPING_PATIENCE, VALIDATION_SPLIT_RATIO_DEFAULT, TRAINED_FUSION_PATH, TRAINED_LORA_ADAPTER_PATH, GLOBAL_SEED, USE_LORA, LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGET_MODULES, FIG_PATH, EPOCH_METRICS_SAVE_PATH, LOG_FILE_PATH, HYPERPARAMS_FILE_PATH, RUN_SPECIFIC_DIR, BEST_VAL_PREDS_SAVE_PATH, METRICS_CI_VAL_SAVE_PATH, BEST_TEST_PREDS_SAVE_PATH, METRICS_CI_TEST_SAVE_PATH, N_BOOTSTRAPS_CI, TEST_SPLIT_RATIO_CONST

    set_seed(GLOBAL_SEED) # Set seed at the very beginning

    # --- Setup Run-Specific Output Directory and Paths ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized_model_name = EMBED_MODEL.replace("/", "_").replace("-", "_") # Sanitize for directory name
    RUN_SPECIFIC_DIR = BASE_RESULTS_DIR / f"{sanitized_model_name}_{timestamp_str}"
    RUN_SPECIFIC_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"All outputs will be saved to: {RUN_SPECIFIC_DIR.resolve()}")

    FIG_PATH = RUN_SPECIFIC_DIR / "figures"
    FIG_PATH.mkdir(parents=True, exist_ok=True)

    EMBED_PATH = RUN_SPECIFIC_DIR / args.output_csv_basename # Final output CSV
    TRAINED_FUSION_PATH = RUN_SPECIFIC_DIR / args.fusion_weights_basename # Fusion/classifier weights
    TRAINED_LORA_ADAPTER_PATH = RUN_SPECIFIC_DIR / args.lora_adapter_dir_basename # LoRA adapters directory
    if USE_LORA: TRAINED_LORA_ADAPTER_PATH.mkdir(parents=True, exist_ok=True) # Create LoRA dir if using LoRA

    EPOCH_METRICS_SAVE_PATH = FIG_PATH / "all_epoch_metrics.json" # Metrics per epoch
    LOG_FILE_PATH = RUN_SPECIFIC_DIR / "run_log.log" # File log
    HYPERPARAMS_FILE_PATH = RUN_SPECIFIC_DIR / "hyperparameters.json" # Hyperparameters log

    BEST_VAL_PREDS_SAVE_PATH = RUN_SPECIFIC_DIR / "best_validation_set_predictions.csv"
    METRICS_CI_VAL_SAVE_PATH = RUN_SPECIFIC_DIR / "best_validation_metrics_with_ci.csv"
    BEST_TEST_PREDS_SAVE_PATH = RUN_SPECIFIC_DIR / "best_model_test_set_predictions.csv"
    METRICS_CI_TEST_SAVE_PATH = RUN_SPECIFIC_DIR / "best_model_test_metrics_with_ci.csv"

    # --- Setup File Logging ---
    file_handler = logging.FileHandler(LOG_FILE_PATH)
    file_handler.setLevel(logging.INFO) # Or logging.DEBUG for more verbosity in file
    root_logger = logging.getLogger()
    if root_logger.hasHandlers() and root_logger.handlers[0].formatter: # Use existing formatter if available
        formatter = root_logger.handlers[0].formatter
        file_handler.setFormatter(formatter)
    else: # Fallback formatter
        fallback_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(fallback_formatter)
    root_logger.addHandler(file_handler)
    logging.info(f"Logging to console and to file: {LOG_FILE_PATH.resolve()}")

    # --- Log Hyperparameters ---
    actual_validation_split_for_logging = args.validation_split_ratio if args.validation_split_ratio > 0 else 0.0

    hyperparameters_to_save = {
        "SCRIPT_VERSION": SCRIPT_VERSION, "TIMESTAMP": timestamp_str,
        "GLOBAL_SEED": GLOBAL_SEED, "EMBED_MODEL": EMBED_MODEL, "TEXT_COL": TEXT_COL, "TARGET_COL": TARGET_COL,
        "NUM_FUSION_LAYERS": NUM_FUSION_LAYERS, "NUM_EPOCHS": NUM_EPOCHS, "LEARNING_RATE": LEARNING_RATE,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE, "MAX_SEQ_LENGTH": MAX_SEQ_LENGTH,
        "MIN_SAMPLES_FOR_TOP_LAYER_FINETUNING": MIN_SAMPLES_FOR_TOP_LAYER_FINETUNING,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "TEST_SPLIT_RATIO_FIXED": TEST_SPLIT_RATIO_CONST,
        "VALIDATION_SPLIT_RATIO_FROM_REMAINDER_ARG": args.validation_split_ratio, # The arg value
        "EFFECTIVE_VALIDATION_SPLIT_OVERALL": (1.0 - TEST_SPLIT_RATIO_CONST) * actual_validation_split_for_logging if actual_validation_split_for_logging > 0 else 0.0,
        "EFFECTIVE_TRAIN_SPLIT_OVERALL": (1.0 - TEST_SPLIT_RATIO_CONST) * (1.0 - actual_validation_split_for_logging) if actual_validation_split_for_logging > 0 else (1.0 - TEST_SPLIT_RATIO_CONST),
        "N_BOOTSTRAPS_CI": N_BOOTSTRAPS_CI,
        "USE_LORA": USE_LORA, "LORA_R": LORA_R, "LORA_ALPHA": LORA_ALPHA, "LORA_DROPOUT": LORA_DROPOUT,
        "LORA_TARGET_MODULES": LORA_TARGET_MODULES,
        "INPUT_CSV_PATH": str(INPUT_PATH.resolve()), # Store resolved paths
        "RUN_SPECIFIC_OUTPUT_DIR": str(RUN_SPECIFIC_DIR.resolve()),
        "OUTPUT_CSV_BASENAME_ARG": args.output_csv_basename,
        "FUSION_WEIGHTS_BASENAME_ARG": args.fusion_weights_basename,
        "LORA_ADAPTER_DIR_BASENAME_ARG": args.lora_adapter_dir_basename,
        "EMBED_CSV_FULL_PATH": str(EMBED_PATH.resolve()),
        "FIGURES_DIR_FULL_PATH": str(FIG_PATH.resolve()),
        "TRAINED_FUSION_WEIGHTS_FULL_PATH": str(TRAINED_FUSION_PATH.resolve()),
        "TRAINED_LORA_ADAPTER_DIR_FULL_PATH": str(TRAINED_LORA_ADAPTER_PATH.resolve()),
        "EPOCH_METRICS_FULL_PATH": str(EPOCH_METRICS_SAVE_PATH.resolve()),
        "LOG_FILE_FULL_PATH": str(LOG_FILE_PATH.resolve()),
        "BEST_VAL_PREDS_SAVE_FULL_PATH": str(BEST_VAL_PREDS_SAVE_PATH.resolve()),
        "METRICS_CI_VAL_SAVE_FULL_PATH": str(METRICS_CI_VAL_SAVE_PATH.resolve()),
        "BEST_TEST_PREDS_SAVE_FULL_PATH": str(BEST_TEST_PREDS_SAVE_PATH.resolve()),
        "METRICS_CI_TEST_SAVE_FULL_PATH": str(METRICS_CI_TEST_SAVE_PATH.resolve()),
    }
    try:
        with open(HYPERPARAMS_FILE_PATH, 'w') as f:
            json.dump(hyperparameters_to_save, f, indent=4)
        logging.info(f"Hyperparameters saved to {HYPERPARAMS_FILE_PATH.resolve()}")
    except Exception as e:
        logging.error(f"Failed to save hyperparameters: {e}", exc_info=True)


    logging.info("================ Configuration (from globals after args) ================")
    for key, value in hyperparameters_to_save.items():
        if "PATH" not in key.upper() and "DIR" not in key.upper(): # Don't print full paths again here
             logging.info(f"{key}: {value}")
    logging.info(f"Run output directory (resolved): {RUN_SPECIFIC_DIR.resolve()}")
    logging.info("=======================================================================")


    if USE_LORA and not PEFT_AVAILABLE:
        logging.error("LoRA requested but PEFT library not available. Exiting.")
        return None # Or raise SystemExit

    logging.info("Step 0: Initial data loading and preprocessing")
    if not INPUT_PATH.exists():
        logging.error(f"Input CSV {INPUT_PATH} not found."); return None
    try:
        df = pd.read_csv(INPUT_PATH)
    except Exception as e:
        logging.error(f"Error reading {INPUT_PATH}: {e}", exc_info=True); return None
    logging.info(f"Initial df shape: {df.shape}")

    if TARGET_COL not in df.columns or TEXT_COL not in df.columns:
        logging.error(f"Missing target ('{TARGET_COL}') or text ('{TEXT_COL}') column."); return None

    # Drop rows where essential target or text columns are NaN
    df.dropna(subset=[TARGET_COL, TEXT_COL], inplace=True)
    try:
        df[TARGET_COL] = df[TARGET_COL].astype(int) # Ensure target is integer
    except ValueError as e:
        logging.error(f"Cannot convert target column '{TARGET_COL}' to int: {e}", exc_info=True); return None

    if df.empty:
        logging.error("DataFrame is empty after dropping NaNs from target/text columns."); return None
    logging.info(f"Shape after NaN drop in target/text: {df.shape}. Target dist:\n{df[TARGET_COL].value_counts(normalize=True)}")

    # Prepare data for the training pipeline
    texts_for_training_pipeline, labels_for_training_pipeline = df[TEXT_COL].astype(str).tolist(), df[TARGET_COL].tolist()
    if not texts_for_training_pipeline: # Should be redundant if df is not empty
        logging.error("No text samples available for the training pipeline."); return None

    # Prepare LoRA config for training function
    lora_target_modules_list_main = [m.strip() for m in LORA_TARGET_MODULES.split(',') if m.strip()] if isinstance(LORA_TARGET_MODULES, str) else (LORA_TARGET_MODULES if isinstance(LORA_TARGET_MODULES, list) else [])
    lora_config_dict_for_training = {"r": LORA_R, "lora_alpha": LORA_ALPHA, "lora_dropout": LORA_DROPOUT, "target_modules": lora_target_modules_list_main}

    try:
        # Test dataloader is now created within train_and_save_fusion_model
        train_and_save_fusion_model(
            texts=texts_for_training_pipeline,
            labels=labels_for_training_pipeline,
            base_model_name=EMBED_MODEL,
            num_fusion_layers_to_train=NUM_FUSION_LAYERS,
            fusion_model_save_path=TRAINED_FUSION_PATH, # For fusion/classifier weights
            lora_adapter_save_path=TRAINED_LORA_ADAPTER_PATH, # For LoRA adapters
            use_lora=USE_LORA,
            lora_config_dict=lora_config_dict_for_training,
            validation_split_ratio=args.validation_split_ratio, # From CLI args
            tokenizer_max_len=MAX_SEQ_LENGTH,
            train_batch_size=TRAIN_BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            seed=GLOBAL_SEED,
            test_dataloader_main=None # Explicitly None, will be created inside
        )
    except Exception as e:
        logging.error(f"Critical error during training pipeline: {e}", exc_info=True)
        logging.warning("Attempting to proceed to embedding generation with pre-existing/initialized weights due to training error.")

    logging.info("Step 1: Data cleaning for final embedding DataFrame (using original full df)")
    df_for_embedding_generation = df.copy() # Use the original df (after initial NaN drop) for embedding generation

    # Drop specified columns, except target and always_keep columns
    cols_to_drop_for_final_csv = [c for c in DROP_COLS if c != TARGET_COL and c in df_for_embedding_generation.columns]
    if cols_to_drop_for_final_csv:
        df_for_embedding_generation.drop(columns=cols_to_drop_for_final_csv, errors="ignore", inplace=True)

    df_for_embedding_generation.dropna(axis=1, how='all', inplace=True) # Drop fully empty columns
    logging.info(f"Shape of DataFrame prepared for embedding generation: {df_for_embedding_generation.shape}")

    logging.info("Step 2: Text embedding generation for the output CSV")
    if TEXT_COL in df_for_embedding_generation.columns:
        lora_path_for_final_embed = None
        if USE_LORA and TRAINED_LORA_ADAPTER_PATH.exists() and TRAINED_LORA_ADAPTER_PATH.is_dir() and any(TRAINED_LORA_ADAPTER_PATH.iterdir()):
            lora_path_for_final_embed = TRAINED_LORA_ADAPTER_PATH
            logging.info(f"Using saved LoRA adapters from {TRAINED_LORA_ADAPTER_PATH} for final embedding generation.")
        elif USE_LORA:
             logging.warning(f"LoRA was set to True, but adapters at {TRAINED_LORA_ADAPTER_PATH} are not found or empty for final embedding. Embedding will proceed without these specific LoRA weights.")

        text_embeddings_df_final = embed_text_column(
            df_for_embedding_generation[TEXT_COL].astype(str), # Ensure text column is string
            model_name=EMBED_MODEL,
            num_layers_to_fuse=NUM_FUSION_LAYERS,
            trained_fusion_and_classifier_weights_path=TRAINED_FUSION_PATH, # Use the path to the saved .pth
            trained_lora_adapter_weights_path=lora_path_for_final_embed # Use the path to the saved LoRA adapters
        )
        if not text_embeddings_df_final.empty:
            if TEXT_COL in df_for_embedding_generation.columns: # Drop original text column
                df_for_embedding_generation.drop(columns=[TEXT_COL], inplace=True)
            df_for_embedding_generation = pd.concat([
                df_for_embedding_generation.reset_index(drop=True), # Reset index for clean concat
                text_embeddings_df_final.reset_index(drop=True)
            ], axis=1)
            logging.info(f"Shape of DataFrame after adding embeddings for final output: {df_for_embedding_generation.shape}")
        else: logging.warning("Text embedding for final output CSV returned an empty DataFrame.")
    else: logging.warning(f"Text column '{TEXT_COL}' not in DataFrame for final embedding generation. Skipping embedding step for output CSV.")

    quick_plots(df_for_embedding_generation, TARGET_COL) # Generate plots on the final df

    # Save the final DataFrame
    EMBED_PATH.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
    df_for_embedding_generation.to_csv(EMBED_PATH, index=False)
    logging.info(f"Final embedded data (potentially with features) saved to {EMBED_PATH.resolve()}")
    logging.info(f"All results for this run are in: {RUN_SPECIFIC_DIR.resolve()}")
    logging.info("Script finished successfully.")
    return df_for_embedding_generation


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MIMIC-IV Text Embedding with Adaptive Fusion & LoRA. Saves all outputs to a run-specific directory.")

    # File/Path Arguments
    parser.add_argument("--input-csv", type=str, default=str(INPUT_PATH), help=f"Path to input CSV. Default: {INPUT_PATH}")
    parser.add_argument("--output-csv-basename", type=str, default="embedded_data.csv", help="Basename for the output CSV with embeddings. Default: embedded_data.csv")
    parser.add_argument("--fusion-weights-basename", type=str, default="trained_model_weights.pth", help="Basename for trained fusion/classifier weights. Default: trained_model_weights.pth")
    parser.add_argument("--lora-adapter-dir-basename", type=str, default="lora_adapters", help="Basename for LoRA adapters directory. Default: lora_adapters")

    # Model & Data Column Arguments
    parser.add_argument("--embed-model", type=str, default=EMBED_MODEL, help=f"Base HuggingFace model for embedding. Default: {EMBED_MODEL}")
    parser.add_argument("--text-col", type=str, default=TEXT_COL, help=f"Name of the text column in the input CSV. Default: {TEXT_COL}")
    parser.add_argument("--target-col", type=str, default=TARGET_COL, help=f"Name of the target column in the input CSV. Default: {TARGET_COL}")

    # Training Hyperparameter Arguments
    parser.add_argument("--num-layers", type=int, default=NUM_FUSION_LAYERS, help=f"Number of top transformer layers to fuse. Default: {NUM_FUSION_LAYERS}")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help=f"Number of training epochs. Default: {NUM_EPOCHS}")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help=f"Learning rate for training. Default: {LEARNING_RATE}")
    parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE, help=f"Training batch size. Default: {TRAIN_BATCH_SIZE}")
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH, help=f"Maximum sequence length for tokenizer. Default: {MAX_SEQ_LENGTH}")

    # Advanced Training Arguments
    parser.add_argument("--min-samples-tune", type=int, default=MIN_SAMPLES_FOR_TOP_LAYER_FINETUNING, help=f"Minimum samples to fine-tune top layers of base model (if not using LoRA). Default: {MIN_SAMPLES_FOR_TOP_LAYER_FINETUNING}")
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE, help=f"Patience for early stopping based on validation AUPRC. 0 to disable. Default: {EARLY_STOPPING_PATIENCE}")
    parser.add_argument("--validation-split-ratio", type=float, default=VALIDATION_SPLIT_RATIO_DEFAULT,
                        help=f"Proportion of (Train+Val) data to use for validation (after {TEST_SPLIT_RATIO_CONST*100:.1f}% test split). 0 for no validation. Default: {VALIDATION_SPLIT_RATIO_DEFAULT:.4f} (for ~12.5% overall val).")

    # Reproducibility & CI Arguments
    parser.add_argument("--global-seed", type=int, default=GLOBAL_SEED, help=f"Global random seed for reproducibility. Default: {GLOBAL_SEED}")
    parser.add_argument("--n-bootstraps-ci", type=int, default=N_BOOTSTRAPS_CI, help=f"Number of bootstrap samples for CI calculation. Default: {N_BOOTSTRAPS_CI}")

    # LoRA Specific Arguments
    parser.add_argument("--use-lora", action='store_true', default=USE_LORA_DEFAULT, help="Enable LoRA for fine-tuning the base model. Default: False")
    parser.add_argument("--lora-r", type=int, default=LORA_R_DEFAULT, help=f"LoRA r (rank). Default: {LORA_R_DEFAULT}")
    parser.add_argument("--lora-alpha", type=int, default=LORA_ALPHA_DEFAULT, help=f"LoRA alpha. Default: {LORA_ALPHA_DEFAULT}")
    parser.add_argument("--lora-dropout", type=float, default=LORA_DROPOUT_DEFAULT, help=f"LoRA dropout. Default: {LORA_DROPOUT_DEFAULT}")
    parser.add_argument("--lora-target-modules", type=str, default=LORA_TARGET_MODULES_DEFAULT, help=f"LoRA target modules (comma-separated, e.g., 'query,key,value'). Default: '{LORA_TARGET_MODULES_DEFAULT}'")

    args = parser.parse_args()

    # Update global variables from parsed arguments
    INPUT_PATH = Path(args.input_csv)
    EMBED_MODEL = args.embed_model
    TEXT_COL = args.text_col
    TARGET_COL = args.target_col
    GLOBAL_SEED = args.global_seed
    N_BOOTSTRAPS_CI = args.n_bootstraps_ci
    NUM_FUSION_LAYERS = args.num_layers
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    TRAIN_BATCH_SIZE = args.train_batch_size
    MAX_SEQ_LENGTH = args.max_seq_length
    MIN_SAMPLES_FOR_TOP_LAYER_FINETUNING = args.min_samples_tune
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    # VALIDATION_SPLIT_RATIO_DEFAULT is updated by args.validation_split_ratio in main() logic

    USE_LORA = args.use_lora
    LORA_R = args.lora_r
    LORA_ALPHA = args.lora_alpha
    LORA_DROPOUT = args.lora_dropout
    LORA_TARGET_MODULES = args.lora_target_modules

    # Validate validation_split_ratio
    if not (0 <= args.validation_split_ratio < 1): # Must be in [0, 1)
        logging.error(f"Validation split ratio (from remainder) must be [0, 1). Got: {args.validation_split_ratio}")
        exit(1) # Or raise ValueError

    if args.validation_split_ratio == 0:
        logging.warning(f"Validation split ratio is 0. No validation set will be created from the train/val portion. Test set is still {TEST_SPLIT_RATIO_CONST*100:.1f}%. Early stopping disabled. Model from last epoch saved. Best model on test set will be based on this last epoch model.")
    elif args.early_stopping_patience <= 0 and args.validation_split_ratio > 0 : # Val split exists but no patience
        logging.warning("Validation split > 0 but early stopping patience <=0. Early stopping effectively disabled (will run for all epochs or until AUPRC improves, but best model choice still uses validation AUPRC).")

    main(args)
