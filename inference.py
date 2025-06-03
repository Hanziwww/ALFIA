import logging
from pathlib import Path
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional, Any, Callable
from tqdm import tqdm
import json
import random
from datetime import datetime
import time # Added for performance timing
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, precision_recall_curve, auc
)

# Import PEFT libraries
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning("PEFT library not found. LoRA functionality will be disabled if models require it.")

# ───── logging setup ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler()])
logging.info(f"Inference Script Version: {SCRIPT_VERSION}")
logging.info("Changelog:\n" + CHANGELOG)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' # Or your preferred mirror

# ───── Default Values (many will be overridden by loaded hyperparameters) ───
DEFAULT_MAX_SEQ_LENGTH = 512
DEFAULT_BATCH_SIZE = 8 # Batch size for inference
GLOBAL_SEED = 42 # Default seed, can be overridden by loaded hyperparams if stored
DEFAULT_BASE_RESULTS_PATH = "results" # Default parent directory for training runs
N_BOOTSTRAPS_DEFAULT = 1000 # Default number of bootstrap samples for CI calculation

# ───── Reproducibility Function ─────────────────────────────────────────
def set_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    logging.info(f"Global seed set to {seed_value}")

# ─────（Copied from training script - ensure consistency）─────────
class AdaptiveLayerFusion(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int = 4, 
        num_heads: int = 4,
        dropout: float = 0.1, 
        use_layer_gating: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_layer_gating = use_layer_gating
        self.num_layers_to_fuse = num_layers 

        head_dim = 64 
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
                nn.Linear(hidden_size, self.num_layers_to_fuse),
                nn.Sigmoid()
            )

        self.content_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm1 = nn.LayerNorm(hidden_size)

        self.token_attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.GELU(),
            nn.Linear(hidden_size // 4, 1)
        )
        self.token_attention_dropout = nn.Dropout(dropout)

        self.global_context_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

        self.output_projection = nn.Linear(hidden_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        self.register_buffer('last_attention_mask', None, persistent=False)


    def _compute_layer_weights(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        batch_size = hidden_states[0].size(0)
        num_layers_being_fused = len(hidden_states)
        head_dim = 64

        if self.last_attention_mask is not None:
            attention_mask = self.last_attention_mask
            mask_expanded = attention_mask.unsqueeze(-1).float()
            avg_pooled_layers = []
            for layer in hidden_states:
                masked_layer = layer * mask_expanded
                seq_lengths = attention_mask.sum(dim=1, keepdim=True)
                avg_pooled = masked_layer.sum(dim=1) / seq_lengths.clamp(min=1)
                avg_pooled_layers.append(avg_pooled)
            avg_reps = torch.stack(avg_pooled_layers, dim=1)
        else:
            logging.warning("last_attention_mask not found in AdaptiveLayerFusion. Using simple mean pooling for layer weights.")
            avg_pooled_layers = [layer.mean(dim=1) for layer in hidden_states]
            avg_reps = torch.stack(avg_pooled_layers, dim=1)

        query_input = avg_reps.mean(dim=1)
        query = self.layer_query_projection(query_input).view(batch_size, self.num_heads, head_dim)
        key = self.layer_key_projection(avg_reps).view(batch_size, num_layers_being_fused, self.num_heads, head_dim)
        value = self.layer_value_projection(avg_reps).view(batch_size, num_layers_being_fused, self.num_heads, head_dim)

        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_scores = torch.matmul(query.unsqueeze(2), key.transpose(-1, -2)) / math.sqrt(head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.layer_attention_dropout(attention_weights)

        context = torch.matmul(attention_weights, value)
        context = context.squeeze(2).contiguous().view(batch_size, self.num_heads * head_dim)
        context = self.layer_attention_output(context)

        if self.use_layer_gating:
            if self.layer_gate[0].out_features != num_layers_being_fused:
                logging.warning(
                    f"AdaptiveLayerFusion's layer_gate was initialized for {self.layer_gate[0].out_features} layers (from training config), "
                    f"but received {num_layers_being_fused} layers in forward pass during inference. "
                    f"This could indicate a mismatch if not intended. Gate will still operate on {self.layer_gate[0].out_features} outputs."
                )

            gate_weights = self.layer_gate(context)
            if gate_weights.shape[1] > num_layers_being_fused:
                logging.warning(f"Gate produced {gate_weights.shape[1]} weights, but only {num_layers_being_fused} layers provided. Slicing weights.")
                gate_weights = gate_weights[:, :num_layers_being_fused]
            elif gate_weights.shape[1] < num_layers_being_fused:
                logging.error(f"CRITICAL: Gate produced {gate_weights.shape[1]} weights, but {num_layers_being_fused} layers provided. This is a mismatch. Padding with zeros.")
                padding = torch.zeros(batch_size, num_layers_being_fused - gate_weights.shape[1], device=context.device)
                gate_weights = torch.cat([gate_weights, padding], dim=1)
            return gate_weights
        else:
            return attention_weights.squeeze(2).mean(dim=1)


    def forward(
        self,
        all_hidden_states: List[torch.Tensor],
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not all_hidden_states:
            raise ValueError("all_hidden_states list cannot be empty.")
        self.last_attention_mask = attention_mask

        layer_weights = self._compute_layer_weights(all_hidden_states)
        
        if layer_weights.shape[1] != len(all_hidden_states):
            logging.error(f"Shape mismatch after _compute_layer_weights: layer_weights has {layer_weights.shape[1]} weights, "
                          f"but {len(all_hidden_states)} hidden states were provided. Using uniform weights as fallback.")
            num_fused_layers = len(all_hidden_states)
            layer_weights = torch.ones(all_hidden_states[0].size(0), num_fused_layers, device=all_hidden_states[0].device) / num_fused_layers

        weighted_states = torch.zeros_like(all_hidden_states[0])
        for i, layer_state in enumerate(all_hidden_states):
            current_layer_weight = layer_weights[:, i].unsqueeze(1).unsqueeze(2)
            weighted_states += current_layer_weight * layer_state
        
        interaction_states = self.layer_interaction(weighted_states)
        projected_states = self.content_projection(weighted_states)
        enhanced_states = self.layer_norm1(weighted_states + projected_states + interaction_states)

        token_scores = self.token_attention(enhanced_states).squeeze(-1)
        token_scores = token_scores.masked_fill(attention_mask == 0, -1e9)
        token_weights = F.softmax(token_scores, dim=1)
        token_weights = self.token_attention_dropout(token_weights)
        local_context = torch.bmm(token_weights.unsqueeze(1), enhanced_states).squeeze(1)

        mask_expanded = attention_mask.unsqueeze(-1).float()
        masked_states = enhanced_states * mask_expanded
        seq_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
        global_context = masked_states.sum(dim=1) / seq_lengths
        global_context = self.global_context_layer(global_context)

        combined_context = torch.cat([local_context, global_context], dim=1)
        pooled_output = self.context_fusion(combined_context)

        projected_output = self.output_projection(pooled_output)
        pooled_output = self.layer_norm2(pooled_output + projected_output)

        return pooled_output, layer_weights

# ───── Attentional Classifier Head (Copied from training script) ───────────
class AttentionalClassifierHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, num_attn_heads: int = 4, ffn_expansion: int = 2, dropout_rate: float = 0.1):
        super().__init__()
        if hidden_size % num_attn_heads != 0:
            valid_heads = [h for h in [1, 2, 4, 8, 12, 16, 24, 32] if hidden_size % h == 0]
            if not valid_heads: raise ValueError(f"Hidden size {hidden_size} not divisible by common head counts.")
            original_num_attn_heads = num_attn_heads
            num_attn_heads = valid_heads[-1]
            logging.warning(f"AttentionalClassifierHead: num_attn_heads ({original_num_attn_heads}) adjusted to {num_attn_heads} for hidden_size {hidden_size}.")
        
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

    def forward(self, fused_embeddings: torch.Tensor) -> torch.Tensor:
        x = fused_embeddings.unsqueeze(1)
        attn_output, _ = self.attention(query=x, key=x, value=x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        x_squeezed = x.squeeze(1)
        return self.output_layer(x_squeezed)

# ───── (Copied and adapted for loading) ──────────────────
class TextEmbedderWithClassifier(nn.Module):
    def __init__(self,
                 base_model_name: str,
                 num_fusion_layers: int,
                 fusion_dropout: float = 0.1,
                 num_classes: int = 1,
                 use_lora: bool = False,
                 lora_config_dict: Optional[Dict[str, Any]] = None,
                 classifier_num_attn_heads: int = 4,
                 classifier_ffn_expansion: int = 2):
        super().__init__()
        self.num_fusion_layers = num_fusion_layers
        self.use_lora = use_lora and PEFT_AVAILABLE
        
        self.base_model_internal = AutoModel.from_pretrained(base_model_name, output_hidden_states=True)
        
        if self.use_lora:
            logging.info(f"LoRA is configured for this model. Adapters will be loaded separately if paths are provided. LoRA Config: {lora_config_dict}")
            self.base_model = self.base_model_internal 
        else:
            self.base_model = self.base_model_internal
            if use_lora and not PEFT_AVAILABLE:
                logging.warning("LoRA configured but PEFT not available. Using base model.")

        self.fusion_module = AdaptiveLayerFusion(
            hidden_size=self.base_model_internal.config.hidden_size,
            num_layers=num_fusion_layers,
            dropout=fusion_dropout
        )
        self.classifier = AttentionalClassifierHead(
            hidden_size=self.base_model_internal.config.hidden_size,
            num_classes=num_classes,
            num_attn_heads=classifier_num_attn_heads,
            ffn_expansion=classifier_ffn_expansion,
            dropout_rate=fusion_dropout
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        base_model_outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        all_layers = list(base_model_outputs.hidden_states)

        num_transformer_layers = len(all_layers) - 1
        if not (0 < self.num_fusion_layers <= num_transformer_layers):
            logging.warning(f"Configured num_fusion_layers ({self.num_fusion_layers}) might be problematic. "
                            f"Model has {num_transformer_layers} transformer layers. "
                            f"Ensure this matches training. Will attempt to use top {self.num_fusion_layers}.")
            actual_layers_to_select = min(self.num_fusion_layers, num_transformer_layers)
            if actual_layers_to_select <=0 :
                raise ValueError("num_fusion_layers for selection must be positive.")
            hidden_states_for_fusion = all_layers[-actual_layers_to_select:]
        else:
            hidden_states_for_fusion = all_layers[-self.num_fusion_layers:]
        
        if len(hidden_states_for_fusion) != self.fusion_module.num_layers_to_fuse:
            logging.warning(f"Forward pass: Number of hidden states for fusion ({len(hidden_states_for_fusion)}) "
                            f"does not match fusion module's configured num_layers_to_fuse ({self.fusion_module.num_layers_to_fuse}). "
                            f"This might lead to unexpected behavior or errors in layer weighting if mismatched significantly from training.")

        fused_embeddings, _ = self.fusion_module(hidden_states_for_fusion, attention_mask)
        return self.classifier(fused_embeddings)

# ───── 自定义数据集 (Copied from training script) ────────────────────────
class MIMICIVTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: Optional[List[int]], tokenizer, max_len: int, has_labels: bool = True):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len
        self.has_labels = has_labels
        if self.has_labels and self.labels is None:
            raise ValueError("Labels must be provided if has_labels is True.")
        if not self.has_labels:
            self.labels = [0] * len(self.texts)

    def __len__(self): return len(self.texts)
    def __getitem__(self, item_idx):
        text = str(self.texts[item_idx])
        label_val = self.labels[item_idx] if self.labels is not None else 0

        encoding = self.tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=self.max_len,
            return_token_type_ids=False, padding='max_length', truncation=True,
            return_attention_mask=True, return_tensors='pt'
        )
        item = {'text': text, 
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten()}
        if self.has_labels:
            item['labels'] = torch.tensor(label_val, dtype=torch.float)
        return item

# ───── Main Inference Function ─────────────────────────────────────────
def perform_inference(
    model: TextEmbedderWithClassifier,
    dataloader: DataLoader,
    device: torch.device,
    has_labels: bool = True
) -> Tuple[pd.DataFrame, float]: # Returns DataFrame and total inference time
    model.eval()
    all_texts, all_preds_probs, all_pred_labels = [], [], []
    all_true_labels = [] if has_labels else None

    start_time = time.monotonic() # Start timing for this inference call

    progress_bar = tqdm(dataloader, desc="Inference", leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts_batch = batch['text']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            predicted_labels_batch = (probs > 0.5).astype(int).flatten()

            all_texts.extend(texts_batch)
            all_preds_probs.extend(probs.flatten().tolist())
            all_pred_labels.extend(predicted_labels_batch.tolist())

            if has_labels and all_true_labels is not None:
                true_labels_batch = batch['labels'].cpu().numpy().flatten()
                all_true_labels.extend(true_labels_batch.astype(int).tolist())
    
    end_time = time.monotonic() # End timing
    total_inference_time_seconds = end_time - start_time

    results_data = {"text": all_texts, "predicted_probability": all_preds_probs, "predicted_label": all_pred_labels}
    if has_labels and all_true_labels is not None:
        results_data["true_label"] = all_true_labels
        
    return pd.DataFrame(results_data), total_inference_time_seconds

# ───── Metrics Calculation Helper ──────────────────────────────────────
def calculate_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2: 
        return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def get_bootstrap_ci(
    y_true: np.ndarray, 
    y_pred_or_prob: np.ndarray, 
    metric_func: Callable, 
    n_bootstraps: int = N_BOOTSTRAPS_DEFAULT, 
    random_state: Optional[int] = None, 
    is_prob_metric: bool = False, 
    **metric_func_kwargs
) -> Tuple[float, float]:
    if len(y_true) == 0:
        return (np.nan, np.nan)
        
    bootstrapped_scores = []
    rng = np.random.RandomState(random_state) 
    
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(indices) == 0: continue

        resampled_true = y_true[indices]
        resampled_pred_or_prob = y_pred_or_prob[indices]
        
        try:
            if metric_func == roc_auc_score or metric_func == calculate_auprc:
                if len(np.unique(resampled_true)) < 2:
                    bootstrapped_scores.append(np.nan)
                    continue 
            
            score = metric_func(resampled_true, resampled_pred_or_prob, **metric_func_kwargs)
            bootstrapped_scores.append(score)
        except ValueError: 
            bootstrapped_scores.append(np.nan)

    bootstrapped_scores = [s for s in bootstrapped_scores if not np.isnan(s)] 
    if not bootstrapped_scores: 
        return (np.nan, np.nan)
        
    lower_bound = np.percentile(bootstrapped_scores, 2.5)
    upper_bound = np.percentile(bootstrapped_scores, 97.5)
    return (lower_bound, upper_bound)

def sanitize_for_json(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(i) for i in data]
    elif isinstance(data, tuple):
        return tuple(sanitize_for_json(i) for i in data)
    elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None 
    elif isinstance(data, (np.integer, np.floating)): 
        return data.item()
    return data

# ───── 主流程 ───────────────────────────────────────────────────────────
def run_inference_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    results_dir_path_str = None
    if args.lazy_inference:
        base_path = Path(args.base_results_path)
        if not base_path.is_dir():
            logging.error(f"Base results path for lazy inference not found or not a directory: {base_path}. Exiting.")
            return
        all_subdirs = [d for d in base_path.iterdir() if d.is_dir()]
        if not all_subdirs:
            logging.error(f"No subdirectories found in base results path: {base_path} for lazy inference. Exiting.")
            return
        try:
            latest_dir = sorted(all_subdirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
            results_dir_path_str = str(latest_dir)
            logging.info(f"Lazy inference: Using latest results directory: {results_dir_path_str}")
        except IndexError: 
            logging.error(f"Could not determine latest directory in {base_path}. Exiting.")
            return
        except Exception as e:
            logging.error(f"Error determining latest directory in {base_path}: {e}. Exiting.", exc_info=True)
            return
    elif args.results_dir:
        results_dir_path_str = args.results_dir
        logging.info(f"Using specified results directory: {results_dir_path_str}")
    else:
        logging.error("You must either specify --results-dir or use --lazy-inference. Exiting.")
        return

    results_dir_path = Path(results_dir_path_str)
    if not results_dir_path.is_dir():
        logging.error(f"Training results directory not found: {results_dir_path}. Exiting.")
        return
    
    hyperparams_file = results_dir_path / "hyperparameters.json"
    if not hyperparams_file.exists():
        logging.error(f"Hyperparameters file not found in results directory: {hyperparams_file}. Exiting.")
        return
    
    with open(hyperparams_file, 'r') as f:
        hyperparams = json.load(f)
    logging.info(f"Loaded hyperparameters from: {hyperparams_file}")

    current_seed = hyperparams.get("GLOBAL_SEED", GLOBAL_SEED)
    active_seed = current_seed 
    if args.seed is not None: 
        set_seed(args.seed) 
        active_seed = args.seed
    else:
        set_seed(current_seed) 
    
    base_model_name = hyperparams.get("EMBED_MODEL")
    num_fusion_layers = hyperparams.get("NUM_FUSION_LAYERS")
    max_seq_len = hyperparams.get("MAX_SEQ_LENGTH", DEFAULT_MAX_SEQ_LENGTH)
    use_lora_trained = hyperparams.get("USE_LORA", False)
    
    fusion_weights_basename = hyperparams.get("FUSION_WEIGHTS_BASENAME_ARG", "trained_model_weights.pth")
    lora_adapter_dir_basename = hyperparams.get("LORA_ADAPTER_DIR_BASENAME_ARG", "lora_adapters")
    trained_fusion_weights_path = results_dir_path / fusion_weights_basename
    trained_lora_adapter_path = results_dir_path / lora_adapter_dir_basename

    if not base_model_name or num_fusion_layers is None:
        logging.error("Essential hyperparameters (EMBED_MODEL, NUM_FUSION_LAYERS) not found in hyperparameters.json. Exiting.")
        return

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    fusion_dropout_trained = hyperparams.get("FUSION_DROPOUT", 0.1)
    classifier_num_attn_heads = hyperparams.get("CLASSIFIER_NUM_ATTN_HEADS", 4)
    classifier_ffn_expansion = hyperparams.get("CLASSIFIER_FFN_EXPANSION", 2)
    lora_config_dict_loaded = {
        "r": hyperparams.get("LORA_R"), "lora_alpha": hyperparams.get("LORA_ALPHA"),
        "lora_dropout": hyperparams.get("LORA_DROPOUT"), "target_modules": hyperparams.get("LORA_TARGET_MODULES")
    }

    model = TextEmbedderWithClassifier(
        base_model_name=base_model_name, num_fusion_layers=num_fusion_layers,
        fusion_dropout=fusion_dropout_trained, num_classes=1, use_lora=use_lora_trained,
        lora_config_dict=lora_config_dict_loaded if use_lora_trained else None,
        classifier_num_attn_heads=classifier_num_attn_heads,
        classifier_ffn_expansion=classifier_ffn_expansion
    ).to(device)

    if use_lora_trained and PEFT_AVAILABLE:
        if trained_lora_adapter_path.exists() and any(trained_lora_adapter_path.iterdir()):
            try:
                logging.info(f"Loading LoRA adapters from {trained_lora_adapter_path}...")
                model.base_model = PeftModel.from_pretrained(model.base_model_internal, str(trained_lora_adapter_path))
                model.base_model = model.base_model.merge_and_unload()
                logging.info("LoRA adapters loaded and merged successfully.")
                model.use_lora = True
            except Exception as e:
                logging.error(f"Error loading LoRA adapters from {trained_lora_adapter_path}: {e}. Proceeding without LoRA.", exc_info=True)
                model.base_model = model.base_model_internal
                model.use_lora = False
        else:
            logging.warning(f"LoRA was used in training, but adapter path {trained_lora_adapter_path} not found or empty. Proceeding without LoRA.")
            model.base_model = model.base_model_internal
            model.use_lora = False
    elif use_lora_trained and not PEFT_AVAILABLE:
        logging.warning("LoRA was used in training, but PEFT library is not available. Cannot load LoRA adapters.")
        model.base_model = model.base_model_internal
        model.use_lora = False
    else:
        model.base_model = model.base_model_internal
        model.use_lora = False

    if trained_fusion_weights_path.exists():
        try:
            checkpoint = torch.load(trained_fusion_weights_path, map_location=device)
            if 'fusion_module_state_dict' in checkpoint and 'classifier_head_state_dict' in checkpoint:
                model.fusion_module.load_state_dict(checkpoint['fusion_module_state_dict'])
                model.classifier.load_state_dict(checkpoint['classifier_head_state_dict'])
                logging.info(f"Loaded fusion module and classifier weights from {trained_fusion_weights_path} (structured checkpoint).")
            else:
                logging.error("Could not load weights due to unexpected checkpoint format. Ensure training saved fusion/classifier heads correctly. Exiting.")
                return
        except Exception as e:
            logging.error(f"Error loading fusion/classifier weights from {trained_fusion_weights_path}: {e}. Exiting.", exc_info=True)
            return
    else:
        logging.error(f"Trained fusion/classifier weights file not found: {trained_fusion_weights_path}. Cannot perform inference. Exiting.")
        return

    model.to(device)
    model.eval()

    input_config_path = Path(args.input_config_json)
    if not input_config_path.exists():
        logging.error(f"Input configuration JSON file not found: {input_config_path}. Exiting.")
        return
    try:
        with open(input_config_path, 'r') as f_config:
            input_configs = json.load(f_config)
        if not isinstance(input_configs, list):
            logging.error("Input configuration JSON must contain a list of configurations. Exiting.")
            return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding input configuration JSON {input_config_path}: {e}. Exiting.")
        return
    except Exception as e:
        logging.error(f"Error reading input configuration JSON {input_config_path}: {e}. Exiting.", exc_info=True)
        return

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_output_dir = results_dir_path / "inference" 
    inference_output_dir.mkdir(parents=True, exist_ok=True)

    for i, config_entry in enumerate(input_configs):
        logging.info(f"\nProcessing input configuration {i+1}/{len(input_configs)}: {config_entry.get('file_path')}")

        input_csv_path_str = config_entry.get('file_path')
        text_col_name = config_entry.get('text_col')
        target_col_name = config_entry.get('target_col') 

        if not input_csv_path_str or not text_col_name:
            logging.error(f"Skipping configuration {i+1} due to missing 'file_path' or 'text_col'.")
            continue
        
        input_csv_path = Path(input_csv_path_str)
        input_csv_basename = input_csv_path.stem 

        if not input_csv_path.exists():
            logging.error(f"Input CSV for configuration {i+1} not found: {input_csv_path}. Skipping.")
            continue
        
        try:
            infer_df = pd.read_csv(input_csv_path)
            logging.info(f"Loaded inference data from {input_csv_path}. Shape: {infer_df.shape}")
        except Exception as e:
            logging.error(f"Error reading inference CSV {input_csv_path}: {e}. Skipping.", exc_info=True)
            continue

        if text_col_name not in infer_df.columns:
            logging.error(f"Text column '{text_col_name}' not found in {input_csv_path}. Skipping.")
            continue
        
        texts_to_infer = infer_df[text_col_name].astype(str).tolist()
        num_samples = len(texts_to_infer)
        labels_to_infer = None
        has_labels_in_csv = False

        if target_col_name:
            if target_col_name not in infer_df.columns:
                logging.warning(f"Target column '{target_col_name}' specified but not found in {input_csv_path}. Proceeding without true labels for this file.")
            else:
                try:
                    labels_to_infer = infer_df[target_col_name].astype(int).tolist()
                    has_labels_in_csv = True
                    logging.info(f"Found target column '{target_col_name}' in {input_csv_path}. True labels will be processed.")
                except ValueError:
                    logging.warning(f"Could not convert target column '{target_col_name}' to int in {input_csv_path}. Proceeding without true labels for this file.")
                    labels_to_infer = None
                    has_labels_in_csv = False
        
        infer_dataset = MIMICIVTextDataset(
            texts=texts_to_infer, labels=labels_to_infer, tokenizer=tokenizer, 
            max_len=max_seq_len, has_labels=has_labels_in_csv
        )
        
        g_infer = torch.Generator(); g_infer.manual_seed(active_seed)
        infer_dataloader = DataLoader(
            infer_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=min(2, os.cpu_count()//2 if os.cpu_count() else 1),
            pin_memory=(device.type=="cuda"),
            worker_init_fn=lambda worker_id: set_seed(active_seed + worker_id)
        )

        # Reset peak memory stats for CUDA device if applicable, before this specific inference
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)

        logging.info(f"Starting inference for {input_csv_path} ({num_samples} samples)...")
        inference_results_df, total_inference_time_seconds = perform_inference(model, infer_dataloader, device, has_labels=has_labels_in_csv)
        
        avg_time_per_sample_ms = (total_inference_time_seconds / num_samples) * 1000 if num_samples > 0 else 0
        
        performance_stats = {
            "total_inference_time_seconds": total_inference_time_seconds,
            "samples_processed": num_samples,
            "avg_time_per_sample_ms": avg_time_per_sample_ms
        }
        logging.info(f"  Total inference time for {input_csv_basename}: {total_inference_time_seconds:.2f} seconds.")
        logging.info(f"  Average time per sample for {input_csv_basename}: {avg_time_per_sample_ms:.2f} ms.")

        if device.type == 'cuda':
            peak_gpu_memory_bytes = torch.cuda.max_memory_allocated(device)
            peak_gpu_memory_mb = peak_gpu_memory_bytes / (1024 * 1024)
            performance_stats["peak_gpu_memory_mb"] = peak_gpu_memory_mb
            logging.info(f"  Peak GPU memory for {input_csv_basename}: {peak_gpu_memory_mb:.2f} MB.")
        

        output_csv_filename = f"inference_results_{input_csv_basename}_{timestamp_str}.csv"
        output_csv_filepath = inference_output_dir / output_csv_filename
        try:
            inference_results_df.to_csv(output_csv_filepath, index=False)
            logging.info(f"Inference results for {input_csv_basename} saved to: {output_csv_filepath}")
        except Exception as e:
            logging.error(f"Error saving inference results for {input_csv_basename} to {output_csv_filepath}: {e}", exc_info=True)

        metrics_data = {} # Initialize for current CSV
        if has_labels_in_csv and \
           'true_label' in inference_results_df.columns and \
           'predicted_probability' in inference_results_df.columns and \
           'predicted_label' in inference_results_df.columns:

            logging.info(f"Computing evaluation metrics for {input_csv_basename}...")
            true_labels_np = inference_results_df['true_label'].values.astype(int)
            pred_probs_np = inference_results_df['predicted_probability'].values
            pred_labels_np = inference_results_df['predicted_label'].values.astype(int)
            bootstrap_random_state_seed = active_seed
            
            # Accuracy
            acc_point = accuracy_score(true_labels_np, pred_labels_np)
            acc_ci = get_bootstrap_ci(true_labels_np, pred_labels_np, accuracy_score, 
                                      random_state=bootstrap_random_state_seed, n_bootstraps=args.n_bootstraps)
            metrics_data["accuracy"] = {"value": acc_point, "confidence_interval_95": list(acc_ci)}
            logging.info(f"  Accuracy ({input_csv_basename}): {acc_point:.4f} (95% CI: [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}])")
            
            # Precision, Recall, F1, ROC AUC, AUPRC, Confusion Matrix (as before)
            precision_point = precision_score(true_labels_np, pred_labels_np, zero_division=0)
            precision_ci = get_bootstrap_ci(true_labels_np, pred_labels_np, precision_score, random_state=bootstrap_random_state_seed, n_bootstraps=args.n_bootstraps, zero_division=0)
            metrics_data["precision"] = {"value": precision_point, "confidence_interval_95": list(precision_ci)}
            logging.info(f"  Precision ({input_csv_basename}): {precision_point:.4f} (95% CI: [{precision_ci[0]:.4f}, {precision_ci[1]:.4f}])")

            recall_point = recall_score(true_labels_np, pred_labels_np, zero_division=0)
            recall_ci = get_bootstrap_ci(true_labels_np, pred_labels_np, recall_score, random_state=bootstrap_random_state_seed, n_bootstraps=args.n_bootstraps, zero_division=0)
            metrics_data["recall"] = {"value": recall_point, "confidence_interval_95": list(recall_ci)}
            logging.info(f"  Recall ({input_csv_basename}): {recall_point:.4f} (95% CI: [{recall_ci[0]:.4f}, {recall_ci[1]:.4f}])")

            f1_point = f1_score(true_labels_np, pred_labels_np, zero_division=0)
            f1_ci = get_bootstrap_ci(true_labels_np, pred_labels_np, f1_score, random_state=bootstrap_random_state_seed, n_bootstraps=args.n_bootstraps, zero_division=0)
            metrics_data["f1_score"] = {"value": f1_point, "confidence_interval_95": list(f1_ci)}
            logging.info(f"  F1-score ({input_csv_basename}): {f1_point:.4f} (95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}])")
            
            if len(np.unique(true_labels_np)) > 1:
                try:
                    roc_auc_point = roc_auc_score(true_labels_np, pred_probs_np)
                    roc_auc_ci = get_bootstrap_ci(true_labels_np, pred_probs_np, roc_auc_score, random_state=bootstrap_random_state_seed, n_bootstraps=args.n_bootstraps, is_prob_metric=True)
                    metrics_data["roc_auc"] = {"value": roc_auc_point, "confidence_interval_95": list(roc_auc_ci)}
                    logging.info(f"  ROC AUC ({input_csv_basename}): {roc_auc_point:.4f} (95% CI: [{roc_auc_ci[0]:.4f}, {roc_auc_ci[1]:.4f}])")
                except ValueError as e:
                     logging.warning(f"Could not compute ROC AUC for {input_csv_basename}: {e}. Setting to NaN.")
                     metrics_data["roc_auc"] = {"value": np.nan, "confidence_interval_95": [np.nan, np.nan]}
            else:
                logging.warning(f"ROC AUC not computed for {input_csv_basename} because there is only one class in true labels.")
                metrics_data["roc_auc"] = {"value": np.nan, "confidence_interval_95": [np.nan, np.nan]}

            if len(np.unique(true_labels_np)) > 1:
                try:
                    auprc_point = calculate_auprc(true_labels_np, pred_probs_np)
                    auprc_ci = get_bootstrap_ci(true_labels_np, pred_probs_np, calculate_auprc, random_state=bootstrap_random_state_seed, n_bootstraps=args.n_bootstraps, is_prob_metric=True)
                    metrics_data["auprc"] = {"value": auprc_point, "confidence_interval_95": list(auprc_ci)}
                    logging.info(f"  AUPRC ({input_csv_basename}): {auprc_point:.4f} (95% CI: [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}])")
                except ValueError as e: 
                    logging.warning(f"Could not compute AUPRC for {input_csv_basename}: {e}. Setting to NaN.")
                    metrics_data["auprc"] = {"value": np.nan, "confidence_interval_95": [np.nan, np.nan]}
            else:
                logging.warning(f"AUPRC not computed for {input_csv_basename} because there is only one class in true labels.")
                metrics_data["auprc"] = {"value": np.nan, "confidence_interval_95": [np.nan, np.nan]}

            cm = confusion_matrix(true_labels_np, pred_labels_np, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0] if cm.shape==(1,1) and true_labels_np[0]==0 else 0, 0, 0, cm[0,0] if cm.shape==(1,1) and true_labels_np[0]==1 else 0) 
            if cm.size != 4 and len(np.unique(true_labels_np)) == 1: 
                 if np.unique(true_labels_np)[0] == 0: 
                     tn = len(true_labels_np); fp, fn, tp = 0,0,0
                 else: 
                     tp = len(true_labels_np); tn, fp, fn = 0,0,0
            metrics_data["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
            logging.info(f"  Confusion Matrix ({input_csv_basename}): TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            
            metrics_data["performance"] = performance_stats # Add performance stats to the same dict

            metrics_output_filename = f"inference_metrics_{input_csv_basename}_{timestamp_str}.json"
            metrics_output_filepath = inference_output_dir / metrics_output_filename
            sanitized_metrics_data = sanitize_for_json(metrics_data)
            with open(metrics_output_filepath, 'w') as f_metrics:
                json.dump(sanitized_metrics_data, f_metrics, indent=4)
            logging.info(f"Inference metrics for {input_csv_basename} saved to: {metrics_output_filepath}")

        elif has_labels_in_csv: # Labels were expected, but columns for metrics were missing in results_df
            logging.warning(f"Evaluation metrics calculation skipped for {input_csv_basename}: Necessary columns not found in results DataFrame.")
            # Still save performance stats if no labels
            metrics_data["performance"] = performance_stats
            metrics_output_filename = f"inference_metrics_{input_csv_basename}_{timestamp_str}.json"
            metrics_output_filepath = inference_output_dir / metrics_output_filename
            sanitized_metrics_data = sanitize_for_json(metrics_data) # metrics_data only contains performance here
            with open(metrics_output_filepath, 'w') as f_metrics:
                json.dump(sanitized_metrics_data, f_metrics, indent=4)
            logging.info(f"Performance-only metrics for {input_csv_basename} saved to: {metrics_output_filepath}")

        else: # No labels provided in input CSV
            logging.info(f"No true labels provided for {input_csv_basename}. Skipping evaluation metrics calculation.")
            # Still save performance stats if no labels
            metrics_data["performance"] = performance_stats
            metrics_output_filename = f"inference_metrics_{input_csv_basename}_{timestamp_str}.json"
            metrics_output_filepath = inference_output_dir / metrics_output_filename
            sanitized_metrics_data = sanitize_for_json(metrics_data) # metrics_data only contains performance here
            with open(metrics_output_filepath, 'w') as f_metrics:
                json.dump(sanitized_metrics_data, f_metrics, indent=4)
            logging.info(f"Performance-only metrics for {input_csv_basename} saved to: {metrics_output_filepath}")


    logging.info("\nInference pipeline finished for all configurations.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inference pipeline for text classification model.")
    
    group = parser.add_mutually_exclusive_group(required=False) 
    group.add_argument("--results-dir", type=str, default=None,
                       help="Path to the specific training results directory (e.g., 'results/modelname_timestamp') which contains "
                            "hyperparameters.json and trained model weights. Required if --lazy-inference is not used.")
    group.add_argument("--lazy-inference", action='store_true',
                       help="If set, automatically find and use the latest training results directory from --base-results-path.")

    parser.add_argument("--base-results-path", type=str, default=DEFAULT_BASE_RESULTS_PATH,
                        help=f"Parent directory containing all training run folders. Used with --lazy-inference. Default: {DEFAULT_BASE_RESULTS_PATH}")
    
    parser.add_argument("--input-config-json", type=str, required=True,
                        help="Path to a JSON file specifying input CSVs and their respective text/target columns. "
                             "Example: [{'file_path': 'data.csv', 'text_col': 'text', 'target_col': 'label'}, ...]")
    
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size for inference. Default: {DEFAULT_BATCH_SIZE}")
    
    parser.add_argument("--seed", type=int, default=None, 
                        help="Override random seed for this inference run (optional). If not set, uses seed from loaded hyperparameters or a global default.")
    parser.add_argument("--n-bootstraps", type=int, default=N_BOOTSTRAPS_DEFAULT,
                        help=f"Number of bootstrap samples for CI calculation. Default: {N_BOOTSTRAPS_DEFAULT}")

    parsed_args = parser.parse_args()
    
    if not parsed_args.lazy_inference and not parsed_args.results_dir:
        parser.error("Either --results-dir must be specified, or --lazy-inference must be used.")
    
    if parsed_args.lazy_inference and parsed_args.results_dir:
        logging.warning("--lazy-inference is set, so the value of --results-dir will be ignored. The latest directory from --base-results-path will be used.")

    run_inference_pipeline(parsed_args)
