#!/usr/bin/env python3
"""
Enhanced model architecture for Thai profanity detection
Improvements over base Wav2Vec2 approach
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Config
import torch.nn.functional as F

class ThaiProfanityDetector(nn.Module):
    """
    Enhanced architecture for Thai profanity detection
    Features:
    1. Temporal attention mechanism
    2. Multi-scale feature extraction
    3. Context-aware classification
    4. Focal loss for class imbalance
    """
    
    def __init__(self, config, num_labels=5):
        super().__init__()
        self.config = config
        self.num_labels = num_labels
        
        # Base Wav2Vec2 encoder
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # Multi-scale temporal convolutions
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(config.hidden_size, config.hidden_size//2, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]  # Different temporal receptive fields
        ])
        
        # Temporal attention mechanism
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Context aggregation
        self.context_lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size//2,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Classification head with dropout
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(config.hidden_size, config.hidden_size//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_size//2, num_labels)
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize classification head weights"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, input_values, attention_mask=None, labels=None):
        # Extract features from Wav2Vec2
        wav2vec2_outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask
        )
        
        hidden_states = wav2vec2_outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        
        # Multi-scale temporal feature extraction
        hidden_states_transposed = hidden_states.transpose(1, 2)  # [batch, hidden_size, seq_len]
        multi_scale_features = []
        
        for conv in self.temporal_convs:
            conv_out = F.relu(conv(hidden_states_transposed))
            multi_scale_features.append(conv_out)
        
        # Concatenate multi-scale features
        multi_scale_concat = torch.cat(multi_scale_features, dim=1)  # [batch, hidden_size*1.5, seq_len]
        multi_scale_concat = multi_scale_concat.transpose(1, 2)  # [batch, seq_len, hidden_size*1.5]
        
        # Project back to original dimension
        multi_scale_projected = nn.Linear(
            multi_scale_concat.size(-1), 
            hidden_states.size(-1)
        ).to(hidden_states.device)(multi_scale_concat)
        
        # Apply temporal attention
        attn_output, attn_weights = self.temporal_attention(
            multi_scale_projected,
            multi_scale_projected,
            multi_scale_projected,
            key_padding_mask=~attention_mask.bool() if attention_mask is not None else None
        )
        
        # Context aggregation with LSTM
        lstm_output, _ = self.context_lstm(attn_output)
        
        # Global average pooling with attention weighting
        if attention_mask is not None:
            # Weight by attention mask
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(lstm_output)
            masked_output = lstm_output * attention_mask_expanded
            pooled_output = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled_output = lstm_output.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Use focal loss for class imbalance
            loss = self.focal_loss(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'attention_weights': attn_weights,
            'hidden_states': lstm_output
        }
    
    def focal_loss(self, logits, labels, alpha=0.25, gamma=2.0):
        """
        Focal loss to handle class imbalance
        """
        ce_loss = F.cross_entropy(logits, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

class AdaptiveWindowDetector(nn.Module):
    """
    Adaptive window-based detection that can handle variable-length profanity
    """
    
    def __init__(self, base_model, window_sizes=[0.3, 0.5, 0.7, 1.0]):
        super().__init__()
        self.base_model = base_model
        self.window_sizes = window_sizes
        
        # Window-specific projection layers
        self.window_projections = nn.ModuleDict({
            f'window_{w}s': nn.Linear(base_model.config.hidden_size, base_model.num_labels)
            for w in window_sizes
        })
        
        # Window fusion mechanism
        self.window_fusion = nn.MultiheadAttention(
            embed_dim=base_model.num_labels,
            num_heads=1,
            batch_first=True
        )
        
        # Final classifier
        self.final_classifier = nn.Linear(base_model.num_labels, base_model.num_labels)
    
    def forward(self, input_values, window_size=None, **kwargs):
        # If specific window size is provided, use single window
        if window_size is not None:
            return self.base_model(input_values, **kwargs)
        
        # Multi-window approach
        window_outputs = []
        
        for w in self.window_sizes:
            # Process with specific window context (simplified)
            output = self.base_model(input_values, **kwargs)
            window_logits = self.window_projections[f'window_{w}s'](
                output['hidden_states'].mean(dim=1)
            )
            window_outputs.append(window_logits.unsqueeze(1))
        
        # Concatenate window outputs
        multi_window_logits = torch.cat(window_outputs, dim=1)  # [batch, num_windows, num_labels]
        
        # Fuse window predictions with attention
        fused_output, _ = self.window_fusion(
            multi_window_logits,
            multi_window_logits,
            multi_window_logits
        )
        
        # Final prediction
        final_logits = self.final_classifier(fused_output.mean(dim=1))
        
        return {
            'logits': final_logits,
            'window_logits': multi_window_logits
        }