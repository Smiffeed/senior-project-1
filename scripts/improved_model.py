import torch
from torch import nn
from transformers import Wav2Vec2Model, Wav2Vec2Config

class AttentionPooling(nn.Module):
    """
    Attention pooling layer for weighted feature aggregation
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1)
        )
    
    def forward(self, features, attention_mask=None):
        # Get sequence length from hidden states
        batch_size, seq_length, hidden_size = features.shape
        
        # Apply the attention projection
        # [batch_size, seq_length, hidden_size] -> [batch_size, seq_length, 1]
        attention_scores = self.attention(features)
        
        # Apply mask before softmax if provided
        if attention_mask is not None:
            # Make sure attention_mask has the right shape by adapting it
            # The mask might be [batch_size, orig_seq_length]
            # But we need [batch_size, seq_length]
            
            # First check if dimensions match
            if attention_mask.shape[1] != seq_length:
                # Create a new mask of the correct size (all 1s)
                new_mask = torch.ones((batch_size, seq_length), 
                                      device=attention_mask.device)
                
                # Only use the provided mask if it's smaller (truncate otherwise)
                min_length = min(attention_mask.shape[1], seq_length)
                new_mask[:, :min_length] = attention_mask[:, :min_length]
                attention_mask = new_mask
            
            # Convert to float and unsqueeze for broadcasting
            attention_mask = attention_mask.float().unsqueeze(-1)
            
            # Apply the mask (using a large negative number for masked positions)
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
        
        # Apply softmax to get attention weights [batch_size, seq_length, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention weights to get weighted features
        # [batch_size, seq_length, hidden_size] * [batch_size, seq_length, 1]
        weighted_features = features * attention_weights
        
        # Sum over sequence dimension to get final pooled representation
        # [batch_size, seq_length, hidden_size] -> [batch_size, hidden_size]
        context = weighted_features.sum(dim=1)
        
        return context


class Wav2Vec2ClassifierWithAttention(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        super().__init__()
        
        # Load base model config and model
        self.config = Wav2Vec2Config.from_pretrained(model_name)
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze the feature extractor part for initial training
        self._freeze_feature_extractor()
        
        # Get hidden size from configuration
        hidden_size = self.config.hidden_size
        
        # Add attention pooling layer
        self.attention_pooling = AttentionPooling(hidden_size)
        
        # Add classifier head with multiple dropouts (MC Dropout)
        self.classifier_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, num_labels)
        )
        
    def _freeze_feature_extractor(self):
        """Freeze the feature extractor part of the model"""
        for param in self.wav2vec2.feature_extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze the base model for fine-tuning"""
        for param in self.wav2vec2.parameters():
            param.requires_grad = True
            
    def forward(self, input_values, attention_mask=None, labels=None):
        # Get Wav2Vec2 output features
        outputs = self.wav2vec2(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get the output of the final layer
        hidden_states = outputs.last_hidden_state
        
        # Apply attention pooling
        pooled_output = self.attention_pooling(hidden_states, attention_mask)
        
        # Get logits from classification head
        logits = self.classifier_head(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.extract_features
        }


# For two-stage fine-tuning
class StagedTrainingWav2Vec2(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.3, mixup_alpha=0.2):
        super().__init__()
        self.model = Wav2Vec2ClassifierWithAttention(model_name, num_labels, dropout_rate)
        self.mixup_alpha = mixup_alpha
        self.num_labels = num_labels
        self.is_second_stage = False
        
    def forward(self, input_values, attention_mask=None, labels=None):
        # Apply mixup during training if in second stage
        if self.training and self.is_second_stage and labels is not None and self.mixup_alpha > 0:
            # Create mixup training samples
            lam = torch.distributions.beta.Beta(self.mixup_alpha, self.mixup_alpha).sample()
            batch_size = input_values.size(0)
            index = torch.randperm(batch_size)
            
            mixed_input_values = lam * input_values + (1 - lam) * input_values[index]
            
            # Forward pass with mixed inputs
            outputs = self.model(mixed_input_values, attention_mask)
            
            # Calculate mixed loss
            loss_fct = nn.CrossEntropyLoss()
            loss1 = loss_fct(outputs["logits"], labels)
            loss2 = loss_fct(outputs["logits"], labels[index])
            outputs["loss"] = lam * loss1 + (1 - lam) * loss2
            
            return outputs
        else:
            # Standard forward pass
            return self.model(input_values, attention_mask, labels)
    
    def start_second_stage(self):
        """Transition to second training stage - unfreeze base model and enable mixup"""
        self.model.unfreeze_base_model()
        self.is_second_stage = True
        print("Model switched to second training stage (full fine-tuning with mixup)")


# Factory function to create model
def create_improved_model(model_name, num_labels, use_staged_training=True, dropout_rate=0.3, mask_time_prob=0.05):
    """
    Create an improved model with attention mechanism and staged training support
    
    Parameters:
    - model_name: Name of the pre-trained wav2vec2 model to use
    - num_labels: Number of classes for classification
    - use_staged_training: Whether to use staged training (freeze feature extractor first)
    - dropout_rate: Dropout rate to use in the classification head
    - mask_time_prob: Probability of masking time frames (set to 0 to disable masking)
    
    Returns:
    - Model ready for training
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Config
    from torch import nn
    import torch
    
    class AttentionPooling(nn.Module):
        """
        Attention pooling layer for weighted feature aggregation
        """
        def __init__(self, input_dim):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.Tanh(),
                nn.Linear(input_dim, 1)
            )
        
        def forward(self, features, attention_mask=None):
            # Get sequence length from hidden states
            batch_size, seq_length, hidden_size = features.shape
            
            # Apply the attention projection
            # [batch_size, seq_length, hidden_size] -> [batch_size, seq_length, 1]
            attention_scores = self.attention(features)
            
            # Apply mask before softmax if provided
            if attention_mask is not None:
                # Make sure attention_mask has the right shape by adapting it
                # The mask might be [batch_size, orig_seq_length]
                # But we need [batch_size, seq_length]
                
                # First check if dimensions match
                if attention_mask.shape[1] != seq_length:
                    # Create a new mask of the correct size (all 1s)
                    new_mask = torch.ones((batch_size, seq_length), 
                                          device=attention_mask.device)
                    
                    # Only use the provided mask if it's smaller (truncate otherwise)
                    min_length = min(attention_mask.shape[1], seq_length)
                    new_mask[:, :min_length] = attention_mask[:, :min_length]
                    attention_mask = new_mask
                
                # Convert to float and unsqueeze for broadcasting
                attention_mask = attention_mask.float().unsqueeze(-1)
                
                # Apply the mask (using a large negative number for masked positions)
                attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
            
            # Apply softmax to get attention weights [batch_size, seq_length, 1]
            attention_weights = torch.softmax(attention_scores, dim=1)
            
            # Apply attention weights to get weighted features
            # [batch_size, seq_length, hidden_size] * [batch_size, seq_length, 1]
            weighted_features = features * attention_weights
            
            # Sum over sequence dimension to get final pooled representation
            # [batch_size, seq_length, hidden_size] -> [batch_size, hidden_size]
            context = weighted_features.sum(dim=1)
            
            return context
    
    class ImprovedClassificationHead(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_labels, dropout_rate=0.3):
            super().__init__()
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dense1 = nn.Linear(input_dim, hidden_dim)
            self.layer_norm = nn.LayerNorm(hidden_dim)
            self.activation = nn.GELU()
            self.dropout2 = nn.Dropout(dropout_rate)
            self.dense2 = nn.Linear(hidden_dim, num_labels)
            
        def forward(self, features):
            x = self.dropout1(features)
            x = self.dense1(x)
            x = self.layer_norm(x)
            x = self.activation(x)
            x = self.dropout2(x)
            x = self.dense2(x)
            return x
    
    class StagedTrainingWav2Vec2(nn.Module):
        def __init__(self, model_name, num_labels, dropout_rate=0.3, mask_time_prob=0.05):
            super().__init__()
            
            # Load model with custom config to disable/control masking
            config = Wav2Vec2Config.from_pretrained(model_name)
            config.mask_time_prob = mask_time_prob  # Set masking probability
            
            # If masking is disabled, also disable these related parameters
            if mask_time_prob == 0.0:
                config.mask_time_length = 1  # Minimum value
                config.mask_feature_prob = 0.0
                config.mask_feature_length = 1
            
            self.feature_extractor = Wav2Vec2Model.from_pretrained(model_name, config=config)
            
            hidden_size = self.feature_extractor.config.hidden_size
            self.attention_pooling = AttentionPooling(hidden_size)
            self.classifier = ImprovedClassificationHead(
                input_dim=hidden_size,
                hidden_dim=hidden_size,
                num_labels=num_labels,
                dropout_rate=dropout_rate
            )
            self.num_labels = num_labels
            self.first_stage = True
            
            # Freeze feature extractor initially if using staged training
            if use_staged_training:
                self._freeze_feature_extractor()
        
        def _freeze_feature_extractor(self):
            """Freeze parameters of the feature extractor"""
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        def _unfreeze_feature_extractor(self):
            """Unfreeze parameters of the feature extractor"""
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        
        def start_second_stage(self):
            """Switch to second stage training by unfreezing the feature extractor"""
            self.first_stage = False
            self._unfreeze_feature_extractor()
            
        def forward(self, input_values, attention_mask=None, labels=None):
            # Extract features with explicit control over masking during inference
            outputs = self.feature_extractor(
                input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                # Don't pass mask_time_indices here - it's controlled by config
            )
            
            # Pool features with attention
            hidden_states = outputs.last_hidden_state
            pooled_features = self.attention_pooling(hidden_states, attention_mask)
            
            # Apply classifier
            logits = self.classifier(pooled_features)
            
            # Calculate loss if labels are provided
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
            
            return {'loss': loss, 'logits': logits}
    
    # Create and return the model
    model = StagedTrainingWav2Vec2(
        model_name, 
        num_labels, 
        dropout_rate=dropout_rate,
        mask_time_prob=mask_time_prob
    )
    return model
