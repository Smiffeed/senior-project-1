import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class SpectralAugmentation:
    """Advanced spectral augmentation techniques for better robustness."""
    
    def __init__(self, time_mask_rate=0.1, freq_mask_rate=0.15, num_masks=2):
        self.time_mask_rate = time_mask_rate
        self.freq_mask_rate = freq_mask_rate
        self.num_masks = num_masks
    
    def spec_augment(self, mel_spectrogram):
        """Apply SpecAugment to mel-spectrogram."""
        spec = mel_spectrogram.clone()
        time_len, freq_len = spec.shape[-2:]
        
        # Time masking
        for _ in range(self.num_masks):
            t = int(time_len * self.time_mask_rate * np.random.random())
            t0 = np.random.randint(0, time_len - t)
            spec[..., t0:t0+t, :] = 0
        
        # Frequency masking
        for _ in range(self.num_masks):
            f = int(freq_len * self.freq_mask_rate * np.random.random())
            f0 = np.random.randint(0, freq_len - f)
            spec[..., :, f0:f0+f] = 0
        
        return spec
    
    def mixup_spectrogram(self, spec1, spec2, alpha=0.4):
        """Apply MixUp augmentation to spectrograms."""
        lam = np.random.beta(alpha, alpha)
        mixed_spec = lam * spec1 + (1 - lam) * spec2
        return mixed_spec, lam

class AdvancedFeatureExtractor:
    """Extract multiple types of audio features for enhanced model input."""
    
    def __init__(self, sr=16000, n_mels=128, n_mfcc=13):
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
    
    def extract_spectral_features(self, audio):
        """Extract spectral features from audio."""
        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels, 
            hop_length=512, win_length=1024
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sr, n_mfcc=self.n_mfcc,
            hop_length=512, win_length=1024
        )
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        
        # Temporal features
        rms = librosa.feature.rms(y=audio)
        
        return {
            'mel_spec': mel_spec_db,
            'mfcc': mfcc,
            'spectral_centroids': spectral_centroids,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'zero_crossing_rate': zero_crossing_rate,
            'chroma': chroma,
            'rms': rms
        }
    
    def get_feature_statistics(self, features):
        """Get statistical features from time-series features."""
        stats = {}
        for feature_name, feature_values in features.items():
            if feature_values.ndim > 1:
                # For 2D features, compute stats across time axis
                stats[f'{feature_name}_mean'] = np.mean(feature_values, axis=1)
                stats[f'{feature_name}_std'] = np.std(feature_values, axis=1)
                stats[f'{feature_name}_max'] = np.max(feature_values, axis=1)
                stats[f'{feature_name}_min'] = np.min(feature_values, axis=1)
            else:
                # For 1D features
                stats[f'{feature_name}_mean'] = np.mean(feature_values)
                stats[f'{feature_name}_std'] = np.std(feature_values)
                stats[f'{feature_name}_max'] = np.max(feature_values)
                stats[f'{feature_name}_min'] = np.min(feature_values)
        
        return stats

class TransformerAudioClassifier(nn.Module):
    """Advanced Transformer-based audio classifier with multi-head attention."""
    
    def __init__(self, input_dim=768, num_labels=9, num_heads=None, num_layers=6):
        super().__init__()
        original_input_dim = input_dim
        self.num_labels = num_labels
        
        # Automatically determine num_heads to ensure input_dim is divisible by num_heads
        if num_heads is None:
            # Find the largest divisor of input_dim that's <= 16 and >= 8
            for candidate_heads in [16, 12, 8]:
                if input_dim % candidate_heads == 0:
                    num_heads = candidate_heads
                    break
            else:
                # Fallback: use 8 and adjust input_dim
                num_heads = 8
                # Round input_dim to nearest multiple of 8
                input_dim = ((input_dim + 4) // 8) * 8
        
        # Ensure input_dim is divisible by num_heads
        assert input_dim % num_heads == 0, f"input_dim ({input_dim}) must be divisible by num_heads ({num_heads})"
        
        self.input_dim = input_dim
        
        # Input projection layer (in case we adjusted input_dim)
        self.input_projection = None
        if original_input_dim != input_dim:
            self.input_projection = nn.Linear(original_input_dim, input_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(input_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=input_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-scale feature fusion
        self.feature_fusion = MultiScaleFeatureFusion(input_dim)
        
        # Classification head with uncertainty estimation
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, num_labels)
        )
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, attention_mask=None):
        # Apply input projection if needed
        if self.input_projection is not None:
            features = self.input_projection(features)
        
        # Add positional encoding
        features = self.pos_encoding(features)
        
        # Apply transformer
        if attention_mask is not None:
            # Convert attention mask to transformer format
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        
        transformer_out = self.transformer(features, src_key_padding_mask=key_padding_mask)
        
        # Multi-scale feature fusion
        fused_features = self.feature_fusion(transformer_out, attention_mask)
        
        # Classification
        logits = self.classifier(fused_features)
        uncertainty = self.uncertainty_head(fused_features)
        
        return {
            'logits': logits,
            'uncertainty': uncertainty,
            'features': fused_features
        }

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class MultiScaleFeatureFusion(nn.Module):
    """Fuse features at multiple temporal scales."""
    
    def __init__(self, input_dim):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_conv = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1)
        self.mid_conv = nn.Conv1d(input_dim, input_dim, kernel_size=7, padding=3)
        self.fusion = nn.Linear(input_dim * 3, input_dim)
        
    def forward(self, features, attention_mask=None):
        # features: (batch, seq_len, dim)
        features_t = features.transpose(1, 2)  # (batch, dim, seq_len)
        
        # Global features
        global_feat = self.global_pool(features_t).squeeze(-1)  # (batch, dim)
        
        # Local features
        local_feat = self.local_conv(features_t)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).expand_as(local_feat)
            local_feat = local_feat * mask_expanded.float()
        local_feat = torch.mean(local_feat, dim=-1)  # (batch, dim)
        
        # Mid-range features
        mid_feat = self.mid_conv(features_t)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(1).expand_as(mid_feat)
            mid_feat = mid_feat * mask_expanded.float()
        mid_feat = torch.mean(mid_feat, dim=-1)  # (batch, dim)
        
        # Fusion
        fused = torch.cat([global_feat, local_feat, mid_feat], dim=1)
        return self.fusion(fused)

class ContrastiveLoss(nn.Module):
    """Contrastive loss for learning better representations."""
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create positive pairs mask
        labels = labels.view(-1, 1)
        positive_mask = torch.eq(labels, labels.T).float()
        negative_mask = 1 - positive_mask
        
        # Remove diagonal (self-similarity)
        positive_mask.fill_diagonal_(0)
        
        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        sum_negative = torch.sum(exp_sim * negative_mask, dim=1, keepdim=True)
        
        loss = 0
        num_positives = 0
        
        for i in range(len(labels)):
            positive_indices = positive_mask[i].nonzero(as_tuple=True)[0]
            if len(positive_indices) > 0:
                positive_sim = similarity_matrix[i, positive_indices]
                positive_loss = -torch.log(
                    torch.exp(positive_sim) / (torch.exp(positive_sim) + sum_negative[i])
                ).mean()
                loss += positive_loss
                num_positives += 1
        
        if num_positives > 0:
            loss = loss / num_positives
        else:
            loss = torch.tensor(0.0, requires_grad=True, device=features.device)
        
        return loss

class MetaLearningOptimizer:
    """Meta-learning approach for better optimization."""
    
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.inner_lr = 1e-4
    
    def meta_update(self, support_data, query_data, num_inner_steps=5):
        """Perform meta-learning update using MAML-style approach."""
        # Save original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        # Inner loop updates on support set
        for _ in range(num_inner_steps):
            support_loss = self.compute_loss(support_data)
            grads = torch.autograd.grad(support_loss, self.model.parameters(), create_graph=True)
            
            # Update parameters
            for (name, param), grad in zip(self.model.named_parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        
        # Compute meta loss on query set
        query_loss = self.compute_loss(query_data)
        
        # Meta update
        self.meta_optimizer.zero_grad()
        query_loss.backward()
        self.meta_optimizer.step()
        
        # Restore original parameters
        for name, param in self.model.named_parameters():
            param.data = original_params[name]
        
        return query_loss.item()
    
    def compute_loss(self, data):
        """Compute loss for given data."""
        inputs, labels = data
        outputs = self.model(**inputs)
        return F.cross_entropy(outputs['logits'], labels)

class ActiveLearningSelector:
    """Active learning for intelligent sample selection."""
    
    def __init__(self, strategy='uncertainty'):
        self.strategy = strategy
    
    def select_samples(self, model, unlabeled_data, num_samples):
        """Select most informative samples for labeling."""
        model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for batch in unlabeled_data:
                outputs = model(**batch)
                
                if self.strategy == 'uncertainty':
                    # Use prediction uncertainty
                    probs = F.softmax(outputs['logits'], dim=-1)
                    uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
                elif self.strategy == 'margin':
                    # Use prediction margin
                    probs = F.softmax(outputs['logits'], dim=-1)
                    top2 = torch.topk(probs, 2, dim=-1)[0]
                    uncertainty = -(top2[:, 0] - top2[:, 1])
                elif self.strategy == 'model_uncertainty':
                    # Use model's uncertainty estimation
                    uncertainty = outputs.get('uncertainty', torch.zeros(len(batch)))
                
                uncertainties.extend(uncertainty.cpu().numpy())
        
        # Select top uncertain samples
        top_indices = np.argsort(uncertainties)[-num_samples:]
        return top_indices

class ModelEnsemble:
    """Ensemble of multiple models for better performance."""
    
    def __init__(self, models):
        self.models = models
    
    def predict(self, inputs):
        """Ensemble prediction with uncertainty."""
        all_predictions = []
        all_uncertainties = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs['logits'], dim=-1)
                all_predictions.append(probs)
                
                if 'uncertainty' in outputs:
                    all_uncertainties.append(outputs['uncertainty'])
        
        # Average predictions
        ensemble_pred = torch.stack(all_predictions).mean(dim=0)
        
        # Calculate prediction variance as uncertainty
        prediction_variance = torch.stack(all_predictions).var(dim=0).mean(dim=-1, keepdim=True)
        
        if all_uncertainties:
            model_uncertainty = torch.stack(all_uncertainties).mean(dim=0)
            total_uncertainty = 0.5 * prediction_variance + 0.5 * model_uncertainty
        else:
            total_uncertainty = prediction_variance
        
        return {
            'predictions': ensemble_pred,
            'uncertainty': total_uncertainty
        }

class AdversarialTraining:
    """Adversarial training for model robustness."""
    
    def __init__(self, epsilon=0.01, alpha=0.005, num_steps=10):
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def pgd_attack(self, model, inputs, labels):
        """Generate adversarial examples using PGD."""
        adv_inputs = inputs['input_values'].clone().detach()
        adv_inputs.requires_grad_(True)
        
        for _ in range(self.num_steps):
            outputs = model(input_values=adv_inputs, attention_mask=inputs['attention_mask'])
            loss = F.cross_entropy(outputs['logits'], labels)
            
            grad = torch.autograd.grad(loss, adv_inputs, retain_graph=False, create_graph=False)[0]
            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            
            # Project to epsilon ball
            delta = torch.clamp(adv_inputs - inputs['input_values'], -self.epsilon, self.epsilon)
            adv_inputs = torch.clamp(inputs['input_values'] + delta, -1, 1).detach()
            adv_inputs.requires_grad_(True)
        
        return {'input_values': adv_inputs, 'attention_mask': inputs['attention_mask']}
    
    def train_step(self, model, optimizer, inputs, labels):
        """Adversarial training step."""
        # Clean training
        clean_outputs = model(**inputs)
        clean_loss = F.cross_entropy(clean_outputs['logits'], labels)
        
        # Adversarial training
        adv_inputs = self.pgd_attack(model, inputs, labels)
        adv_outputs = model(**adv_inputs)
        adv_loss = F.cross_entropy(adv_outputs['logits'], labels)
        
        # Combined loss
        total_loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()

# Advanced evaluation metrics and visualization
class AdvancedEvaluator:
    """Advanced evaluation with detailed metrics and visualizations."""
    
    def __init__(self, class_names):
        self.class_names = class_names
    
    def evaluate_with_uncertainty(self, model, dataloader):
        """Evaluate model with uncertainty quantification."""
        model.eval()
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                labels = batch['labels']
                
                outputs = model(**inputs)
                predictions = torch.argmax(outputs['logits'], dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if 'uncertainty' in outputs:
                    all_uncertainties.extend(outputs['uncertainty'].cpu().numpy())
        
        # Calculate metrics
        report = classification_report(
            all_labels, all_predictions, 
            target_names=self.class_names, 
            output_dict=True
        )
        
        results = {
            'classification_report': report,
            'predictions': all_predictions,
            'labels': all_labels,
            'uncertainties': all_uncertainties
        }
        
        return results
    
    def plot_uncertainty_analysis(self, results, save_path=None):
        """Plot uncertainty analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Uncertainty distribution
        axes[0, 0].hist(results['uncertainties'], bins=50, alpha=0.7)
        axes[0, 0].set_title('Uncertainty Distribution')
        axes[0, 0].set_xlabel('Uncertainty')
        axes[0, 0].set_ylabel('Frequency')
        
        # Uncertainty vs Accuracy
        correct_predictions = np.array(results['predictions']) == np.array(results['labels'])
        axes[0, 1].scatter(results['uncertainties'], correct_predictions.astype(int), alpha=0.6)
        axes[0, 1].set_title('Uncertainty vs Accuracy')
        axes[0, 1].set_xlabel('Uncertainty')
        axes[0, 1].set_ylabel('Correct Prediction')
        
        # Class-wise uncertainty
        class_uncertainties = {}
        for i, class_name in enumerate(self.class_names):
            mask = np.array(results['labels']) == i
            if np.any(mask):
                class_uncertainties[class_name] = np.array(results['uncertainties'])[mask]
        
        axes[1, 0].boxplot(list(class_uncertainties.values()), labels=list(class_uncertainties.keys()))
        axes[1, 0].set_title('Class-wise Uncertainty Distribution')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Confusion matrix with uncertainty
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(results['labels'], results['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calibration_analysis(self, results):
        """Analyze model calibration."""
        uncertainties = np.array(results['uncertainties'])
        correct = (np.array(results['predictions']) == np.array(results['labels'])).astype(int)
        
        # Bin by uncertainty
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        bin_confidences = []
        
        for i in range(len(bins) - 1):
            mask = (uncertainties >= bins[i]) & (uncertainties < bins[i + 1])
            if np.any(mask):
                bin_accuracies.append(np.mean(correct[mask]))
                bin_confidences.append(1 - np.mean(uncertainties[mask]))  # Convert uncertainty to confidence
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
        
        # Plot calibration curve
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.plot(bin_confidences, bin_accuracies, 'o-', label='Model Calibration')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return {
            'bin_centers': bin_centers,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences
        }
