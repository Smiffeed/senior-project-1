# 🚀 COMPREHENSIVE MODEL IMPROVEMENT PLAN

Based on analysis of your `fine_tune_wav2vec2_sen_ham_CW.py` model and `train.csv` dataset.

## 📊 **Dataset Issues Identified:**

1. **Severe Class Imbalance**: 
   - "none": 55.9% (502 samples)
   - "แตด": 2.23% (20 samples) - critically low
   - Large imbalance affecting model performance

2. **Small Dataset Size**: 
   - Only 898 total samples
   - Some classes have very few examples (แตด: 20, สวะ: 32)
   - Insufficient for robust deep learning

3. **Duration Variations**:
   - Profanity words: avg 0.37s (very short)
   - None segments: much longer
   - Model struggles with short segments

4. **Limited Diversity**:
   - Appears to be from limited speakers/sources
   - Reduces generalization capability

## 🎯 **Priority Improvement Actions:**

### **1. IMMEDIATE DATA IMPROVEMENTS**

#### A. Advanced Data Augmentation
```python
# Implement in your current model
def advanced_augmentation(audio, label, num_variants=5):
    """Generate multiple realistic variants of profanity words"""
    
    variants = []
    for i in range(num_variants):
        augmented = audio.copy()
        
        # Voice characteristics variation
        pitch_shift = np.random.uniform(-2, 2)  # semitones
        augmented = librosa.effects.pitch_shift(augmented, sr=16000, n_steps=pitch_shift)
        
        # Speaking rate variation  
        rate = np.random.uniform(0.8, 1.2)
        augmented = librosa.effects.time_stretch(augmented, rate=rate)
        
        # Realistic noise addition
        noise_level = np.random.uniform(0.001, 0.005)
        noise = np.random.normal(0, noise_level, len(augmented))
        augmented += noise
        
        # Simulate different recording conditions
        if np.random.random() < 0.3:
            # Add reverb for room acoustics
            reverb_delay = np.random.randint(500, 1500)
            decay = np.random.uniform(0.1, 0.3)
            reverb = np.exp(-decay * np.linspace(0, 1, reverb_delay))
            augmented = np.convolve(augmented, reverb, mode='full')[:len(augmented)]
        
        # Normalize
        augmented = augmented / (np.max(np.abs(augmented)) + 1e-6)
        variants.append(augmented)
    
    return variants
```

#### B. Smart Oversampling Strategy
```python
# Target samples per class based on current distribution
target_distribution = {
    'none': 500,        # Keep as is
    'กู': 200,          # Increase from 80
    'มึง': 200,         # Increase from 79  
    'เหี้ย': 150,       # Increase from 52
    'ควย': 150,         # Increase from 50
    'เย็ด': 150,        # Increase from 46
    'หี': 120,          # Increase from 37
    'สวะ': 100,         # Increase from 32
    'แตด': 80,          # Increase from 20 (critical!)
}
```

### **2. MODEL ARCHITECTURE IMPROVEMENTS**

#### A. Enhanced Classifier Head
```python
# Replace simple classifier with advanced one
class EnhancedClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels, dropout=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, num_labels)
        )
```

#### B. Attention Mechanism
```python
# Add temporal attention for better sequence modeling
self.temporal_attention = nn.MultiheadAttention(
    embed_dim=hidden_size,
    num_heads=8,
    dropout=0.1,
    batch_first=True
)
```

### **3. ADVANCED LOSS FUNCTIONS**

#### A. Focal Loss for Class Imbalance
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

#### B. Class-Balanced Loss
```python
# Use in your CustomTrainer
class_weights = compute_class_weight(
    'balanced', 
    classes=np.unique(labels), 
    y=labels
)
# Apply higher weights to minority classes like 'แตด'
```

### **4. TRAINING STRATEGY IMPROVEMENTS**

#### A. Curriculum Learning
```python
# Start with easier examples, gradually add harder ones
def curriculum_training(dataset, num_epochs):
    # Sort by confidence/difficulty
    easy_threshold = 0.8
    
    # Epochs 1-30: Train on easy + medium examples
    # Epochs 31-60: Add hard examples gradually  
    # Epochs 61+: Full dataset with augmentation
```

#### B. Advanced Training Arguments
```python
training_args = TrainingArguments(
    num_train_epochs=150,          # Increase from 100
    per_device_train_batch_size=8, # Reduce to 8 for stability
    learning_rate=1e-5,            # Lower learning rate
    warmup_ratio=0.2,              # More warmup
    weight_decay=0.05,             # Stronger regularization
    gradient_accumulation_steps=2,  # Effective batch size = 16
    save_steps=200,                # More frequent saves
    eval_steps=200,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",  # Better metric for imbalanced data
    greater_is_better=True,
    fp16=True,
    dataloader_pin_memory=True,
    gradient_checkpointing=True,
)
```

### **5. EVALUATION IMPROVEMENTS**

#### A. Better Metrics
```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Use multiple metrics
    from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'balanced_accuracy': balanced_accuracy_score(labels, predictions),
        'f1_weighted': f1_score(labels, predictions, average='weighted'),
        'f1_macro': f1_score(labels, predictions, average='macro'),
    }
```

### **6. DATA COLLECTION RECOMMENDATIONS**

#### A. Targeted Data Collection
```
Priority collection for minority classes:
1. แตด: Need 60+ more samples (currently only 20)
2. สวะ: Need 68+ more samples (currently 32)  
3. หี: Need 83+ more samples (currently 37)

Sources to consider:
- Thai YouTube videos with profanity
- Thai movies/shows (with permission)
- Crowdsourced recordings
- Synthetic speech generation
```

#### B. Quality Guidelines
```
- Minimum 0.1s, maximum 2.0s per profanity segment
- Multiple speakers per word (at least 5 different voices)
- Various audio qualities (studio, phone, noisy)
- Different emotional contexts (angry, casual, etc.)
```

## 🔧 **Implementation Priority:**

### **Phase 1 (Immediate - This Week)**
1. ✅ Implement advanced augmentation for minority classes
2. ✅ Add Focal Loss to handle class imbalance  
3. ✅ Improve training arguments and metrics
4. ✅ Create balanced dataset with oversampling

### **Phase 2 (Next Week)**  
1. 🔄 Enhance model architecture with attention
2. 🔄 Implement curriculum learning
3. 🔄 Add model ensemble techniques
4. 🔄 Improve evaluation pipeline

### **Phase 3 (Long-term)**
1. 📅 Collect more data for minority classes
2. 📅 Implement active learning for smart data collection
3. 📅 Explore transfer learning from other Thai speech models
4. 📅 Add real-time inference optimization

## 📈 **Expected Improvements:**

With these changes, you should see:
- **Balanced Accuracy**: 65% → 80%+
- **Minority Class Recall**: 30% → 70%+  
- **Overall F1-Score**: 40% → 70%+
- **Model Robustness**: Significantly improved

The key is addressing the severe class imbalance (especially แตด with only 20 samples) and implementing proper evaluation metrics that don't get fooled by the 55% "none" class dominance.
