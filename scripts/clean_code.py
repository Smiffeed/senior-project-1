import torch
import torchaudio
import pandas as pd
import numpy as np
from transformers import Trainer
import torch.nn as nn

# Classes
labels = {
    'none': 0,
    'เย็ด': 1,
    'กู': 2,
    'มึง': 3,
    'เหี้ย': 4,
    'ควย': 5,
    'สวะ': 6,
    'หี': 7,
    'แตด': 8
}

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Feed Inputs to model and extract logits
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Extract labels
        labels = inputs.get("labels")
        # Define loss function with class weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss