#!/usr/bin/env python3
"""Test import issues"""

print("Starting import test...")

try:
    print("1. Importing pandas...")
    import pandas as pd
    print("   ✓ pandas imported")
except Exception as e:
    print(f"   ✗ pandas failed: {e}")

try:
    print("2. Importing numpy...")
    import numpy as np
    print("   ✓ numpy imported")
except Exception as e:
    print(f"   ✗ numpy failed: {e}")

try:
    print("3. Importing sklearn...")
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    print("   ✓ sklearn imported")
except Exception as e:
    print(f"   ✗ sklearn failed: {e}")

try:
    print("4. Importing torch...")
    import torch
    print("   ✓ torch imported")
except Exception as e:
    print(f"   ✗ torch failed: {e}")

try:
    print("5. Importing transformers...")
    from transformers import Wav2Vec2ForSequenceClassification
    print("   ✓ transformers imported")
except Exception as e:
    print(f"   ✗ transformers failed: {e}")

print("Import test completed!")
