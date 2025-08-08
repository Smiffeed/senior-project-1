#!/usr/bin/env python3
"""
Quick fix for scheduler configuration issue
"""

import re

def fix_scheduler_config():
    file_path = 'scripts/fine_tune_wav2vec2_sen_ham_CW.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace both occurrences of the problematic scheduler configuration
    old_pattern = r'lr_scheduler_type="cosine_with_restarts",\s*lr_scheduler_kwargs=\{"T_0": 500, "T_mult": 2\},'
    new_replacement = 'lr_scheduler_type="cosine",  # Fixed scheduler type'
    
    content = re.sub(old_pattern, new_replacement, content)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Fixed scheduler configuration in the training script")

if __name__ == "__main__":
    fix_scheduler_config()
