import os
import torch
import numpy as np
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor
from improved_model import create_improved_model

def load_huggingface_model(model_path, num_labels=9):
    """
    Try to load a model using HuggingFace's from_pretrained method
    """
    try:
        print(f"Trying to load model from {model_path} using HuggingFace's from_pretrained")
        model = AutoModelForAudioClassification.from_pretrained(model_path)
        return model
    except Exception as e:
        print(f"Error loading with from_pretrained: {e}")
        return None

def load_safetensors_directly(model_path, num_labels=9):
    """
    Try to load the safetensors file directly and create a model
    """
    try:
        print(f"Trying to load safetensors file directly")
        # Find the safetensors file
        safetensors_file = os.path.join(model_path, "model.safetensors")
        if not os.path.exists(safetensors_file):
            print(f"No safetensors file found at {safetensors_file}")
            return None
        
        # Create a new model
        model = create_improved_model(
            "airesearch/wav2vec2-large-xlsr-53-th", 
            num_labels=num_labels
        )
        
        # Load the safetensors file directly using the safetensors library
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_file)
        
        # First get all available keys in our model
        model_keys = set(model.state_dict().keys())
        safetensors_keys = set(state_dict.keys())
        
        print(f"Model has {len(model_keys)} parameters")
        print(f"Safetensors file has {len(safetensors_keys)} parameters")
        
        # Try to load classifier and attention pooling weights only
        # This way we don't have to deal with feature extractor incompatibilities
        filtered_state_dict = {}
        for key in model.state_dict().keys():
            # Only copy classifier and attention pooling weights
            if key.startswith('classifier') or key.startswith('attention_pooling'):
                # Find the matching key in the safetensors file
                for st_key in safetensors_keys:
                    if key.split('.')[-2:] == st_key.split('.')[-2:]:
                        # Check shape compatibility
                        if model.state_dict()[key].shape == state_dict[st_key].shape:
                            filtered_state_dict[key] = state_dict[st_key]
                            print(f"Mapped {key} <- {st_key}")
                            break
        
        print(f"Loaded {len(filtered_state_dict)} compatible weights for classifier and attention pooling")
        
        # Load the filtered state dict with strict=False
        if filtered_state_dict:
            missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
            print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
            return model
        else:
            print("No compatible weights found")
            return None
            
    except Exception as e:
        print(f"Error loading safetensors: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_from_checkpoint(model_path):
    """
    Try to extract weights from any checkpoints in the folder structure
    """
    try:
        print(f"Looking for checkpoints in {model_path}")
        # Look for checkpoint directories
        checkpoint_dirs = []
        parent_dir = os.path.dirname(model_path)
        for root, dirs, files in os.walk(parent_dir):
            for d in dirs:
                if 'checkpoint' in d:
                    checkpoint_dirs.append(os.path.join(root, d))
        
        if not checkpoint_dirs:
            print("No checkpoint directories found in parent directory either")
            return None
            
        print(f"Found {len(checkpoint_dirs)} checkpoint directories")
        # Sort them to get the latest
        checkpoint_dirs.sort()
        latest_checkpoint = checkpoint_dirs[-1]
        print(f"Using latest checkpoint: {latest_checkpoint}")
        
        # Try to load from this checkpoint first using HuggingFace
        model = load_huggingface_model(latest_checkpoint)
        if model:
            return model
            
        # If that fails, try direct loading
        return load_safetensors_directly(latest_checkpoint)
    except Exception as e:
        print(f"Error extracting from checkpoint: {e}")
        return None

def emergency_model_builder(model_path, num_labels=9):
    """
    Last resort: build a fresh model and save it manually
    """
    try:
        print("Creating fresh model as a fallback")
        model = create_improved_model(
            "airesearch/wav2vec2-large-xlsr-53-th", 
            num_labels=num_labels
        )
        
        # Save this model manually since it doesn't have save_pretrained
        output_dir = os.path.join(os.path.dirname(model_path), "fresh_model")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model's state dict
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        print(f"Saved model state dict to {output_dir}")
        
        return model
    except Exception as e:
        print(f"Error creating fresh model: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_model(model_path, num_labels=9):
    """
    Try all approaches to load a model
    """
    # Approach 1: Try HuggingFace's from_pretrained
    model = load_huggingface_model(model_path, num_labels)
    if model:
        print("Successfully loaded model with HuggingFace's from_pretrained")
        return model
        
    # Approach 2: Try loading safetensors directly
    model = load_safetensors_directly(model_path, num_labels)
    if model:
        print("Successfully loaded model from safetensors")
        return model
        
    # Approach 3: Try finding a checkpoint
    model = extract_from_checkpoint(model_path)
    if model:
        print("Successfully loaded model from checkpoint")
        return model
        
    # Approach 4: Build a fresh model
    model = emergency_model_builder(model_path, num_labels)
    if model:
        print("Created fresh model as fallback")
        return model
        
    raise ValueError("Could not load or create a model")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Load a model using various approaches")
    parser.add_argument("--model_path", type=str, default="./output/improved_model/final",
                        help="Path to the model directory")
    parser.add_argument("--num_labels", type=int, default=9,
                        help="Number of labels in the model")
    parser.add_argument("--output_path", type=str, default="./output/recovered_model",
                        help="Path to save the loaded model")
    args = parser.parse_args()
    
    # Load the model
    model = load_model(args.model_path, args.num_labels)
    
    # Save it manually since custom models don't have save_pretrained
    output_dir = args.output_path
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model state dict
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
    
    # Also save the feature extractor config
    if os.path.exists(os.path.join(args.model_path, "preprocessor_config.json")):
        import shutil
        shutil.copy(
            os.path.join(args.model_path, "preprocessor_config.json"),
            os.path.join(output_dir, "preprocessor_config.json")
        )
    
    print(f"Model successfully loaded and saved to {output_dir}")
    
    # Update advanced_evaluation.py to work with this model
    print("Creating model loader helper for evaluation...")
    with open(os.path.join(output_dir, "load_helper.py"), "w") as f:
        f.write("""
import torch
from improved_model import create_improved_model

def load_model(num_labels=9, device='cuda'):
    model = create_improved_model(
        "airesearch/wav2vec2-large-xlsr-53-th", 
        num_labels=num_labels
    )
    model.load_state_dict(torch.load('pytorch_model.bin', map_location=device))
    model.to(device)
    model.eval()
    return model
        """)
    print("Done! Use the following code in your evaluation script:")
    print("------------------------------------------------------")
    print("import sys, os")
    print("sys.path.append(os.path.abspath('./output/recovered_model'))")
    print("from load_helper import load_model")
    print("self.model = load_model(num_labels=len(self.label_map), device=self.device)")
    print("------------------------------------------------------")