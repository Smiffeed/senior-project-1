#!/usr/bin/env python3
"""
Setup script for the Advanced Thai Profanity Detection Model.
This script helps you set up the environment and verify everything works.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors."""
    print(f"🔧 {description}")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"   ✅ Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stdout:
            print(f"   stdout: {e.stdout}")
        if e.stderr:
            print(f"   stderr: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is suitable."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    print(f"   Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ is required")
        return False
    
    print("   ✅ Python version is suitable")
    return True

def check_gpu_availability():
    """Check if GPU is available for training."""
    print("🖥️ Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("   ⚠️ No GPU available, will use CPU (training will be slower)")
            return False
    except ImportError:
        print("   ⚠️ PyTorch not installed yet, will check after installation")
        return False

def install_requirements():
    """Install required packages."""
    print("📦 Installing requirements...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"   ❌ Requirements file not found: {requirements_file}")
        return False
    
    # Install core requirements
    command = f"{sys.executable} -m pip install -r {requirements_file}"
    return run_command(command, "Installing Python packages")

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "models",
        "experiments", 
        "logs",
        "evaluation_results",
        "plots"
    ]
    
    base_path = Path(__file__).parent
    
    for directory in directories:
        dir_path = base_path / directory
        try:
            dir_path.mkdir(exist_ok=True)
            print(f"   ✅ Created/verified: {directory}")
        except Exception as e:
            print(f"   ❌ Failed to create {directory}: {e}")
            return False
    
    return True

def verify_data():
    """Verify that the dataset is available."""
    print("📊 Verifying dataset...")
    
    csv_path = Path(__file__).parent / "csv" / "main.csv"
    
    if not csv_path.exists():
        print(f"   ❌ Dataset not found: {csv_path}")
        print("   Please ensure your dataset is in the correct location")
        return False
    
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        print(f"   ✅ Dataset loaded: {len(df)} samples")
        print(f"   Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['file_path', 'start_time', 'end_time', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"   ⚠️ Missing columns: {missing_columns}")
            return False
        
        # Show label distribution
        label_counts = df['label'].value_counts()
        print(f"   Label distribution:")
        for label, count in label_counts.head().items():
            print(f"     {label}: {count}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error reading dataset: {e}")
        return False

def verify_audio_files():
    """Verify that some audio files exist."""
    print("🎵 Verifying audio files...")
    
    # Check if main audio directory exists
    audio_dirs = [
        Path(__file__).parent / "main",
        Path(__file__).parent / "main_clean"
    ]
    
    found_audio = False
    
    for audio_dir in audio_dirs:
        if audio_dir.exists():
            audio_files = list(audio_dir.glob("*.wav"))
            if audio_files:
                print(f"   ✅ Found {len(audio_files)} audio files in {audio_dir.name}")
                found_audio = True
                break
    
    if not found_audio:
        print("   ⚠️ No audio files found. Please ensure audio files are in the correct location")
        return False
    
    return True

def run_system_test():
    """Run the system test."""
    print("🧪 Running system test...")
    
    test_script = Path(__file__).parent / "test_system.py"
    
    if not test_script.exists():
        print(f"   ❌ Test script not found: {test_script}")
        return False
    
    command = f"{sys.executable} {test_script}"
    return run_command(command, "Running system verification tests")

def print_next_steps():
    """Print next steps for the user."""
    print("\n🎉 Setup completed!")
    print("\n📋 Next Steps:")
    print("="*50)
    
    steps = [
        "1. Test the system:",
        "   python test_system.py",
        "",
        "2. Run basic training:",
        "   python scripts/ultimate_model_training.py",
        "",
        "3. Run ablation study:",
        "   python run_experiments.py --action ablation",
        "",
        "4. Evaluate models:",
        "   python scripts/comprehensive_evaluation.py",
        "",
        "5. Compare experiments:",
        "   python run_experiments.py --action list",
        "   python run_experiments.py --action compare --experiments exp1 exp2"
    ]
    
    for step in steps:
        print(step)
    
    print("\n📚 Documentation:")
    print("- Read ADVANCED_README.md for detailed information")
    print("- Check config/advanced_training_config.json for configuration options")
    print("- Look at scripts/ directory for implementation details")
    
    print("\n⚡ Performance Tips:")
    print("- Use GPU for faster training (if available)")
    print("- Start with smaller experiments before full training")
    print("- Monitor memory usage and adjust batch size if needed")
    print("- Use ensemble methods for best performance")

def main():
    """Main setup function."""
    print("🚀 Advanced Thai Profanity Detection Model - Setup")
    print("="*60)
    
    # Check prerequisites
    checks = [
        ("Python Version", check_python_version),
        ("GPU Availability", check_gpu_availability),
        ("Create Directories", create_directories),
        ("Install Requirements", install_requirements),
        ("Verify Dataset", verify_data),
        ("Verify Audio Files", verify_audio_files),
    ]
    
    failed_checks = []
    
    for check_name, check_func in checks:
        print(f"\n[CHECK] {check_name}")
        try:
            if not check_func():
                failed_checks.append(check_name)
        except Exception as e:
            print(f"   ❌ Unexpected error in {check_name}: {e}")
            failed_checks.append(check_name)
    
    print(f"\n{'='*60}")
    print(f"📋 SETUP RESULTS")
    print(f"{'='*60}")
    
    if failed_checks:
        print(f"❌ {len(failed_checks)} checks failed:")
        for check in failed_checks:
            print(f"   - {check}")
        print("\nPlease address these issues before proceeding.")
        
        print("\n🔧 Common Solutions:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Check dataset file paths and format")
        print("- Verify audio files are in the correct directories")
        print("- Ensure Python 3.8+ is installed")
        
        return False
    else:
        print("✅ All checks passed!")
        
        # Optional: Run system test
        print(f"\n[OPTIONAL] Running system test...")
        test_passed = run_system_test()
        
        if test_passed:
            print("✅ System test passed!")
        else:
            print("⚠️ System test had some issues, but setup basics are complete")
        
        print_next_steps()
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
