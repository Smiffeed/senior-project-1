#!/usr/bin/env python3
"""
Test script to verify Optuna installation and basic functionality
"""

def test_optuna():
    print("🔍 Testing Optuna installation...")
    
    try:
        import optuna
        print(f"✅ Optuna successfully imported! Version: {optuna.__version__}")
        
        # Test basic functionality
        def objective(trial):
            x = trial.suggest_float('x', -10, 10)
            return (x - 2) ** 2
        
        study = optuna.create_study()
        study.optimize(objective, n_trials=5)
        
        print(f"✅ Optuna basic test passed!")
        print(f"Best value: {study.best_value:.4f}")
        print(f"Best params: {study.best_params}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import Optuna: {e}")
        return False
    except Exception as e:
        print(f"❌ Optuna test failed: {e}")
        return False

def test_other_dependencies():
    print("\n🔍 Testing other dependencies...")
    
    dependencies = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
    ]
    
    all_good = True
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} is available")
        except ImportError:
            print(f"❌ {name} is missing")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🧪 Dependency Test for Hyperparameter Optimization")
    print("=" * 50)
    
    optuna_ok = test_optuna()
    deps_ok = test_other_dependencies()
    
    print("\n" + "=" * 50)
    if optuna_ok and deps_ok:
        print("🎉 All dependencies are working correctly!")
        print("You can now run the hyperparameter optimizer with Bayesian optimization.")
    else:
        print("⚠️ Some dependencies are missing or not working.")
        print("Please install missing packages before running the optimizer.")
    
    print("\n💡 To activate your environment and run the optimizer:")
    print("1. env\\Scripts\\activate")
    print("2. python hyperparameter_optimizer.py --method bayesian --trials 10")
