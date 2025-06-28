#!/usr/bin/env python3
"""
🚀 INTEGRATION EXAMPLES & USAGE GUIDE
Practical examples showing how to integrate the unified profanity detection system
into different types of applications and use cases.

This guide covers:
1. Basic usage examples
2. Configuration for different scenarios  
3. Integration patterns
4. Error handling
5. Performance optimization
6. Production deployment considerations
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

def example_basic_usage():
    """Example 1: Basic usage - Simple profanity detection and censoring"""
    print("🔧 EXAMPLE 1: Basic Usage")
    print("=" * 50)
    print("Basic profanity detection with default settings")
    print()
    
    # Check if we have test files
    test_files = list(Path('./eval').glob('*.wav')) if Path('./eval').exists() else []
    if not test_files:
        print("❌ No test files found in ./eval directory")
        print("Please add .wav files to test with.")
        return
    
    input_file = str(test_files[0])
    output_file = './example_basic_output.wav'
    
    print(f"📁 Input: {input_file}")
    print(f"💾 Output: {output_file}")
    print()
    
    try:
        # Import and initialize the system
        from unified_profanity_system import UnifiedProfanitySystem
        
        # Basic usage - just initialize and process
        system = UnifiedProfanitySystem()
        
        print("🚀 Processing with default settings...")
        results = system.process_audio(input_file, output_file)
        
        if results['success']:
            print("✅ Processing completed successfully!")
            print(f"   Detections found: {results['detections']['total_count']}")
            print(f"   Processing time: {results['performance']['total_processing_time']:.2f}s")
            print(f"   Efficiency gain: {results['performance']['efficiency_gain']:.1f}%")
        else:
            print(f"❌ Processing failed: {results.get('error')}")
            
    except Exception as e:
        print(f"❌ Example failed: {e}")
    
    print("\n" + "="*50)

def example_custom_configuration():
    """Example 2: Custom configuration for specific use cases"""
    print("⚙️ EXAMPLE 2: Custom Configuration")
    print("=" * 50)
    print("Configuring the system for different scenarios")
    print()
    
    # Example configurations for different use cases
    configurations = {
        "real_time": {
            "description": "Optimized for real-time processing",
            "config": {
                "vad_enabled": True,
                "multi_stage_detection": False,
                "context_analysis": False,
                "window_sizes": {"balanced": 0.5},
                "advanced_preprocessing": False,
                "censor_method": "beep"
            }
        },
        "maximum_accuracy": {
            "description": "Maximum accuracy for content moderation",
            "config": {
                "vad_enabled": True,
                "multi_stage_detection": True,
                "context_analysis": True,
                "window_sizes": {"precision": 0.25},
                "confidence_thresholds": {"minimum_detection": 0.2},
                "advanced_preprocessing": True,
                "censor_method": "silence"
            }
        },
        "batch_processing": {
            "description": "Optimized for batch processing large files",
            "config": {
                "vad_enabled": True,
                "adaptive_windowing": True,
                "multi_stage_detection": True,
                "context_analysis": True,
                "performance_monitoring": True,
                "generate_report": True
            }
        }
    }
    
    # Demonstrate each configuration
    for config_name, config_info in configurations.items():
        print(f"🔧 {config_name.upper()} Configuration:")
        print(f"   Purpose: {config_info['description']}")
        print("   Settings:")
        for key, value in config_info['config'].items():
            print(f"     {key}: {value}")
        print()
    
    # Show how to use custom configuration
    print("💻 Code Example:")
    print("""
    from unified_profanity_system import UnifiedProfanitySystem
    
    # Create custom configuration
    config = {
        "vad_enabled": True,
        "multi_stage_detection": False,
        "censor_method": "beep",
        "confidence_thresholds": {"minimum_detection": 0.4}
    }
    
    # Initialize with custom config
    system = UnifiedProfanitySystem()
    system.config.update(config)
    
    # Process audio
    results = system.process_audio('input.wav', 'output.wav')
    """)
    
    print("\n" + "="*50)

def example_batch_processing():
    """Example 3: Batch processing multiple files"""
    print("📁 EXAMPLE 3: Batch Processing")
    print("=" * 50)
    print("Processing multiple files efficiently")
    print()
    
    # Find test files
    test_files = list(Path('./eval').glob('*.wav')) if Path('./eval').exists() else []
    if len(test_files) < 2:
        print("❌ Need at least 2 test files for batch processing example")
        return
    
    # Use first 3 files
    test_files = test_files[:3]
    
    print(f"📁 Processing {len(test_files)} files:")
    for i, file in enumerate(test_files, 1):
        print(f"   {i}. {file.name}")
    print()
    
    try:
        from unified_profanity_system import UnifiedProfanitySystem
        
        # Initialize system once for batch processing
        system = UnifiedProfanitySystem()
        
        # Configure for batch processing
        system.config.update({
            "generate_report": False,  # Skip individual reports
            "performance_monitoring": True,
            "detailed_logging": False  # Reduce console output
        })
        
        batch_results = []
        total_start_time = time.time()
        
        print("🚀 Starting batch processing...")
        
        for i, input_file in enumerate(test_files, 1):
            output_file = f'./example_batch_{i}.wav'
            
            print(f"   📁 File {i}/{len(test_files)}: {input_file.name}")
            
            file_start_time = time.time()
            results = system.process_audio(str(input_file), output_file, generate_report=False)
            file_time = time.time() - file_start_time
            
            if results['success']:
                print(f"      ✅ Completed in {file_time:.2f}s")
                print(f"      🎯 Detections: {results['detections']['total_count']}")
                batch_results.append(results)
            else:
                print(f"      ❌ Failed: {results.get('error')}")
        
        total_time = time.time() - total_start_time
        
        # Batch summary
        print(f"\n📊 BATCH PROCESSING SUMMARY:")
        print(f"   Files processed: {len(batch_results)}/{len(test_files)}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average time per file: {total_time/max(1, len(batch_results)):.2f}s")
        
        total_detections = sum(r['detections']['total_count'] for r in batch_results)
        print(f"   Total detections: {total_detections}")
        
        # Session statistics from the last result
        if batch_results:
            session_stats = batch_results[-1]['session_stats']
            print(f"   Average efficiency gain: {session_stats['average_efficiency_gain']:.1f}%")
        
    except Exception as e:
        print(f"❌ Batch processing example failed: {e}")
    
    print("\n" + "="*50)

def example_error_handling():
    """Example 4: Proper error handling and validation"""
    print("🛡️ EXAMPLE 4: Error Handling")
    print("=" * 50)
    print("Robust error handling for production use")
    print()
    
    print("💻 Code Example:")
    print("""
    from unified_profanity_system import UnifiedProfanitySystem
    import os
    from pathlib import Path
    
    def safe_profanity_detection(input_file: str, output_file: str) -> Dict:
        '''Safely process audio with comprehensive error handling.'''
        
        # Input validation
        if not os.path.exists(input_file):
            return {'success': False, 'error': f'Input file not found: {input_file}'}
        
        # Check file format
        if not input_file.lower().endswith(('.wav', '.mp3', '.flac')):
            return {'success': False, 'error': 'Unsupported audio format'}
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Initialize system with error handling
            system = UnifiedProfanitySystem()
            
            if system.model is None:
                return {'success': False, 'error': 'System not properly initialized'}
            
            # Process with timeout handling
            results = system.process_audio(input_file, output_file)
            
            # Validate results
            if not results.get('success'):
                return {'success': False, 'error': results.get('error', 'Unknown error')}
            
            # Verify output file was created
            if not os.path.exists(output_file):
                return {'success': False, 'error': 'Output file was not created'}
            
            return results
            
        except FileNotFoundError as e:
            return {'success': False, 'error': f'File error: {e}'}
        except MemoryError:
            return {'success': False, 'error': 'Insufficient memory for processing'}
        except Exception as e:
            return {'success': False, 'error': f'Unexpected error: {e}'}
    
    # Usage
    result = safe_profanity_detection('input.wav', 'output.wav')
    
    if result['success']:
        print("✅ Processing successful!")
        print(f"Detections: {result['detections']['total_count']}")
    else:
        print(f"❌ Processing failed: {result['error']}")
        # Handle error appropriately (log, retry, notify user, etc.)
    """)
    
    print("\n" + "="*50)

def example_performance_optimization():
    """Example 5: Performance optimization techniques"""
    print("⚡ EXAMPLE 5: Performance Optimization")
    print("=" * 50)
    print("Optimizing the system for different performance requirements")
    print()
    
    optimization_strategies = {
        "Speed Optimization": {
            "config": {
                "vad_enabled": True,
                "multi_stage_detection": False,
                "context_analysis": False,
                "window_sizes": {"efficiency": 1.0},
                "advanced_preprocessing": False,
                "confidence_thresholds": {"minimum_detection": 0.5}
            },
            "benefits": ["Fastest processing", "Lower CPU usage", "Good for real-time"]
        },
        "Memory Optimization": {
            "config": {
                "vad_enabled": True,
                "adaptive_windowing": True,
                "generate_report": False,
                "detailed_logging": False,
                "performance_monitoring": False
            },
            "benefits": ["Lower memory usage", "Better for large files", "Reduced overhead"]
        },
        "Accuracy Optimization": {
            "config": {
                "multi_stage_detection": True,
                "context_analysis": True,
                "window_sizes": {"precision": 0.25},
                "confidence_thresholds": {"minimum_detection": 0.2},
                "advanced_preprocessing": True
            },
            "benefits": ["Highest accuracy", "Best detection rate", "Good for content moderation"]
        }
    }
    
    for strategy_name, strategy_info in optimization_strategies.items():
        print(f"🎯 {strategy_name}:")
        print("   Configuration:")
        for key, value in strategy_info['config'].items():
            print(f"     {key}: {value}")
        print("   Benefits:")
        for benefit in strategy_info['benefits']:
            print(f"     • {benefit}")
        print()
    
    print("💻 Performance Monitoring Example:")
    print("""
    from unified_profanity_system import UnifiedProfanitySystem
    import time
    import psutil
    import os
    
    def monitor_performance(input_file: str, output_file: str):
        '''Monitor system performance during processing.'''
        
        # Get initial system stats
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        system = UnifiedProfanitySystem()
        
        # Enable performance monitoring
        system.config['performance_monitoring'] = True
        
        start_time = time.time()
        results = system.process_audio(input_file, output_file)
        end_time = time.time()
        
        # Get final system stats
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage = final_memory - initial_memory
        
        print(f"⏱️ Total time: {end_time - start_time:.2f}s")
        print(f"💾 Memory usage: {memory_usage:.1f} MB")
        print(f"📊 Processing efficiency: {results['performance']['efficiency_gain']:.1f}%")
        
        return results
    """)
    
    print("\n" + "="*50)

def example_integration_patterns():
    """Example 6: Integration patterns for different applications"""
    print("🔗 EXAMPLE 6: Integration Patterns")
    print("=" * 50)
    print("Common integration patterns for different applications")
    print()
    
    patterns = {
        "Web API Integration": """
    from flask import Flask, request, jsonify, send_file
    from unified_profanity_system import UnifiedProfanitySystem
    import tempfile
    import os
    
    app = Flask(__name__)
    profanity_system = UnifiedProfanitySystem()
    
    @app.route('/api/censor', methods=['POST'])
    def censor_audio():
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        with tempfile.NamedTemporaryFile(suffix='.wav') as input_temp:
            with tempfile.NamedTemporaryFile(suffix='.wav') as output_temp:
                audio_file.save(input_temp.name)
                
                results = profanity_system.process_audio(
                    input_temp.name, 
                    output_temp.name
                )
                
                if results['success']:
                    return send_file(
                        output_temp.name,
                        as_attachment=True,
                        attachment_filename='censored_audio.wav'
                    )
                else:
                    return jsonify({'error': results['error']}), 500
        """,
        
        "Streaming Processing": """
    from unified_profanity_system import UnifiedProfanitySystem
    import threading
    import queue
    
    class StreamingProfanityProcessor:
        def __init__(self):
            self.system = UnifiedProfanitySystem()
            self.processing_queue = queue.Queue()
            self.worker_thread = threading.Thread(target=self._process_queue)
            self.worker_thread.daemon = True
            self.worker_thread.start()
        
        def submit_for_processing(self, input_file: str, output_file: str):
            '''Submit audio file for asynchronous processing.'''
            self.processing_queue.put((input_file, output_file))
        
        def _process_queue(self):
            '''Background worker thread for processing.'''
            while True:
                input_file, output_file = self.processing_queue.get()
                try:
                    results = self.system.process_audio(input_file, output_file)
                    print(f"✅ Processed: {input_file}")
                except Exception as e:
                    print(f"❌ Failed {input_file}: {e}")
                finally:
                    self.processing_queue.task_done()
        """,
        
        "GUI Application": """
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from unified_profanity_system import UnifiedProfanitySystem
    import threading
    
    class ProfanityDetectorGUI:
        def __init__(self, root):
            self.root = root
            self.system = UnifiedProfanitySystem()
            self.setup_ui()
        
        def setup_ui(self):
            self.root.title("Profanity Detector")
            
            # File selection
            tk.Button(self.root, text="Select Audio File", 
                     command=self.select_file).pack(pady=10)
            
            # Progress bar
            self.progress = ttk.Progressbar(self.root, mode='indeterminate')
            self.progress.pack(pady=10, fill='x')
            
            # Process button
            tk.Button(self.root, text="Process Audio", 
                     command=self.process_audio).pack(pady=10)
        
        def select_file(self):
            self.input_file = filedialog.askopenfilename(
                title="Select Audio File",
                filetypes=[("Audio files", "*.wav *.mp3")]
            )
        
        def process_audio(self):
            if not hasattr(self, 'input_file'):
                messagebox.showerror("Error", "Please select an audio file")
                return
            
            output_file = filedialog.asksaveasfilename(
                title="Save Censored Audio",
                defaultextension=".wav",
                filetypes=[("WAV files", "*.wav")]
            )
            
            if output_file:
                self.progress.start()
                thread = threading.Thread(
                    target=self._process_in_background,
                    args=(self.input_file, output_file)
                )
                thread.start()
        
        def _process_in_background(self, input_file, output_file):
            try:
                results = self.system.process_audio(input_file, output_file)
                self.progress.stop()
                
                if results['success']:
                    messagebox.showinfo("Success", 
                        f"Processing complete!\\n"
                        f"Detections: {results['detections']['total_count']}")
                else:
                    messagebox.showerror("Error", results['error'])
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", str(e))
        """
    }
    
    for pattern_name, code_example in patterns.items():
        print(f"🔧 {pattern_name}:")
        print(code_example)
        print("\n" + "-"*50 + "\n")

def create_configuration_templates():
    """Create configuration templates for common use cases"""
    print("📝 CREATING CONFIGURATION TEMPLATES")
    print("=" * 50)
    
    templates = {
        "real_time_config.json": {
            "description": "Optimized for real-time processing with low latency",
            "config": {
                "vad_enabled": True,
                "multi_stage_detection": False,
                "context_analysis": False,
                "adaptive_windowing": False,
                "advanced_preprocessing": False,
                "window_sizes": {"balanced": 0.5},
                "confidence_thresholds": {"minimum_detection": 0.5},
                "censor_method": "beep",
                "generate_report": False,
                "detailed_logging": False
            }
        },
        "maximum_accuracy_config.json": {
            "description": "Maximum accuracy for content moderation applications",
            "config": {
                "vad_enabled": True,
                "multi_stage_detection": True,
                "context_analysis": True,
                "adaptive_windowing": True,
                "advanced_preprocessing": True,
                "window_sizes": {"precision": 0.25},
                "confidence_thresholds": {"minimum_detection": 0.2},
                "censor_method": "silence",
                "generate_report": True,
                "detailed_logging": True
            }
        },
        "batch_processing_config.json": {
            "description": "Optimized for processing large batches of files",
            "config": {
                "vad_enabled": True,
                "multi_stage_detection": True,
                "context_analysis": True,
                "adaptive_windowing": True,
                "advanced_preprocessing": True,
                "performance_monitoring": True,
                "generate_report": True,
                "detailed_logging": False
            }
        },
        "mobile_optimized_config.json": {
            "description": "Lightweight configuration for mobile/edge devices",
            "config": {
                "vad_enabled": True,
                "multi_stage_detection": False,
                "context_analysis": False,
                "adaptive_windowing": False,
                "advanced_preprocessing": False,
                "window_sizes": {"efficiency": 1.0},
                "confidence_thresholds": {"minimum_detection": 0.6},
                "generate_report": False,
                "performance_monitoring": False
            }
        }
    }
    
    for filename, template_info in templates.items():
        template_data = {
            "name": filename.replace('.json', '').replace('_', ' ').title(),
            "description": template_info["description"],
            "configuration": template_info["config"],
            "usage_example": f"""
# Load and use this configuration:
from unified_profanity_system import UnifiedProfanitySystem

system = UnifiedProfanitySystem(config_path='{filename}')
results = system.process_audio('input.wav', 'output.wav')
""".strip()
        }
        
        # Save template file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created: {filename}")
        print(f"   Purpose: {template_info['description']}")
    
    print(f"\n📁 Created {len(templates)} configuration templates")
    print("These templates can be used as starting points for your specific use case.")

def main():
    """Run all integration examples"""
    print("🚀 UNIFIED PROFANITY DETECTION SYSTEM")
    print("INTEGRATION EXAMPLES & USAGE GUIDE")
    print("=" * 80)
    print()
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Custom Configuration", example_custom_configuration),
        ("Batch Processing", example_batch_processing),
        ("Error Handling", example_error_handling),
        ("Performance Optimization", example_performance_optimization),
        ("Integration Patterns", example_integration_patterns)
    ]
    
    for name, example_func in examples:
        try:
            example_func()
            print()
        except Exception as e:
            print(f"❌ {name} example failed: {e}")
            print()
    
    # Create configuration templates
    print("\n" + "="*80)
    create_configuration_templates()
    
    print("\n" + "="*80)
    print("🎉 INTEGRATION GUIDE COMPLETE")
    print("=" * 80)
    print("The unified profanity detection system is ready for integration!")
    print("Check the generated configuration templates and code examples above.")
    print()
    print("📚 Next Steps:")
    print("1. Choose the appropriate configuration for your use case")
    print("2. Test with your audio files")
    print("3. Integrate using the provided patterns")
    print("4. Monitor performance and adjust as needed")
    print()
    print("🔗 For production deployment:")
    print("• Use appropriate error handling")
    print("• Monitor system resources")
    print("• Configure logging appropriately")
    print("• Test with your specific audio types")

if __name__ == "__main__":
    main()
