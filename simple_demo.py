#!/usr/bin/env python3
"""
🎤 SIMPLE THAI PROFANITY DETECTION DEMO
Lightweight demo application using basic HTML interface

Features:
- File upload functionality
- Audio processing with your trained models
- Results display with timestamps
- No complex web framework dependencies
"""

import http.server
import socketserver
import json
import os
import tempfile
import urllib.parse
from pathlib import Path
import base64
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
from typing import List, Dict

class SimpleProfanityDemo:
    """Simple demo server for profanity detection"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = None
        self.binary_model = None
        self.multiclass_model = None
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load the trained models"""
        try:
            print("🔄 Loading models...")
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base")
            
            # Try to load trained models
            try:
                self.binary_model = Wav2Vec2ForSequenceClassification.from_pretrained("models/binary_classifier_fast")
                self.multiclass_model = Wav2Vec2ForSequenceClassification.from_pretrained("models/multiclass_classifier_fast")
                print("✅ Loaded trained models")
            except:
                print("⚠️ Using base models (demo mode)")
                self.binary_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=2)
                self.multiclass_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base", num_labels=5)
            
            self.binary_model.to(self.device)
            self.multiclass_model.to(self.device)
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def predict_window(self, audio_window: np.ndarray, model, labels: List[str]) -> Dict:
        """Predict on audio window"""
        try:
            inputs = self.feature_extractor(audio_window, sampling_rate=16000, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            return {'label': labels[predicted_class], 'confidence': confidence}
        except:
            return {'label': labels[0], 'confidence': 0.5}
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """Analyze audio file"""
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            duration = len(audio) / sr
            
            binary_labels = ['Clean', 'Profanity']
            multiclass_labels = ['Clean', 'เย็ด', 'กู', 'มึง', 'เหี้ย']
            
            results = []
            
            # Binary analysis with 2.0s windows
            window_size = 2.0
            stride = 1.0
            current_time = 0
            
            profanity_regions = []
            
            while current_time + window_size <= duration:
                start_sample = int(current_time * sr)
                end_sample = int((current_time + window_size) * sr)
                window_audio = audio[start_sample:end_sample]
                
                # Ensure correct length
                target_samples = int(window_size * sr)
                if len(window_audio) < target_samples:
                    window_audio = np.pad(window_audio, (0, target_samples - len(window_audio)), mode='constant')
                elif len(window_audio) > target_samples:
                    window_audio = window_audio[:target_samples]
                
                prediction = self.predict_window(window_audio, self.binary_model, binary_labels)
                
                if prediction['label'] == 'Profanity' and prediction['confidence'] > 0.6:
                    profanity_regions.append((current_time, current_time + window_size))
                
                current_time += stride
            
            # Multiclass analysis on detected regions
            for region_start, region_end in profanity_regions:
                window_size_fine = 0.3
                stride_fine = 0.15
                current_time = region_start
                
                while current_time + window_size_fine <= region_end:
                    start_sample = int(current_time * sr)
                    end_sample = int((current_time + window_size_fine) * sr)
                    
                    if end_sample <= len(audio):
                        window_audio = audio[start_sample:end_sample]
                        
                        target_samples = int(window_size_fine * sr)
                        if len(window_audio) < target_samples:
                            window_audio = np.pad(window_audio, (0, target_samples - len(window_audio)), mode='constant')
                        elif len(window_audio) > target_samples:
                            window_audio = window_audio[:target_samples]
                        
                        prediction = self.predict_window(window_audio, self.multiclass_model, multiclass_labels)
                        
                        if prediction['label'] != 'Clean' and prediction['confidence'] > 0.5:
                            results.append({
                                'start_time': round(current_time, 2),
                                'end_time': round(current_time + window_size_fine, 2),
                                'word': prediction['label'],
                                'confidence': round(prediction['confidence'], 3)
                            })
                    
                    current_time += stride_fine
            
            return {
                'success': True,
                'duration': round(duration, 2),
                'total_detections': len(results),
                'detections': results
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# Global demo instance
demo = SimpleProfanityDemo()

class DemoHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler for demo server"""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Thai Profanity Detection Demo</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5rem;
        }
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #764ba2;
            background: #f8f9ff;
        }
        .upload-btn {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        .results {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9ff;
            border-radius: 10px;
            display: none;
        }
        .detection-item {
            background: white;
            margin: 10px 0;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #f44336;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .timestamp {
            color: #2196f3;
            font-weight: bold;
        }
        .word {
            color: #f44336;
            font-size: 1.2em;
            font-weight: bold;
        }
        .confidence {
            color: #4caf50;
            font-size: 0.9em;
        }
        .status {
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .success {
            background: #e8f5e8;
            color: #2e7d32;
            border: 2px solid #4caf50;
        }
        .error {
            background: #ffebee;
            color: #c62828;
            border: 2px solid #f44336;
        }
        .loading {
            background: #e3f2fd;
            color: #1976d2;
            border: 2px solid #2196f3;
        }
        .summary {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Thai Profanity Detection Demo</h1>
        
        <div class="upload-area">
            <h3>📁 Upload Audio File</h3>
            <p>Support formats: WAV, MP3, FLAC, M4A</p>
            <input type="file" id="audioFile" accept="audio/*" style="display: none;">
            <button class="upload-btn" onclick="document.getElementById('audioFile').click();">
                Choose Audio File
            </button>
            <button class="upload-btn" onclick="analyzeAudio()" id="analyzeBtn" disabled>
                🚀 Analyze Audio
            </button>
        </div>
        
        <div id="status"></div>
        <div id="results" class="results"></div>
    </div>

    <script>
        document.getElementById('audioFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('analyzeBtn').disabled = false;
                showStatus('File selected: ' + file.name, 'success');
            }
        });

        function showStatus(message, type) {
            const status = document.getElementById('status');
            status.innerHTML = '<div class="status ' + type + '">' + message + '</div>';
        }

        function analyzeAudio() {
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            
            if (!file) {
                showStatus('Please select an audio file first', 'error');
                return;
            }

            showStatus('🔄 Analyzing audio... This may take a few moments', 'loading');
            document.getElementById('analyzeBtn').disabled = true;

            const formData = new FormData();
            formData.append('audio', file);

            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResults(data);
                document.getElementById('analyzeBtn').disabled = false;
            })
            .catch(error => {
                showStatus('Error: ' + error.message, 'error');
                document.getElementById('analyzeBtn').disabled = false;
            });
        }

        function displayResults(data) {
            if (!data.success) {
                showStatus('Analysis failed: ' + data.error, 'error');
                return;
            }

            const resultsDiv = document.getElementById('results');
            
            let html = '<div class="summary">';
            html += '<h3>📊 Analysis Summary</h3>';
            html += '<p>Duration: ' + data.duration + ' seconds</p>';
            html += '<p>Profanity instances found: ' + data.total_detections + '</p>';
            html += '</div>';

            if (data.total_detections > 0) {
                showStatus('⚠️ Profanity detected in audio', 'error');
                
                html += '<h3>🎯 Detailed Detection Results</h3>';
                
                data.detections.forEach((detection, index) => {
                    const startMin = Math.floor(detection.start_time / 60);
                    const startSec = (detection.start_time % 60).toFixed(2);
                    const endMin = Math.floor(detection.end_time / 60);
                    const endSec = (detection.end_time % 60).toFixed(2);
                    
                    html += '<div class="detection-item">';
                    html += '<div><strong>Instance ' + (index + 1) + '</strong></div>';
                    html += '<div class="word">Word: ' + detection.word + '</div>';
                    html += '<div class="timestamp">Time: ' + startMin + ':' + startSec.padStart(5, '0') + ' - ' + endMin + ':' + endSec.padStart(5, '0') + '</div>';
                    html += '<div class="confidence">Confidence: ' + (detection.confidence * 100).toFixed(1) + '%</div>';
                    html += '</div>';
                });
            } else {
                showStatus('✅ No profanity detected in audio', 'success');
            }

            resultsDiv.innerHTML = html;
            resultsDiv.style.display = 'block';
        }
    </script>
</body>
</html>
            """
            
            self.wfile.write(html.encode())
        else:
            super().do_GET()
    
    def do_POST(self):
        if self.path == '/analyze':
            try:
                # Parse multipart form data
                content_type = self.headers['content-type']
                if not content_type.startswith('multipart/form-data'):
                    self.send_error(400, "Expected multipart/form-data")
                    return
                
                # Get content length
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    # Extract file data (simplified - would need proper multipart parsing for production)
                    # For demo, we'll use a simple approach
                    boundary = content_type.split('boundary=')[1].encode()
                    parts = post_data.split(boundary)
                    
                    for part in parts:
                        if b'filename=' in part and b'Content-Type: audio' in part:
                            # Find the start of file data
                            data_start = part.find(b'\r\n\r\n') + 4
                            if data_start > 3:
                                file_data = part[data_start:]
                                # Remove trailing boundary markers
                                if file_data.endswith(b'\r\n'):
                                    file_data = file_data[:-2]
                                tmp_file.write(file_data)
                                break
                    
                    tmp_file_path = tmp_file.name
                
                # Analyze audio
                results = demo.analyze_audio(tmp_file_path)
                
                # Clean up
                os.unlink(tmp_file_path)
                
                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(results).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'success': False,
                    'error': str(e)
                }).encode())

def run_demo(port=8000):
    """Run the demo server"""
    print(f"🚀 Starting Thai Profanity Detection Demo Server")
    print(f"📡 Server running on http://localhost:{port}")
    print(f"🌐 Open your browser and go to: http://localhost:{port}")
    print(f"🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        with socketserver.TCPServer(("", port), DemoHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Demo server stopped")

if __name__ == "__main__":
    run_demo()