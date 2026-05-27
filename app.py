from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import tensorflow as tf
import base64
from datetime import datetime
import os
import json
import threading
import time
from collections import deque

app = Flask(__name__)
app.config['SECRET_KEY'] = 'waste_classifier_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

IMG_SIZE = 128
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
CLASS_LABELS = {
    'cardboard': 'CARTÓN',
    'glass': 'VIDRIO',
    'metal': 'METAL',
    'paper': 'PAPEL',
    'plastic': 'PLÁSTICO'
}
CLASS_COLORS = {
    'cardboard': '#FF9800',
    'glass': '#4CAF50',
    'metal': '#9E9E9E',
    'paper': '#FFC107',
    'plastic': '#2196F3'
}
CLASS_ICONS = {
    'cardboard': '📦',
    'glass': '🥃',
    'metal': '🥫',
    'paper': '📄',
    'plastic': '🧴'
}
CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_WINDOW = 5

class WasteClassifier:
    def __init__(self, model_paths):
        self.models = {}
        self.active = None
        self.active_name = None
        self._load_models(model_paths)
    
    def _load_models(self, model_paths):
        for name, path in model_paths.items():
            if os.path.exists(path):
                try:
                    self.models[name] = tf.keras.models.load_model(path)
                    print(f"✓ Modelo cargado: {name}")
                except Exception as e:
                    print(f"✗ Error cargando {name}: {e}")
        
        if 'transfer' in self.models:
            self.active = self.models['transfer']
            self.active_name = 'transfer'
        elif 'scratch' in self.models:
            self.active = self.models['scratch']
            self.active_name = 'scratch'
    
    def switch_model(self, name):
        if name in self.models:
            self.active = self.models[name]
            self.active_name = name
            return True
        return False
    
    def predict(self, frame):
        if self.active is None:
            return None, 0.0, None
        
        if len(frame.shape) == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame
        
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        probs = self.active.predict(np.expand_dims(normalized, 0), verbose=0)[0]
        idx = np.argmax(probs)
        
        return CLASSES[idx], float(probs[idx]), probs.tolist()

model_paths = {
    'scratch': 'models/scratch_best.keras',
    'transfer': 'models/transfer_best.keras'
}
classifier = WasteClassifier(model_paths)

class StatsTracker:
    def __init__(self):
        self.counts = {c: 0 for c in CLASSES}
        self.last_prediction = {}
        self.session_start = datetime.now()
        self.predictions_queue = deque(maxlen=SMOOTHING_WINDOW)
    
    def add_prediction(self, class_name, confidence):
        self.predictions_queue.append((class_name, confidence, time.time()))
        
        if len(self.predictions_queue) >= SMOOTHING_WINDOW:
            recent_classes = [p[0] for p in self.predictions_queue]
            most_common = max(set(recent_classes), key=recent_classes.count)
            
            if most_common and confidence >= CONFIDENCE_THRESHOLD:
                last_class, last_time = self.last_prediction.get(most_common, (None, 0))
                current_time = time.time()
                if current_time - last_time > 1.0:
                    self.counts[most_common] += 1
                    self.last_prediction[most_common] = (most_common, current_time)
        
        return self.get_stats()
    
    def get_stats(self):
        total = sum(self.counts.values())
        percentages = {c: (self.counts[c] / total * 100) if total > 0 else 0 
                      for c in CLASSES}
        return {
            'counts': self.counts,
            'percentages': percentages,
            'total': total,
            'session_duration': int((datetime.now() - self.session_start).total_seconds())
        }
    
    def reset(self):
        self.counts = {c: 0 for c in CLASSES}
        self.last_prediction = {}
        self.predictions_queue.clear()
        self.session_start = datetime.now()
        return self.get_stats()

stats_tracker = StatsTracker()

@app.route('/')
def index():
    return render_template('index.html', 
                         classes=CLASSES,
                         class_labels=CLASS_LABELS,
                         class_colors=CLASS_COLORS,
                         class_icons=CLASS_ICONS)

@app.route('/api/stats')
def get_stats():
    return jsonify(stats_tracker.get_stats())

@app.route('/api/reset_stats', methods=['POST'])
def reset_stats():
    stats = stats_tracker.reset()
    return jsonify(stats)

@app.route('/api/switch_model/<model_name>')
def switch_model(model_name):
    success = classifier.switch_model(model_name)
    return jsonify({
        'success': success,
        'active_model': classifier.active_name
    })

@app.route('/api/status')
def get_status():
    return jsonify({
        'model_loaded': classifier.active is not None,
        'active_model': classifier.active_name,
        'available_models': list(classifier.models.keys())
    })

@socketio.on('frame')
def handle_frame(data):
    """Recibe frame en base64, realiza predicción y envía resultados"""
    try:
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return
        
        predicted_class, confidence, all_probs = classifier.predict(frame)
        
        if predicted_class:
            stats = stats_tracker.add_prediction(predicted_class, confidence)
            
            emit('prediction', {
                'class': predicted_class,
                'label': CLASS_LABELS[predicted_class],
                'icon': CLASS_ICONS[predicted_class],
                'confidence': confidence,
                'all_probs': [
                    {'class': c, 'label': CLASS_LABELS[c], 
                     'probability': all_probs[i] if all_probs else 0,
                     'color': CLASS_COLORS[c]}
                    for i, c in enumerate(CLASSES)
                ],
                'stats': stats
            })
    
    except Exception as e:
        print(f"Error en frame: {e}")

@socketio.on('reset_stats')
def handle_reset_stats():
    stats = stats_tracker.reset()
    emit('stats_update', stats)

@socketio.on('switch_model')
def handle_switch_model(data):
    success = classifier.switch_model(data['model'])
    emit('model_switched', {
        'success': success,
        'active_model': classifier.active_name
    })

@socketio.on('connect')
def handle_connect():
    print(f"Cliente conectado")
    emit('connected', {
        'model_loaded': classifier.active is not None,
        'active_model': classifier.active_name,
        'available_models': list(classifier.models.keys()),
        'stats': stats_tracker.get_stats(),
        'classes': [
            {'id': c, 'label': CLASS_LABELS[c], 'color': CLASS_COLORS[c], 'icon': CLASS_ICONS[c]}
            for c in CLASSES
        ]
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("   SMART WASTE CLASSIFIER - WEB APPLICATION")
    print("="*60)
    print(f"\n  Modelos cargados: {list(classifier.models.keys())}")
    print(f"  Modelo activo: {classifier.active_name}")
    print("\n  Abre tu navegador en: http://localhost:5000")
    print("  Asegúrate de permitir acceso a la cámara")
    print("\n" + "="*60 + "\n")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)