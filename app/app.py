"""
Controles:
  Q / ESC  → salir
  S        → mostrar/ocultar estadísticas
  R        → resetear contadores
  1        → modelo CNN Scratch
  2        → modelo Transfer Learning
"""

import cv2
import numpy as np
import tensorflow as tf
import time
import os
import collections
from datetime import datetime

IMG_SIZE   = 128
CLASSES    = ['cardboard', 'glass', 'metal', 'paper', 'plastic']
CLASS_LABELS = {
    'cardboard': 'CARTON',
    'glass':     'VIDRIO',
    'metal':     'METAL',
    'paper':     'PAPEL',
    'plastic':   'PLASTICO',
}
CLASS_ICONS = {
    'cardboard': '[BOX]',
    'glass':     '[JR] ',
    'metal':     '[CAN]',
    'paper':     '[PG] ',
    'plastic':   '[BTL]',
}
CLASS_COLORS = {
    'cardboard': (0,   165, 255),   # naranja
    'glass':     (0,   210, 80 ),   # verde
    'metal':     (200, 200, 200),   # gris claro
    'paper':     (230, 230, 0  ),   # amarillo
    'plastic':   (80,  130, 255),   # azul
}
CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_WINDOW     = 8
ROI_SCALE            = 0.45

UI = {
    'bg':    (15,  20,  30 ),
    'panel': (25,  30,  45 ),
    'accent':(0,   200, 150),
    'white': (240, 240, 240),
    'gray':  (120, 130, 145),
    'warn':  (40,  200, 250),
}


def rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    x1, y1 = pt1; x2, y2 = pt2
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(img, (cx, cy), radius, color, thickness)


def draw_bar(img, x, y, w, h, value, color, bg=(40, 45, 60)):
    cv2.rectangle(img, (x, y), (x+w, y+h), bg, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (60, 65, 80), 1)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)


def put_text(img, text, pos, scale=0.55, color=(240,240,240), thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


class WasteClassifier:
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.models      = {}
        self.active      = None
        self.active_name = ''
        self._load_available()

    def _load_available(self):
        for name, path in self.model_paths.items():
            if os.path.exists(path):
                try:
                    self.models[name] = tf.keras.models.load_model(path)
                    print(f"✓ Modelo cargado: {name}")
                except Exception as e:
                    print(f"✗ Error cargando {name}: {e}")
        if self.models:
            for p in ['transfer', 'scratch']:
                if p in self.models:
                    self.active = self.models[p]
                    self.active_name = p
                    break
        else:
            print("⚠  No se encontraron modelos. Ejecuta train.py primero.")

    def switch_model(self, name):
        if name in self.models:
            self.active = self.models[name]
            self.active_name = name
            return True
        print(f"⚠  Modelo '{name}' no disponible.")
        return False

    def predict(self, frame_roi):
        if self.active is None:
            return None, 0.0, np.zeros(len(CLASSES))
        rgb        = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2RGB)
        resized    = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        probs      = self.active.predict(np.expand_dims(normalized, 0), verbose=0)[0]
        idx        = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx]), probs


class StatsTracker:
    def __init__(self):
        self.counts        = {c: 0 for c in CLASSES}
        self.fps_times     = collections.deque(maxlen=30)
        self.session_start = datetime.now()
        self._last_record  = 0

    def record(self, class_name, confidence):
        now = time.time()
        if now - self._last_record > 1.0:
            self.counts[class_name] += 1
            self._last_record = now

    def fps(self):
        if len(self.fps_times) < 2:
            return 0.0
        diffs = [self.fps_times[i+1]-self.fps_times[i] for i in range(len(self.fps_times)-1)]
        return 1.0 / (np.mean(diffs) + 1e-9)

    def total(self):
        return sum(self.counts.values())

    def reset(self):
        self.counts        = {c: 0 for c in CLASSES}
        self.session_start = datetime.now()


def run_app():
    classifier = WasteClassifier({
        'scratch':  'models/scratch_best.keras',
        'transfer': 'models/transfer_best.keras',
    })
    stats      = StatsTracker()
    smooth_buf = collections.deque(maxlen=SMOOTHING_WINDOW)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: No se pudo abrir la cámara.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    show_stats  = True
    last_class  = None
    last_conf   = 0.0
    last_probs  = np.zeros(len(CLASSES))
    frame_count = 0

    print("\n=== Smart Waste Classifier ===")
    print("Controles: Q/ESC=salir | S=stats | R=reset | 1=scratch | 2=transfer\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        now = time.time()
        stats.fps_times.append(now)

        h, w    = frame.shape[:2]
        roi_sz  = int(min(h, w) * ROI_SCALE)
        cx, cy  = w//2, h//2
        x1, y1  = cx - roi_sz//2, cy - roi_sz//2
        x2, y2  = cx + roi_sz//2, cy + roi_sz//2

        # Classify every 3rd frame
        if frame_count % 3 == 0 and classifier.active:
            roi = frame[y1:y2, x1:x2]
            cls, conf, probs = classifier.predict(roi)
            smooth_buf.append((cls, conf, probs))

            if smooth_buf:
                cls_buf      = [s[0] for s in smooth_buf]
                smoothed_cls = max(set(cls_buf), key=cls_buf.count)
                avg_conf     = float(np.mean([s[1] for s in smooth_buf if s[0] == smoothed_cls]))
                avg_probs    = np.mean([s[2] for s in smooth_buf], axis=0)
                last_class   = smoothed_cls
                last_conf    = avg_conf
                last_probs   = avg_probs
                if avg_conf >= CONFIDENCE_THRESHOLD:
                    stats.record(smoothed_cls, avg_conf)

        canvas = frame.copy()

        dark = cv2.addWeighted(canvas, 0.35, np.zeros_like(canvas), 0.65, 0)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        canvas[mask == 0] = dark[mask == 0]

        border_color = CLASS_COLORS.get(last_class, UI['gray']) \
                       if last_class and last_conf >= CONFIDENCE_THRESHOLD else UI['gray']
        cv2.rectangle(canvas, (x1-1, y1-1), (x2+1, y2+1), (0,0,0), 3)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 2)

        L = 28
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(canvas, (px, py), (px+dx*L, py), border_color, 3)
            cv2.line(canvas, (px, py), (px, py+dy*L), border_color, 3)

        scan_y = y1 + int((now * 80) % roi_sz)
        cv2.line(canvas, (x1, scan_y), (x2, scan_y), UI['accent'], 1)

        cv2.rectangle(canvas, (0, 0), (w, 50), (10, 15, 25), -1)
        put_text(canvas, "SMART WASTE CLASSIFIER", (12, 33), 0.75, UI['accent'], 2)
        put_text(canvas, f"FPS: {stats.fps():.1f}", (w-130, 33), 0.55, UI['gray'])
        model_lbl = f"Modelo: {classifier.active_name.upper()}" if classifier.active else "SIN MODELO"
        put_text(canvas, model_lbl, (w//2-90, 33), 0.55, UI['warn'])

        px, py, pw = 12, 60, 268
        ph = 10 + 30 + 20 + 12 + 10 + len(CLASSES)*19 + 10
        overlay = canvas.copy()
        rounded_rect(overlay, (px, py), (px+pw, py+ph), (20, 25, 38), radius=10)
        cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
        cv2.rectangle(canvas, (px, py), (px+pw, py+ph), (50,55,75), 1)

        put_text(canvas, "DETECCION", (px+10, py+22), 0.45, UI['gray'])

        if last_class and last_conf >= CONFIDENCE_THRESHOLD:
            cc  = CLASS_COLORS[last_class]
            lbl = f"{CLASS_ICONS[last_class]} {CLASS_LABELS[last_class]}"
            put_text(canvas, lbl, (px+10, py+52), 0.85, cc, 2)
            put_text(canvas, f"Confianza: {last_conf:.1%}", (px+10, py+76), 0.52, UI['white'])
            draw_bar(canvas, px+10, py+84, pw-20, 8, last_conf, cc)
        else:
            put_text(canvas, "Coloca el objeto en el recuadro", (px+10, py+52),
                     0.47, UI['gray'])

        put_text(canvas, "Probabilidades:", (px+10, py+104), 0.42, UI['gray'])
        for i, cls in enumerate(CLASSES):
            by   = py + 116 + i*19
            pval = float(last_probs[i]) if len(last_probs) == len(CLASSES) else 0.0
            cc   = CLASS_COLORS[cls]
            put_text(canvas, CLASS_ICONS[cls], (px+8, by+12), 0.38, UI['white'])
            draw_bar(canvas, px+58, by, pw-82, 11, pval, cc)
            put_text(canvas, f"{pval:.0%}", (px+pw-30, by+11), 0.38, UI['gray'])

        if show_stats:
            sp_x = w - 280
            sp_y = 60
            sp_w = 268
            sp_h = 60 + len(CLASSES)*40 + 30

            overlay2 = canvas.copy()
            rounded_rect(overlay2, (sp_x, sp_y), (sp_x+sp_w, sp_y+sp_h), (20, 25, 38), radius=10)
            cv2.addWeighted(overlay2, 0.88, canvas, 0.12, 0, canvas)
            cv2.rectangle(canvas, (sp_x, sp_y), (sp_x+sp_w, sp_y+sp_h), (50,55,75), 1)

            put_text(canvas, "ESTADISTICAS", (sp_x+10, sp_y+22), 0.45, UI['gray'])
            total = stats.total()
            put_text(canvas, f"Total detectado: {total}", (sp_x+10, sp_y+48),
                     0.58, UI['accent'], 2)

            for i, cls in enumerate(CLASSES):
                by    = sp_y + 62 + i*40
                count = stats.counts[cls]
                ratio = count / max(total, 1)
                cc    = CLASS_COLORS[cls]
                put_text(canvas, f"{CLASS_ICONS[cls]} {CLASS_LABELS[cls]}",
                         (sp_x+10, by+14), 0.46, UI['white'])
                draw_bar(canvas, sp_x+10, by+18, sp_w-60, 13, ratio, cc)
                put_text(canvas, str(count), (sp_x+sp_w-40, by+30), 0.6, cc, 2)

            elapsed = datetime.now() - stats.session_start
            m, s = divmod(int(elapsed.total_seconds()), 60)
            put_text(canvas, f"Sesion: {m:02d}:{s:02d}", (sp_x+10, sp_y+sp_h-12),
                     0.45, UI['gray'])

        cv2.rectangle(canvas, (0, h-36), (w, h), (10, 15, 25), -1)
        put_text(canvas, "Q:Salir  S:Stats  R:Reset  1:Scratch  2:Transfer",
                 (w//2-200, h-12), 0.45, UI['gray'])

        cv2.imshow('Smart Waste Classifier', canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            show_stats = not show_stats
        elif key == ord('r'):
            stats.reset(); smooth_buf.clear()
            last_class = None; last_conf = 0.0
            last_probs = np.zeros(len(CLASSES))
            print("Estadísticas reseteadas.")
        elif key == ord('1'):
            if classifier.switch_model('scratch'):
                print("Modelo: CNN Scratch")
        elif key == ord('2'):
            if classifier.switch_model('transfer'):
                print("Modelo: MobileNetV2 Transfer")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*40)
    print("   RESUMEN DE SESION")
    print("="*40)
    for cls in CLASSES:
        bar = "█" * stats.counts[cls]
        print(f"  {CLASS_LABELS[cls]:<10}: {stats.counts[cls]:>4}  {bar}")
    print(f"\n  Total : {stats.total()}")
    print("="*40)


if __name__ == '__main__':
    run_app()