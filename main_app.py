import sys
import torch
import torchvision
from torchvision import transforms
from PyQt6 import QtCore, QtGui, QtWidgets
import os
import cv2
import numpy as np
import time
import math
import random
from collections import deque, Counter

# --- إعدادات النظام ---
THRESHOLD = 0.2 
EMBEDDINGS_DIR = "embeddings"
IMG_SIZE = (380, 380)
SMOOTHING_FRAMES = 5

# ألوان التصميم (Futuristic / Neural Style)
COLOR_NEON_GREEN = (57, 255, 20)
COLOR_NEON_BLUE = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_GOLD = (255, 215, 0)

# اختيار الجهاز
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- فئات المؤثرات البصرية (VFX) ---

class Particle:
    def __init__(self, w, h):
        self.reset(w, h)
        
    def reset(self, w, h):
        self.x = random.randint(0, w)
        self.y = random.randint(0, h)
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.life = random.randint(20, 50)
        self.color = COLOR_NEON_BLUE if random.random() > 0.5 else COLOR_NEON_GREEN

    def update(self, target_x=None, target_y=None):
        if target_x is not None:
            # انجذاب نحو الهدف (الوجه)
            dx = target_x - self.x
            dy = target_y - self.y
            dist = math.sqrt(dx*dx + dy*dy) + 1
            self.vx += dx / dist * 0.5
            self.vy += dy / dist * 0.5
            
        self.x += self.vx
        self.y += self.vy
        self.life -= 1
        return self.life > 0

class Ripple:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 0
        self.max_radius = 100
        self.opacity = 1.0

    def update(self):
        self.radius += 5
        self.opacity -= 0.05
        return self.opacity > 0

# --- الأجهزة والمنطق الأساسي ---

def load_model():
    model = torchvision.models.efficientnet_v2_m()
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(num_ftrs, 4)
    if os.path.exists("best_model.pth"):
        state_dict = torch.load("best_model.pth", map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

full_model = load_model()

def get_embedding(model, x):
    with torch.no_grad():
        features = model.features(x)
        features = model.avgpool(features)
        features = torch.flatten(features, 1)
        return features[0].cpu().numpy()

known_embeddings = {}
if os.path.exists(EMBEDDINGS_DIR):
    for filename in os.listdir(EMBEDDINGS_DIR):
        if filename.endswith(".npy"):
            name = os.path.splitext(filename)[0]
            emb = np.load(os.path.join(EMBEDDINGS_DIR, filename))
            known_embeddings[name] = emb

def calculate_distance(emb1, emb2):
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0: return 1.0
    return 1 - np.dot(emb1, emb2) / (norm1 * norm2)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_face(face_img):
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_tensor = transform(face_img)
    return face_tensor.unsqueeze(0).to(device)

# --- أدوات الرسم الاحترافية (Neural Nexus HUD) ---

def draw_nexus_hud(frame, x, y, w, h, color, label, confidence, scan_pos, particles, ripples):
    # 1. الشبكة الحيوية (Biometric Grid)
    grid_overlay = frame.copy()
    step = 20
    for i in range(x, x + w, step):
        cv2.line(grid_overlay, (i, y), (i, y + h), color, 1)
    for j in range(y, y + h, step):
        cv2.line(grid_overlay, (x, j), (x + w, j), color, 1)
    cv2.addWeighted(grid_overlay, 0.1, frame, 0.9, 0, frame)

    # 2. المربعات الركنية الفاخرة
    length = 40
    t = 4
    cv2.line(frame, (x-5, y-5), (x + length, y-5), color, t)
    cv2.line(frame, (x-5, y-5), (x-5, y + length), color, t)
    cv2.line(frame, (x + w + 5, y-5), (x + w - length + 5, y-5), color, t)
    cv2.line(frame, (x + w + 5, y-5), (x + w + 5, y + length), color, t)
    cv2.line(frame, (x-5, y + h + 5), (x + length, y + h + 5), color, t)
    cv2.line(frame, (x-5, y + h + 5), (x-5, y + h - length + 5), color, t)
    cv2.line(frame, (x + w + 5, y + h + 5), (x + w - length + 5, y + h + 5), color, t)
    cv2.line(frame, (x + w + 5, y + h + 5), (x + w + 5, y + h - length + 5), color, t)

    # 3. الليزر والنبضات
    scan_y = y + int(scan_pos * h)
    cv2.line(frame, (x, scan_y), (x + w, scan_y), color, 2)
    
    for ripple in ripples:
        alpha = ripple.opacity
        cv2.circle(frame, (ripple.x, ripple.y), ripple.radius, color, 2)

    # 4. تتبع الجسيمات (Particles)
    for p in particles:
        cv2.circle(frame, (int(p.x), int(p.y)), 1, p.color, -1)

    # 5. البطاقة التعريفية بالعربي (Arabic ID Card)
    if label != "جاري التعرف..." and label != "غير معروف":
        card_x = x + w + 20
        card_w = 220
        # خلفية متدرجة للبطاقة
        for i in range(110):
            alpha = (110 - i) / 110 * 0.6
            cv2.line(frame, (card_x, y + i), (card_x + card_w, y + i), (10, 15, 20), 1)
        
        cv2.rectangle(frame, (card_x, y), (card_x + card_w, y + 110), color, 1)
        
        # نصوص البطاقة
        status_text = "مصرح بالدخول" if label != "غير معروف" else "دخول مرفوض"
        cv2.putText(frame, "BIOMETRIC PROFILE", (card_x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(frame, f"NAME: {label.upper()}", (card_x + 10, y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
        cv2.putText(frame, f"STATUS: {status_text}", (card_x + 10, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # شريط المزامنة العصبية
        sync_val = int((1 - confidence) * 100)
        cv2.putText(frame, f"SYNC: {sync_val}%", (card_x + 10, y + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_WHITE, 1)
        cv2.rectangle(frame, (card_x + 70, y + 90), (card_x + 170, y + 98), (30, 30, 30), -1)
        cv2.rectangle(frame, (card_x + 70, y + 90), (card_x + 70 + sync_val, y + 98), color, -1)

# --- ثريد الفيديو - النسخة الأسطورية 4.0 ---

class VideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray, float, str, list) # Frame, FPS, State, Logs

    def __init__(self):
        super().__init__()
        self.running = True
        self.history = deque(maxlen=SMOOTHING_FRAMES)
        self.prev_time = time.time()
        self.scan_ticker = 0
        self.particles = [Particle(640, 480) for _ in range(50)]
        self.ripples = []
        self.logs = deque(maxlen=15)
        self.add_log("System Initialized. Awaiting biometric input...")

    def add_log(self, msg):
        timestamp = time.strftime("[%H:%M:%S]")
        self.logs.append(f"{timestamp} {msg}")

    def run(self):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        while self.running:
            ret, frame = cap.read()
            if not ret: continue
            
            h_frame, w_frame = frame.shape[:2]
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time)
            self.prev_time = curr_time

            self.scan_ticker += 0.04
            scan_pos = (math.sin(self.scan_ticker) + 1) / 2

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(160, 160))

            current_state = "جاري المسح الحيوى..."
            
            # تحديث الجسيمات
            target_x, target_y = None, None
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                target_x, target_y = fx + fw//2, fy + fh//2
                if len(self.ripples) == 0:
                    self.ripples.append(Ripple(target_x, target_y))
                    self.add_log(f"Signature Detected at Region ({fx},{fy})")

            # تحديث VFX
            new_particles = []
            for p in self.particles:
                if p.update(target_x, target_y):
                    new_particles.append(p)
                else:
                    new_particles.append(Particle(w_frame, h_frame))
            self.particles = new_particles

            self.ripples = [r for r in self.ripples if r.update()]

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                name = "جاري التعرف..."
                color = COLOR_NEON_BLUE
                min_dist = 1.0
                
                if known_embeddings:
                    try:
                        input_face = preprocess_face(face_img)
                        curr_emb = get_embedding(full_model, input_face)
                        
                        best_match = "غير معروف"
                        for user_name, user_emb in known_embeddings.items():
                            dist = calculate_distance(curr_emb, user_emb)
                            if dist < min_dist:
                                min_dist = dist
                                if dist <= THRESHOLD:
                                    best_match = user_name

                        self.history.append(best_match)
                        counts = Counter(self.history)
                        name = counts.most_common(1)[0][0]
                        
                        if name != "غير معروف":
                            color = COLOR_NEON_GREEN
                            current_state = "تم التحقق من الهوية"
                            if len(self.history) == 1 or self.history[-2] != name:
                                self.add_log(f"Access Granted: User {name.upper()}")
                        else:
                            color = COLOR_RED
                            current_state = "هوية غير مصرح بها"
                            if len(self.history) == 1 or self.history[-2] != "غير معروف":
                                self.add_log("WARNING: Unauthorized Identity Detected")

                    except Exception as e:
                        name = "خطأ في النظام"
                        self.add_log(f"CRITICAL ERROR: {str(e)}")

                draw_nexus_hud(frame, x, y, w, h, color, name, min_dist, scan_pos, self.particles, self.ripples)

            self.frame_ready.emit(frame, fps, current_state, list(self.logs))

        cap.release()

    def stop(self):
        self.running = False

# --- الواجهة النهائية (Neural Nexus Dashboard) ---

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEURAL NEXUS - مركز القيادة الحيوية")
        self.resize(1280, 720)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #020406;
            }
            QLabel {
                color: #00FFFF;
                font-family: 'Segoe UI', 'Arial';
            }
            #SidePanel {
                background-color: rgba(5, 10, 15, 0.95);
                border-left: 2px solid #00FFFF;
                padding: 10px;
            }
            #Terminal {
                background-color: #000000;
                color: #00FF00;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                border: 1px solid #113311;
                padding: 10px;
            }
            #Header {
                border-bottom: 2px solid #00FFFF;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #050A0F, stop:1 #003333);
            }
        """)

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # 1. الهيدر
        header = QtWidgets.QWidget()
        header.setObjectName("Header")
        header.setFixedHeight(60)
        header_layout = QtWidgets.QHBoxLayout(header)
        
        title = QtWidgets.QLabel("THE NEURAL NEXUS CORE v4.0")
        title.setStyleSheet("font-size: 24px; font-weight: bold; letter-spacing: 3px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        self.time_label = QtWidgets.QLabel()
        self.time_label.setStyleSheet("font-size: 18px; color: #FFFFFF;")
        header_layout.addWidget(self.time_label)
        
        main_layout.addWidget(header)

        # 2. منطقة العمل الرئيسية
        body = QtWidgets.QWidget()
        body_layout = QtWidgets.QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # الكاميرا
        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        body_layout.addWidget(self.video_label, 7)

        # اللوحة الجانبية (Terminal + Status)
        side_panel = QtWidgets.QWidget()
        side_panel.setObjectName("SidePanel")
        side_panel.setFixedWidth(350)
        side_layout = QtWidgets.QVBoxLayout(side_panel)

        side_layout.addWidget(QtWidgets.QLabel("قوة المعالجة وحالة النظام"))
        self.fps_label = QtWidgets.QLabel("FPS: 00.0")
        self.fps_label.setStyleSheet("font-size: 14px; color: #57FF14;")
        side_layout.addWidget(self.fps_label)

        self.state_label = QtWidgets.QLabel("الحالة: جاري التشغيل")
        self.state_label.setStyleSheet("font-size: 14px; color: #FFD700;")
        side_layout.addWidget(self.state_label)

        side_layout.addSpacing(20)
        side_layout.addWidget(QtWidgets.QLabel("سجل أحداث النظام (LIVE TERMINAL)"))
        self.terminal = QtWidgets.QPlainTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setObjectName("Terminal")
        side_layout.addWidget(self.terminal)

        side_layout.addSpacing(20)
        side_layout.addWidget(QtWidgets.QLabel("قاعدة البيانات المشفرة"))
        for user in known_embeddings.keys():
            lbl = QtWidgets.QLabel(f"IDENT_SIG::{user.upper()}")
            lbl.setStyleSheet("font-size: 10px; color: #446666;")
            side_layout.addWidget(lbl)

        side_layout.addStretch()
        ver = QtWidgets.QLabel("SECURITY LEVEL: ALPHA-1")
        ver.setStyleSheet("font-size: 10px; color: #FF0000; text-align: center;")
        side_layout.addWidget(ver)

        body_layout.addWidget(side_panel)
        main_layout.addWidget(body)

        # تحديث الوقت
        self.time_timer = QtCore.QTimer()
        self.time_timer.timeout.connect(lambda: self.time_label.setText(time.strftime("%Y-%m-%d | %H:%M:%S")))
        self.time_timer.start(1000)

        # تشغيل المحرك
        self.thread = VideoThread()
        self.thread.frame_ready.connect(self.update_ui)
        self.thread.start()

    def update_ui(self, frame, fps, state, logs):
        self.fps_label.setText(f"سرعة المحرك: {fps:.1f} فريم/ثانية")
        self.state_label.setText(f"الحالة الحالية: {state}")
        
        # تحديث الترمينال
        self.terminal.setPlainText("\n".join(logs))
        self.terminal.verticalScrollBar().setValue(self.terminal.verticalScrollBar().maximum())

        # عرض الفيديو
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(img).scaled(self.video_label.size(), 
                                                     QtCore.Qt.AspectRatioMode.KeepAspectRatio, 
                                                     QtCore.Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, e):
        self.thread.stop()
        self.thread.wait()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())