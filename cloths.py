import sys
import cv2
import mediapipe as mp
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel,
    QTabWidget, QPushButton, QHBoxLayout ,QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

# --- Mediapipe setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Known constants
KNOWN_FACE_WIDTH_CM = 16
KNOWN_SHOULDER_WIDTH_CM = 40
focal_length = None


def euclidean_dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def classify_size_by_ratio(ratio):
    if ratio < 0.21:
        return "S"
    elif ratio < 0.23:
        return "M"
    elif ratio < 0.25:
        return "L"
    elif ratio < 0.27:
        return "XL"
    else:
        return "XXL"


class MyAI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.label1 = QLabel("We Will show Your Measurement, Don't worry", self)
        self.camera = QLabel(self)
        self.button1 = QPushButton("Start")
        self.button2 = QPushButton("Stop")

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.initui()

    def initui(self):
        self.setWindowTitle("Cloth Size Measurement App")
        self.resize(900, 700)

        # Tabs
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Home tab
        self.home_tab = QWidget()
        vbox = QVBoxLayout()
        hbox = QHBoxLayout()

        vbox.addWidget(self.label1)
        vbox.addWidget(self.camera)

        hbox.addWidget(self.button1)
        hbox.addWidget(self.button2)
        vbox.addLayout(hbox)

        self.home_tab.setLayout(vbox)
        
        
        self.label1.setObjectName("label1")
        self.camera.setObjectName("camera")
        self.button1.setObjectName("startBtn")
        self.button2.setObjectName("stopBtn")

        self.camera.setStyleSheet("border: 2px dashed gray;")
        
        self.camera.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        
        self.camera.setAlignment(Qt.AlignCenter)

        # Button actions
        self.button1.clicked.connect(self.start_camera)
        self.button2.clicked.connect(self.stop_camera)

        # About tab
        self.about_tab = QWidget()
        about_layout = QVBoxLayout()
        about_layout.addWidget(QLabel("""✨ Cloth Size Measurement AI – Your Virtual Fitting Room ✨

Imagine walking into a store where you don’t need a trial room, no measuring tape,

and no guesswork about whether a shirt will fit you. 

That’s exactly what this AI-powered Cloth Size Measurement App delivers!

Built with the power of Mediapipe’s Pose Estimation, OpenCV, and a smooth PyQt5 interface,

this app transforms your regular webcam into a smart virtual tailor. 👔👗

🔹 How it works:

The app detects your body landmarks in real time.

It auto-calibrates using your face to calculate accurate distances.

Using your shoulder width and height proportions, it estimates your perfect clothing size (S, M, L, XL, XXL).

A live status bar guides you to stand at the ideal distance for best results.

🔹 Why it’s exciting:

🚀 No need for manual measuring tapes.

🎥 Real-time camera preview with AI-powered body landmark tracking.

🧠 Smart calibration – adapts to your body automatically.

🖥️ Clean, modern interface with start/stop controls and multiple tabs.

👕 Accurate virtual fitting assistance for online or offline shopping.

In short, your app is like a virtual stylist + personal tailor that ensures you always know your perfect fit before you buy.

Perfect for e-commerce clothing platforms, fitness tracking, or even personal wardrobe planning."""))
        self.about_tab.setLayout(about_layout)

        # Add tabs
        self.tabs.addTab(self.home_tab, "HOME")
        self.tabs.addTab(self.about_tab, "About")
        
        self.home_tab.setObjectName("homeTab")
        self.about_tab.setObjectName("aboutTab")

        # Styles
        self.setStyleSheet("""
            #camera {
                background-color: hsl(157, 25%, 56%);
            }
            QWidget, QMainWindow {
                background-color: hsl(164, 86%, 79%);   /* or use hex like #b7e87a */
            }
            #label1 {
                font-size: 32px;
                font-style: italic;
                padding-top:50px;
                padding-bottom:50px;
            }
            QTabBar::tab {
                background: #b2ebf2;
                border: 2px solid #009688;
                padding: 10px 20px;
                margin: 2px;
                border-radius: 8px;
            }

            QTabBar::tab:selected {
                background: #009688;
                color: white;
                font-weight:bold;
            }

            QTabBar::tab:hover {
                background: #80deea;

            }

            
            #startBtn,#stopBtn{
                font-size:20px;
            }
            #startBtn:hover {
            background-color: green;
            color: white;   
            }
            #stopBtn:hover {
            background-color: red;
            color: white;  
            }

            QLabel{
                font-size:16px;  
                font-weight:bold;
            }
            Qlabel:hover{
                
            }
        """)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.camera.clear()

    def update_frame(self):
        global focal_length

        ret, frame = self.cap.read()
        if not ret:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        status_text = "Detecting..."
        bar_color = (0, 0, 255)
        rect_color = (0, 0, 255)
        size_text = ""
        distance_cm = None

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark

            # Face width calibration
            l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            
            # recognize hieght on basis of X-axis and Y-axis w is width and h is hieght.....
            l_ear_x, l_ear_y = int(l_ear.x * w), int(l_ear.y * h)   
            r_ear_x, r_ear_y = int(r_ear.x * w), int(r_ear.y * h)
             
            #by using euclidean_dist you can calculate distance in 2-d space distance=(x2−x1)^2+(y2−y1)^2
      
            face_width_px = euclidean_dist(l_ear_x, l_ear_y, r_ear_x, r_ear_y)
            
            #   focal_length=((face_width_in_pixels)×(known_distance))/real_face_width


            if face_width_px > 0 and focal_length is None:
                focal_length = (face_width_px * 50) / KNOWN_FACE_WIDTH_CM
                status_text = "Auto-calibrated ✔"

            # Shoulder width detection
            l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

            l_sh_x, l_sh_y = int(l_shoulder.x * w), int(l_shoulder.y * h)
            r_sh_x, r_sh_y = int(r_shoulder.x * w), int(r_shoulder.y * h)

            shoulder_width_px = euclidean_dist(l_sh_x, l_sh_y, r_sh_x, r_sh_y)

            if shoulder_width_px > 0 and focal_length:
                distance_cm = (KNOWN_SHOULDER_WIDTH_CM * focal_length) / shoulder_width_px

                # Bounding box
                x_coords = [lm.x * w for lm in landmarks]
                y_coords = [lm.y * h for lm in landmarks]
                min_x, max_x = int(min(x_coords)), int(max(x_coords))
                min_y, max_y = int(min(y_coords)), int(max(y_coords))

                if 63 <= distance_cm <= 120:
                    status_text = f"Perfect Distance: {distance_cm/100:.2f} m"
                    bar_color = (0, 255, 0)
                    rect_color = (0, 255, 0)

                    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
                    l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
                    r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

                    nose_y = int(nose.y * h)
                    l_ankle_y = int(l_ankle.y * h)
                    r_ankle_y = int(r_ankle.y * h)
                    person_height_px = max(l_ankle_y, r_ankle_y) - nose_y

                    if person_height_px > 0:
                        ratio = shoulder_width_px / person_height_px
                        size_text = f"Estimated Size: {classify_size_by_ratio(ratio)}"
                elif distance_cm < 63:
                    status_text = f"Move Back: {distance_cm/100:.2f} m"
                else:
                    status_text = f"Move Closer: {distance_cm/100:.2f} m"

                cv2.rectangle(frame, (min_x-20, min_y-20), (max_x+20, max_y+20), rect_color, 3)

            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Draw texts
        cv2.putText(frame, status_text, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, bar_color, 3)

        if size_text:
            cv2.putText(frame, size_text, (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        # Convert frame to QImage for display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera.setPixmap(QPixmap.fromImage(qimg))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MyAI()
    win.show()
    sys.exit(app.exec_())
