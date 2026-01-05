import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from collections import deque, Counter
import socket
import struct
import time
import json

HOST = "127.0.0.1"
PORT1 = 5005 # Sign port
PORT2 = 5001 # Camera port

setlandmarks = 1
# Create socket
server_socket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket1.bind((HOST, PORT1))
server_socket1.listen(1)
server_socket2.bind((HOST, PORT2))
server_socket2.listen(1)
print("Waiting for Unity connection...")
conn2, addr2 = server_socket2.accept()
conn1, addr1 = server_socket1.accept()
conn1.setblocking(False)
print("Connected to:", addr2)

def send_sign(sign, confidence):
    data = {
        "sign": sign,
        "confidence": float(confidence),
        "time": time.time()
    }
    message = json.dumps(data)+'\n'
    conn1.sendall(message.encode("utf-8"))
# ================= MODEL =================
class SignLanguageMLP(nn.Module):
    def __init__(self, input_size=63, num_classes=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ================= DEVICE =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SignLanguageMLP(input_size=63, num_classes=28).to(device)
model.load_state_dict(torch.load("sign_model.pth", map_location=device))
model.eval()

# ================= LABELS (28 ONLY) =================
LABELS = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','space'
]

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ================= NORMALIZED LANDMARKS =================
def extract_normalized_63(hand_landmarks):
    lm = hand_landmarks.landmark

    # Wrist as origin
    wx, wy, wz = lm[0].x, lm[0].y, lm[0].z

    # Middle MCP for scale
    mx, my, mz = lm[9].x, lm[9].y, lm[9].z

    scale = np.sqrt(
        (mx - wx) ** 2 +
        (my - wy) ** 2 +
        (mz - wz) ** 2
    )

    if scale < 1e-6:
        scale = 1.0

    data = []
    for p in lm:
        data.extend([
            (p.x - wx) / scale,
            (p.y - wy) / scale,
            (p.z - wz) / scale
        ])

    return np.array(data, dtype=np.float32)

# ================= TEXT BUILDER =================cam
prediction_buffer = deque(maxlen=15)
sentence = ""
last_committed = None
COOLDOWN_FRAMES = 25
cooldown = 0

def get_stable_prediction(buffer):
    if len(buffer) < buffer.maxlen:
        return None
    return Counter(buffer).most_common(1)[0][0]

# ================= OPENCV LOOP =================
cap = cv2.VideoCapture(0)
def send_sign(sign, confidence):
    data = {
        "sign": sign,
        "confidence": float(confidence),
        "time": time.time()
    }
    message = json.dumps(data) + "\n"
    conn1.sendall(message.encode("utf-8"))
def SendImage(mode):
    if mode:
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    data = buffer.tobytes()

    # Send length + data
    message = struct.pack("Q", len(data)) + data
    conn2.sendall(message)
def LandmarkToggle():
    try:
        global setlandmarks
        setlandmarks = conn1.recv(1)[0]
        print("Received")
        print(setlandmarks)
    except:
        pass
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    predicted_label = None
    confidence = 0.0

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        landmarks_63 = extract_normalized_63(hand)
        x = torch.tensor(landmarks_63).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
            predicted_label = LABELS[pred]
            

        prediction_buffer.append(predicted_label)
        LandmarkToggle()
        if setlandmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)


        stable = get_stable_prediction(prediction_buffer)

        if stable and cooldown == 0 and stable != "not":
            if stable == "space":
                sentence += " "
                send_sign(" ",confidence)
            elif stable == "del":
                sentence = sentence[:-1]
                send_sign("-",confidence)
            else:
                sentence += stable
                send_sign(stable,confidence)
            cooldown = COOLDOWN_FRAMES
    SendImage(True)


    if cooldown > 0:
        cooldown -= 1

    # ================= UI =================



    

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
conn2.close()
