import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

from collections import deque
class SignMLP(nn.Module):
    def __init__(self, input_dim=126, num_classes=26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SignMLP().to(device)
model.load_state_dict(torch.load("sign_mlp_126_to_26.pth", map_location=device))
model.eval()
LABELS = [chr(ord('A') + i) for i in range(26)]
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,        # MUST be False
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
cap = cv2.VideoCapture(0)

mp_draw = mp.solutions.drawing_utils
pred_buffer = deque(maxlen=7)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    features = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

    # Ensure EXACTLY 126 features
    if len(features) == 126:
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

        pred_buffer.append(pred.item())

        # Majority vote smoothing
        final_pred = max(set(pred_buffer), key=pred_buffer.count)
        label = LABELS[final_pred]

        # Confidence threshold
        if conf.item() > 0.7:
            cv2.putText(
                frame,
                f"Sign: {label}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )
        else:
            cv2.putText(
                frame,
                "Sign: Unknown",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

    cv2.imshow("Sign Language Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
hands.close()
