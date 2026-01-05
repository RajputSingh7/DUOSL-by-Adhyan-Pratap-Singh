import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=True,   
    min_detection_confidence=0.5
)

def extract_and_draw_landmarks(image_path):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect")

    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    
    results = hands.process(image_rgb)

    
    landmarks_126 = np.zeros(126, dtype=np.float32)

    idx = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            
            for lm in hand_landmarks.landmark:
                landmarks_126[idx] = lm.x
                landmarks_126[idx + 1] = lm.y
                landmarks_126[idx + 2] = lm.z
                idx += 3

    
    cv2.imshow("Hand Landmarks", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return landmarks_126



image_path = "hand_image.jpg"  # <-- replace with your image path
landmarks = extract_and_draw_landmarks(image_path)

print("Landmarks shape:", landmarks.shape)
print("Landmarks (first 10 values):", landmarks[:10])