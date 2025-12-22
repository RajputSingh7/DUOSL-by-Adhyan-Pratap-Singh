import cv2
import mediapipe as mp
import pandas as pd
import os


IMAGE_ROOT = "dataset/phrases"
OUTPUT_CSV = "dataset/landmarks.csv"
IMG_SIZE = 640


mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=True,
    min_detection_confidence=0.7,
    refine_face_landmarks=False
)

def get_hand_landmarks(hand_landmarks):
    if hand_landmarks:
        return [coord for lm in hand_landmarks.landmark
                      for coord in (lm.x, lm.y, lm.z)]
    else:
        return [0.0] * 63

rows = []

for label in os.listdir(IMAGE_ROOT):
    class_path = os.path.join(IMAGE_ROOT, label)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing sign: {label}")

    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        image = cv2.imread(img_path)
        if image is None:
            continue

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = holistic.process(rgb)

        right_hand = get_hand_landmarks(results.right_hand_landmarks)
        left_hand  = get_hand_landmarks(results.left_hand_landmarks)

        feature_vector = right_hand + left_hand 
        rows.append(feature_vector + [label])


columns = []
for hand in ["RH", "LH"]:
    for i in range(21):
        for axis in ["x", "y", "z"]:
            columns.append(f"{hand}_{i}_{axis}")

columns.append("label")

df = pd.DataFrame(rows, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("Dataset saved to:", OUTPUT_CSV)
print("Total samples:", len(df))