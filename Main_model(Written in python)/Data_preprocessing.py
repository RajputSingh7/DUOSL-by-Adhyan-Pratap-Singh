import cv2
import mediapipe as mp
import os
import csv
import pandas as pd

INPUT_DIR = "Testing_dataset"
OUTPUT_DIR = "output_landmarks"
CSV_PATH = "landmarks_63_raw_mirrored_test.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_csv():    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.6
    )

    header = []
    for i in range(21):
        header += [f"lm{i}_x", f"lm{i}_y", f"lm{i}_z"]
    header.append("label")

    with open(CSV_PATH, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for label in os.listdir(INPUT_DIR):
            label_path = os.path.join(INPUT_DIR, label)

            if not os.path.isdir(label_path):
                continue

            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                image = cv2.imread(img_path)

                if image is None:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if not result.multi_hand_landmarks:
                    continue

                hand = result.multi_hand_landmarks[0]

                original = []
                mirrored = [] # from here mirroring takes place 

                for lm in hand.landmark:
                    x, y, z = lm.x, lm.y, lm.z

                    original.extend([x, y, z])
                    mirrored.extend([1.0 - x, y, z])

                writer.writerow(original + [label])
                writer.writerow(mirrored + [label]) # it doubles a dataset! 

                mp_draw.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS)
                cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), image)

    print("✅ Dataset augmented: CSV size doubled")
if __name__ == "__main__": # Used in other codes as well 
    generate_csv()