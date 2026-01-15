import cv2
import mediapipe as mp # type: ignore
import numpy as np
import csv
import os
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def draw_ui(img, current, total, gesture_name, recording, countdown_val):
    h, w, _ = img.shape
    
    # Progress Bar
    cv2.rectangle(img, (0, h-30), (w, h), (30, 30, 30), -1)
    bar_width = int((current / total) * w)
    cv2.rectangle(img, (0, h-30), (bar_width, h), (0, 200, 0), -1)
    
    status = f"Samples: {current}/{total}"
    cv2.putText(img, status, (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Header Info
    cv2.putText(img, f"TARGET: {gesture_name.upper()}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if not recording:
        cv2.putText(img, "Press 'R' to Record", (w//2 - 100, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    elif countdown_val > 0:
        cv2.putText(img, str(countdown_val), (w//2 - 20, h//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4)
    else:
        cv2.circle(img, (w - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(img, "REC", (w - 80, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def collect_gesture(gesture_name, num_samples=150):
    save_path = f"data/{gesture_name}.csv"
    os.makedirs("data", exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    
    data = []
    recording = False
    start_time = 0
    
    print(f"[SYSTEM] Initializing collection for: '{gesture_name}'")
    print("[INSTRUCTION] Press 'R' to start. Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        countdown_val = 0
        if recording:
            elapsed = time.time() - start_time
            if elapsed < 3:
                countdown_val = 3 - int(elapsed)
            else:
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        data.append(landmarks)
                
                if len(data) >= num_samples:
                    print(f"[SUCCESS] Collected {len(data)} samples.")
                    break

        draw_ui(frame, len(data), num_samples, gesture_name, recording, countdown_val)
        cv2.imshow("Data Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r') and not recording:
            recording = True
            start_time = time.time()

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if len(data) > 0:
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(data)
        print(f"[SYSTEM] Data saved to {save_path}")

if __name__ == "__main__":
    name = input("Enter gesture name: ").strip().lower()
    if name:
        collect_gesture(name)