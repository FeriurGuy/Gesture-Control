import cv2
import mediapipe as mp # type: ignore
import numpy as np
import joblib # type: ignore
import time
from speaker import Speaker

class Button:
    def __init__(self, pos, text, size=[150, 50]):
        self.pos = pos
        self.size = size
        self.text = text
        self.is_active = True
        self.cooldown = 0

    def draw(self, img):
        x, y = self.pos
        w, h = self.size
        
        color = (0, 200, 0) if self.is_active else (0, 0, 200)
        status_text = "ON" if self.is_active else "OFF"
        label = f"{self.text}: {status_text}"
        
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(img, label, (x + 15, y + 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def is_clicked(self, x, y):
        if self.pos[0] < x < self.pos[0] + self.size[0] and \
           self.pos[1] < y < self.pos[1] + self.size[1]:
            if time.time() - self.cooldown > 1.0:
                self.is_active = not self.is_active
                self.cooldown = time.time()
                return True
        return False

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

try:
    model = joblib.load("models/gesture_model.pkl")
    print(f"[SYSTEM] Model loaded successfully. Classes: {model.classes_}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit()

gesture_map = {
    "hallo": "Halo, salam kenal semuanya",
    "nama": "Perkenalkan, nama saya Feri Ramadhan",
    "izinn": "Permisi, saya izin sebentar",
    "gerakan_hallo": "Ini adalah gerakan menyapa",
    "gerakan_anjay": "Luar biasa, mantap!"
}

def predict_gesture(landmarks):
    data = np.array(landmarks).flatten().reshape(1, -1)
    return model.predict(data)[0]

def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    speaker = Speaker()
    btn_voice = Button([20, 60], "VOICE")
    
    prev_gesture = None
    gesture_streak = 0
    last_detected = None
    display_text = "System Ready"

    print("[SYSTEM] Camera initialized. Press 'Q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        # Header UI
        cv2.rectangle(frame, (0, 0), (w, 50), (40, 40, 40), -1)
        cv2.putText(frame, "GESTURE RECOGNITION SYSTEM V1.0", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # Finger cursor logic
                idx_finger = hand_landmarks.landmark[8]
                cx, cy = int(idx_finger.x * w), int(idx_finger.y * h)
                cv2.circle(frame, (cx, cy), 8, (255, 0, 0), -1)
                
                if btn_voice.is_clicked(cx, cy):
                    print(f"[EVENT] Button Clicked. Voice Active: {btn_voice.is_active}")

                # Prediction logic
                gesture = predict_gesture(landmarks)
                
                wrist = hand_landmarks.landmark[0]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, gesture.upper(), (wx, wy + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if gesture == last_detected:
                    gesture_streak += 1
                else:
                    gesture_streak = 0
                    last_detected = gesture

                if gesture_streak > 15:
                    if gesture != prev_gesture:
                        if gesture in gesture_map:
                            spoken_text = gesture_map[gesture]
                            display_text = f"Speaking: {spoken_text}"
                            
                            if btn_voice.is_active:
                                speaker.say(spoken_text)
                            else:
                                display_text += " (Muted)"
                                
                        prev_gesture = gesture
                        gesture_streak = 0  

        btn_voice.draw(frame)
        
        # Footer UI
        cv2.rectangle(frame, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, display_text, (15, h-12), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    speaker.stop()

if __name__ == "__main__":
    main()