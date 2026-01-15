import cv2
import mediapipe as mp # type: ignore

class HandDetector:
    def __init__(self, max_hands=1, detection_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands, min_detection_confidence=detection_conf)
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        return results

    def draw_landmarks(self, frame, hand_landmarks):
        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

    def classify_gesture(self, hand_landmarks):
        tips = [4, 8, 12, 16, 20]
        fingers = []

        for tip in tips[1:]: 
            fingers.append(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y)

        thumb_up = hand_landmarks.landmark[tips[0]].y < hand_landmarks.landmark[tips[0] - 2].y
        open_hand = all(fingers)
        fist = not any(fingers) and not thumb_up
        peace = fingers[0] and fingers[1] and not fingers[2] and not fingers[3]

        if open_hand:
            return "open_hand"
        elif thumb_up:
            return "thumbs_up"
        elif peace:
            return "peace"
        elif fist:
            return "fist"
        else:
            return None
