import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class MediaPeopleHands:
    def __init__(self, max_num_hands=2, detection_confidence=0.9, tracking_confidence=0.9):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.results = None
        self.hand_list = []

    def update_info(self, rgb_frame):
        self.results = self.hands.process(rgb_frame)
        hand_list = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                hand = []
                for landmark in hand_landmarks.landmark:
                    hand.append([landmark.x, landmark.y, landmark.z])
                hand = np.array(hand)
                hand_list.append(hand)
        self.hand_list = hand_list

    def get_hand_list(self):
        return self.hand_list

    def get_hand_results(self):
        return self.results

    def draw_landmarks(self, color_image, rgb_frame):
        self.update_info(rgb_frame)
        if self.results.multi_hand_landmarks is not None:
            for hand_landmarks in self.results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    color_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )
        return color_image

    def plt_3D_repr(self, ax):
        if self.hand_list is not []:
            for landmarks in self.hand_list:
                x = landmarks[:, 0]
                y = landmarks[:, 1]
                z = landmarks[:, 2]
                ax.scatter(x, y, z)
            ax.set_title('hands 3D Representation')
