import numpy as np
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class MediaPeoplePoses:
    def __init__(self, num_poses=1, detection_confidence=0.9, tracking_confidence=0.9):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.results = None
        self.poselist = []

    def update_info(self, rgb_frame):
        self.results = self.pose.process(rgb_frame)
        poselist = []
        if self.results.pose_landmarks:
            pose_landmarks = self.results.pose_landmarks
            pose = []
            for landmark in pose_landmarks.landmark:
                pose.append(np.array([landmark.x, landmark.y, landmark.z]))
            pose = np.array(pose)
            poselist.append(pose)
        self.poselist = poselist

    def get_pose_list(self):
        return self.poselist

    def get_results(self):
        return self.results

    def draw_landmarks(self, color_image, rgb_frame):
        self.update_info(rgb_frame)
        if self.results is not None:
            mp_drawing.draw_landmarks(
                color_image,
                self.results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )
        return color_image

    def plt_3D_repr(self, ax):
        if self.poselist is not []:
            for pose_landmarks in self.poselist:
                x = pose_landmarks[:, 0]
                y = pose_landmarks[:, 1]
                z = pose_landmarks[:, 2]
                ax.scatter(x, y, z)
            ax.set_title('Pose 3D Representation')
