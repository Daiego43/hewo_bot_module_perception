from src.hewo_bot_module_perception.perception.vision.mpface import MediaPeopleFaces
from src.hewo_bot_module_perception.perception.vision.mppose import MediaPeoplePoses
from src.hewo_bot_module_perception.perception.vision.mphands import MediaPeopleHands


class MediaPeople:
    def __init__(self):
        self.face = MediaPeopleFaces()
        self.pose = MediaPeoplePoses()
        self.hand = MediaPeopleHands()
        self.face_list = []
        self.pose_list = []
        self.hand_list = []
        self.bbox_list = []

    def update_info(self, rgb_frame, color_image):
        self.face.update_info(rgb_frame=rgb_frame, color_image=color_image)
        self.pose.update_info(rgb_frame=rgb_frame)
        self.hand.update_info(rgb_frame=rgb_frame)
        self.face_list = self.face.get_face_list()
        self.pose_list = self.pose.get_pose_list()
        self.hand_list = self.hand.get_hand_list()
        self.bbox_list = self.face.get_bbox_list()

    def draw_landmarks(self, color_image, rgb_frame):
        self.update_info(color_image=color_image, rgb_frame=rgb_frame)
        color_image = self.face.draw_landmarks(color_image, rgb_frame)
        color_image = self.pose.draw_landmarks(color_image, rgb_frame)
        color_image = self.hand.draw_landmarks(color_image, rgb_frame)
        return color_image

    def plt_3D_repr(self, ax):
        l = [
            self.hand_list,
            self.face_list,
            self.pose_list
        ]
        # print(f"Faces: {len(self.face_list)}, Hands: {len(self.hand_list)}, Pose: {len(self.pose_list)}")
        for s in l:
            if s is not None:
                for landmarks in s:
                    x = landmarks[:, 0]
                    y = landmarks[:, 1]
                    z = landmarks[:, 2]
                    ax.scatter(x, y, z)
