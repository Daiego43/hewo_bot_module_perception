import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection


class MediaPeopleFaces:
    def __init__(self, max_num_faces=1, detection_confidence=0.9, tracking_confidence=0.9):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.face_detector = mp_face_detection.FaceDetection(
            model_selection=max_num_faces,
            min_detection_confidence=detection_confidence
        )
        self.mesh_results = None
        self.detector_results = None
        self.bbox_list = []
        self.face_list = []
        self.face_connections = mp_face_mesh.FACEMESH_TESSELATION

    def get_bbox_list(self):
        return self.bbox_list

    def get_face_list(self):
        return self.face_list

    def get_mesh_results(self):
        return self.mesh_results

    def get_detector_results(self):
        return self.detector_results

    def update_info(self, color_image, rgb_frame):
        self.detector_results = self.face_detector.process(rgb_frame)
        self.mesh_results = self.face_mesh.process(rgb_frame)
        bbox_list = []
        if self.detector_results.detections:
            for detection in self.detector_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = color_image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                dims = (x, y, w, h)
                bbox_list.append(dims)
        self.bbox_list = bbox_list

        face_list = []
        if self.mesh_results.multi_face_landmarks:
            for face_landmarks in self.mesh_results.multi_face_landmarks:
                face = []
                for landmark in face_landmarks.landmark:
                    face.append(np.array([landmark.x, landmark.y, landmark.z]))
                face = np.array(face)
                face_list.append(face)
        self.face_list = face_list

    def draw_landmarks(self, color_image, rgb_frame):
        self.update_info(color_image, rgb_frame)
        img = self.draw_mesh(color_image)
        img = self.draw_boxes(img)
        return img

    def draw_mesh(self, color_image):
        if self.mesh_results.multi_face_landmarks is not None:
            for face_landmarks in self.mesh_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    color_image,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1)
                )
        return color_image

    def draw_boxes(self, color_image):
        if self.bbox_list:
            for box in self.bbox_list:
                x, y, w, h = box
                cv2.rectangle(color_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return color_image

    def plt_3D_repr(self, ax):
        if self.face_list:
            for landmarks in self.face_list:
                x = landmarks[:, 0]
                y = landmarks[:, 1]
                z = landmarks[:, 2]
                ax.scatter(x, y, z)
            ax.set_title('hands 3D Representation')
