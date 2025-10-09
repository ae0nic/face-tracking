import math

import cv2
import time

import numpy
import numpy as np
from mediapipe import solutions
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

class Landmarker(object):
    def draw_facial_landmarks(self, rgb_image, detection_result):
        face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(face_landmarks_list)):
            face_landmarks = face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())

        return annotated_image

    def draw_pose_landmarks(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the face landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                mp.solutions.pose.POSE_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style())

            return annotated_image


    def draw_hand_landmarks(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the face landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=hand_landmarks_proto,
                connections=mp.solutions.hands_connections.HAND_CONNECTIONS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_hand_connections_style())

        return annotated_image

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.deinit()

    def __enter__(self):
        return self

    def __init__(self):
        self.init_time = time.time()
        self.face_model_path = r'landmarker/face_landmarker.task'
        # self.hand_model_path = r'landmarker/hand_landmarker.task'
        self.pose_model_path = r'landmarker/pose_landmarker_full.task'

        # (0) in VideoCapture is used to connect to your computer's default camera
        self.capture = cv2.VideoCapture(0)

        # Initializing current time and precious time for calculating the FPS
        self.previousTime = 0
        self.currentTime = 0

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # HandLandmarker = mp.tasks.vision.HandLandmarker
        # HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

        self.mp_drawing = mp.solutions.drawing_utils

        self.face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.face_model_path),
            running_mode=VisionRunningMode.VIDEO,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )

        # self.hand_options = HandLandmarkerOptions(
        #     base_options=BaseOptions(model_asset_path=self.hand_model_path),
        #     num_hands=2,
        #     running_mode=VisionRunningMode.VIDEO
        # )

        self.pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.pose_model_path),
            num_poses=1,
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_landmarker = FaceLandmarker.create_from_options(self.face_options)
        # self.hand_landmarker = HandLandmarker.create_from_options(self.hand_options)
        self.pose_landmarker = PoseLandmarker.create_from_options(self.pose_options)
        self.timestamp = 0


    def run(self):
        # Get the input from the webcam and prepare it for processing
        ret, frame = self.capture.read()
        frame = cv2.resize(frame, (800, 600))
        frame = cv2.flip(frame, 1) # Horizontal flip
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Making predictions using face and hand models
        # Allow pass-by-reference by making the image read-only
        image.flags.writeable = False
        face_results = self.face_landmarker.detect_for_video(mp_image, int(self.timestamp * 1000))
        # hand_results = self.hand_landmarker.detect_for_video(mp_image, int(self.timestamp * 1000))
        pose_results = self.pose_landmarker.detect_for_video(mp_image, int(self.timestamp * 1000))
        image.flags.writeable = True

        # Extract face angles from the model output
        roll = pitch = yaw = 0
        x    = y     = z   = 0
        if len(pose_results.pose_landmarks) > 0:
            poses = pose_results.pose_world_landmarks[0]
            (x, y, z) = ((poses[11].x + poses[12].x) / 2., (poses[11].y + poses[12].y) / 2., (poses[11].z + poses[12].z) / 2.)
        face_detected = False
        if len(face_results.facial_transformation_matrixes) > 0:
            face_detected = True

            # Orthonormalize vectors (they should already be orthonormal, but do this just in case)
            # Credit to (https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) for process
            vi = np.array(face_results.facial_transformation_matrixes[0][0][:3])  # Currently, we discard translation data
            vj = np.array(face_results.facial_transformation_matrixes[0][1][:3])
            vk = np.array(face_results.facial_transformation_matrixes[0][2][:3])
            # Gram-Schmidt process for orthonormalization
            ui = vi
            ei = ui / np.linalg.norm(ui)
            uj = vj - ei * numpy.dot(ui, vj)
            ej = uj / np.linalg.norm(uj)
            uk = vk - ei * numpy.dot(ui, vk) - ej * numpy.dot(uj, vk)
            ek = uk / np.linalg.norm(uk)

            # Making sure that we orthonormalized properly
            epsilon = 1e-11
            assert abs(1 - np.linalg.norm(ei)) <= epsilon
            assert abs(1 - np.linalg.norm(ej)) <= epsilon
            assert abs(1 - np.linalg.norm(ek)) <= epsilon
            assert abs(np.dot(ei, ej)) <= epsilon
            assert abs(np.dot(ej, ek)) <= epsilon
            assert abs(np.dot(ei, ek)) <= epsilon


            # Now we can extract angles
            # The order in which pitch, roll, and yaw are applied is important
            # We assume: Yaw first, then pitch, then roll
            # Credit to (https://www.geometrictools.com/Documentation/EulerAngles.pdf) for pseudocode
            rotation_matrix = np.array([ui, uj, uk])
            if rotation_matrix[1][2] < 1:
                if rotation_matrix[1][2] > -1:
                    pitch = math.asin(-rotation_matrix[1][2])
                    yaw = math.atan2(rotation_matrix[0][2], rotation_matrix[2][2])
                    roll = math.atan2(rotation_matrix[1][0], rotation_matrix[1][1])
                else:
                    pitch = np.pi / 2
                    yaw = -math.atan2(-rotation_matrix[0][1], rotation_matrix[0][0])
                    roll = 0
            else:
                pitch = -np.pi / 2
                yaw = math.atan2(-rotation_matrix[0][1], rotation_matrix[0][0])
                roll = 0

        else:
            print("No face detected")



        # Calculating the FPS to place on the image and update timestamp
        self.currentTime = time.time() - self.init_time
        self.timestamp += self.currentTime - self.previousTime
        fps = 1 / (self.currentTime - self.previousTime)
        self.previousTime = self.currentTime

        ## Enable for debugging
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated_image = image
        if len(face_results.face_landmarks) > 0:
            annotated_image = self.draw_facial_landmarks(image, face_results)
        if len(pose_results.pose_landmarks) > 0:
            annotated_image = self.draw_pose_landmarks(annotated_image, pose_results)
        cv2.putText(annotated_image, "Yaw: " + str(yaw), (10, 130),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, "Pitch: " + str(pitch), (10, 150),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, "Roll: " + str(roll), (10, 170),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, "X: " + str(x), (10, 200), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, "Y: " + str(y), (10, 230), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, "Z: " + str(z), (10, 260), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Landmarks", annotated_image)
        cv2.waitKey(1)

        return (face_detected, (yaw, pitch, roll, x, y, z), self.timestamp,
                face_results.face_blendshapes[0] if len(face_results.face_blendshapes) > 0 else [],
                pose_results.pose_world_landmarks[0] if len(pose_results.pose_world_landmarks) > 0 else [])

    def deinit(self):
        # Release the webcam and destroy debug windows when done
        self.capture.release()
        cv2.destroyAllWindows()
