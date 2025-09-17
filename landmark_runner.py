from cgitb import handler

import cv2
import time

import numpy
import numpy as np
from mediapipe import solutions
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.framework.formats import landmark_pb2

def draw_facial_landmarks(rgb_image, detection_result):
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

def draw_hand_landmarks(rgb_image, detection_result):
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

def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))
    fig, ax = plt.subplots(figsize=(12, 12))

    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")

    plt.tight_layout()
    plt.show()

def run(data_queue):
    face_model_path = r'C:\Users\s612751\Downloads\face-tracking-master\face-tracking-master\face_landmarker.task'
    hand_model_path = r'C:\Users\s612751\Downloads\face-tracking-master\face-tracking-master\hand_landmarker.task'

    # (0) in VideoCapture is used to connect to your computer's default camera
    capture = cv2.VideoCapture(0)

    # Initializing current time and precious time for calculating the FPS
    previousTime = 0
    currentTime = 0

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

    mp_drawing = mp.solutions.drawing_utils

    face_options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=face_model_path),
        running_mode=VisionRunningMode.VIDEO,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_tracking_confidence=0.3
    )

    hand_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=hand_model_path),
        num_hands=2,
        running_mode=VisionRunningMode.VIDEO
    )

    with FaceLandmarker.create_from_options(face_options) as face_landmarker, \
            HandLandmarker.create_from_options(hand_options) as hand_landmarker:
        timestamp = 0
        while capture.isOpened():

            ret, frame = capture.read()
            frame = cv2.resize(frame, (800, 600))
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            # Making predictions using holistic model
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            face_results = face_landmarker.detect_for_video(mp_image, int(timestamp * 1000))
            hand_results = hand_landmarker.detect_for_video(mp_image, int(timestamp * 1000))
            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            annotated_image = draw_facial_landmarks(image, face_results)
            annotated_image = draw_hand_landmarks(annotated_image, hand_results)



            roll = pitch = yaw = 0
            if len(face_results.facial_transformation_matrixes) > 0:
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
                        pitch = np.asin(-rotation_matrix[1][2])
                        yaw = np.atan2(rotation_matrix[0][2], rotation_matrix[2][2])
                        roll = np.atan2(rotation_matrix[1][0], rotation_matrix[1][1])
                    else:
                        pitch = np.pi / 2
                        yaw = -np.atan2(-rotation_matrix[0][1], rotation_matrix[0][0])
                        roll = 0
                else:
                    pitch = -np.pi / 2
                    yaw = np.atan2(-rotation_matrix[0][1], rotation_matrix[0][0])
                    roll = 0

            else:
                print("No face detected")

            cv2.putText(annotated_image, "Yaw: " + str(yaw), (10, 130),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_image, "Pitch: " + str(pitch), (10, 150),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_image, "Roll: " + str(roll), (10, 170),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Calculating the FPS
            currentTime = time.time()
            timestamp += currentTime - previousTime
            fps = 1 / (currentTime - previousTime)
            previousTime = currentTime

            # Displaying FPS on the image
            cv2.putText(annotated_image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            if len(face_results.face_blendshapes) > 0:
                for shape in face_results.face_blendshapes[0]:
                    if shape.category_name == "jawOpen":
                        cv2.putText(annotated_image, shape.category_name + ": " + str(shape.score), (10, 100),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            data_queue.append(timestamp)
            # Display the resulting image
            cv2.imshow("Facial and Hand Landmarks", annotated_image)

            # Enter key 'q' to break the loop
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    # When all the process is done
    # Release the capture and destroy all windows
    capture.release()
    cv2.destroyAllWindows()
