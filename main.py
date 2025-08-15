import cv2
import time
import numpy as np
from mediapipe import solutions
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.framework.formats import landmark_pb2


def draw_landmarks_on_image(rgb_image, detection_result):
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


model_path = r'C:\Users\aeonic\PycharmProjects\faceTracker\face_landmarker.task'

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

mp_drawing = mp.solutions.drawing_utils

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

with FaceLandmarker.create_from_options(options) as landmarker:
    while capture.isOpened():

        ret, frame = capture.read()
        frame = cv2.resize(frame, (800, 600))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Making predictions using holistic model
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = landmarker.detect(mp_image)
        image.flags.writeable = True

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        annotated_image = draw_landmarks_on_image(image, results)

        # Calculating the FPS
        currentTime = time.time()
        fps = 1 / (currentTime - previousTime)
        previousTime = currentTime

        # Displaying FPS on the image
        cv2.putText(annotated_image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        if len(results.face_blendshapes) > 0:
            for shape in results.face_blendshapes[0]:
                if shape.category_name == "jawOpen":
                    cv2.putText(annotated_image, shape.category_name + ": " + str(shape.score), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        print(results.facial_transformation_matrixes)
        # Display the resulting image
        cv2.imshow("Facial and Hand Landmarks", annotated_image)


        # Enter key 'q' to break the loop
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# mp_holistic = mp.solutions.holistic
# holistic_model = mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
#
# # Initializing the drawing utils for drawing the facial landmarks on image
# mp_drawing = mp.solutions.drawing_utils
#
# while capture.isOpened():
#     # capture frame by frame
#     ret, frame = capture.read()
#
#     # resizing the frame for better view
#     frame = cv2.resize(frame, (800, 600))
#
#     # Converting the from BGR to RGB
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # Making predictions using holistic model
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     results = holistic_model.process(image)
#     image.flags.writeable = True
#
#     # Converting back the RGB image to BGR
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#     # Drawing the Facial Landmarks
#     mp_drawing.draw_landmarks(
#         image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_CONTOURS,
#         mp_drawing.DrawingSpec(
#             color=(255, 0, 255),
#             thickness=1,
#             circle_radius=1
#         ),
#         mp_drawing.DrawingSpec(
#             color=(0, 255, 255),
#             thickness=1,
#             circle_radius=1
#         )
#     )
#     print(results.face_landmarks)
#     #
#     # # Drawing Right hand Land Marks
#     # mp_drawing.draw_landmarks(
#     #     image,
#     #     results.right_hand_landmarks,
#     #     mp_holistic.HAND_CONNECTIONS
#     # )
#     #
#     # # Drawing Left hand Land Marks
#     # mp_drawing.draw_landmarks(
#     #     image,
#     #     results.left_hand_landmarks,
#     #     mp_holistic.HAND_CONNECTIONS
#     # )
#
#     # Calculating the FPS
#     currentTime = time.time()
#     fps = 1 / (currentTime - previousTime)
#     previousTime = currentTime
#
#     # Displaying FPS on the image
#     cv2.putText(image, str(int(fps)) + " FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#
#     # Display the resulting image
#     cv2.imshow("Facial and Hand Landmarks", image)
#
#     # Enter key 'q' to break the loop
#     if cv2.waitKey(5) & 0xFF == ord('q'):
#         break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()
