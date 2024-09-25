# -*- coding: utf-8 -*-

import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt


def get_vessel_length(image_path, height_person_cm):
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize mediapipe pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Load image
    image = cv2.imread(image_path)

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run detection
    results = pose.process(image)
    annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    # bg_image = np.zeros(image.shape, dtype=np.uint8)
    # bg_image[:] = BG_COLOR
    # annotated_image = np.where(condition, annotated_image, bg_image)
    # Draw pose landmarks on the image.
    # mp_drawing.draw_landmarks(
    #     annotated_image,
    #     results.pose_landmarks,
    #     mp_pose.POSE_CONNECTIONS,
    #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # cv2.imwrite('./annotated_image' + '.png', annotated_image)
    # Plot pose world landmarks.
    # mp_drawing.plot_landmarks(
    #     results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    # Get image height and width
    height, width, _ = image.shape

    # Define landmarks for heart and forehead
    heart_landmarks = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW]

    forehead_landmarks = [mp_pose.PoseLandmark.LEFT_EYE, mp_pose.PoseLandmark.RIGHT_EYE]

    foot_landmarks = [mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL]

    # Extract landmark locations
    keypoints = {}
    for landmark in mp_pose.PoseLandmark:
        keypoints[landmark] = (int(results.pose_landmarks.landmark[landmark].x * width),
                            int(results.pose_landmarks.landmark[landmark].y * height))

    # Calculate the heart location
    heart_x = sum([keypoints[landmark][0] for landmark in heart_landmarks]) // len(heart_landmarks)
    heart_y = sum([keypoints[landmark][1] for landmark in heart_landmarks]) // len(heart_landmarks)
    heart = (heart_x, heart_y)

    # Calculate the forehead location
    forehead_x = sum([keypoints[landmark][0] for landmark in forehead_landmarks]) // len(forehead_landmarks)
    forehead_y = sum([keypoints[landmark][1] for landmark in forehead_landmarks]) // len(forehead_landmarks)
    forehead = (forehead_x, forehead_y)

    # Calculate the foot location
    foot_x = sum([keypoints[landmark][0] for landmark in foot_landmarks]) // len(foot_landmarks)
    foot_y = sum([keypoints[landmark][1] for landmark in foot_landmarks]) // len(foot_landmarks)
    foot = (foot_x, foot_y)

    # Calculate heart-to-forehead distance
    heart_to_forehead = math.dist(heart, forehead)

    # Calculate the hand locations
    left_hand = keypoints[mp_pose.PoseLandmark.LEFT_WRIST]
    right_hand = keypoints[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Calculate heart-to-hand distances
    heart_to_left_hand = math.dist(heart, left_hand)
    heart_to_right_hand = math.dist(heart, right_hand)
    

    height_person_pixels = math.dist(foot, forehead)
    height_person_pixels = 1.05*height_person_pixels
    ratio = height_person_cm/height_person_pixels
    blood_vessel_length = 0.01*abs(heart_to_forehead - heart_to_right_hand)*ratio

    # print("Heart-to-Forehead Distance:", heart_to_forehead)
    # print("Heart-to-Left-Hand Distance:", heart_to_left_hand)
    # print("Heart-to-Right-Hand Distance:", heart_to_right_hand)
    return blood_vessel_length
