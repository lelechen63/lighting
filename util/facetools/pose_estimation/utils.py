# Head Pose Estimation

# Input is an image file. Facial landmark positions are found by dlib. 

import dlib 
import cv2
import numpy as np
# *****************************************************************************
# Parameters, Input Data
# *****************************************************************************


def _get_full_model_points( filename='./dlib.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

def solve_pose_by_68_points( image_points, size = (480, 640), model_points_68=None):
        """
        Solve pose from all the 68 image points
        Return (rotation_vector, translation_vector) as pose.
        """
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Mouth left corner
            (150.0, -150.0, -125.0)      # Mouth right corner
        ]) / 4.5

        # model_points_68 = _get_full_model_points()

        # Camera internals
        focal_length = size[1]
        camera_center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, camera_center[0]],
             [0, focal_length, camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        dist_coeefs = np.zeros((4, 1))

        # Rotation vector and translation vector
        r_vec = np.array([[0.01891013], [0.08560084], [-3.14392813]])
        t_vec = np.array(
            [[-14.97821226], [-10.62040383], [-2053.03596872]])

        if r_vec is None:
            (_, rotation_vector, translation_vector) = cv2.solvePnP(
                model_points_68, image_points, camera_matrix, dist_coeefs)
            r_vec = rotation_vector
            t_vec = translation_vector

        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points_68,
            image_points,
            camera_matrix,
            dist_coeefs,
            rvec=r_vec,
            tvec=t_vec,
            useExtrinsicGuess=True)

        return (rotation_vector, translation_vector)
