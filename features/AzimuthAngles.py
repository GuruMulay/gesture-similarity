import numpy as np
import math
from numpy import linalg as LA
from core.PCA import *
from Inclination_Angles import get_joint_combinations
from BendingAngles import calculate_basis_vectors

def calculate_azimuth_angle(frame, j):
    frame = frame.reshape(-1, 3)

    u_x,u_y,u_z = calculate_basis_vectors(frame)

    '''
    torso_joints = frame[[0, 1, 4, 8, 12], :]
    torso_pca =  PCA(torso_joints.T)[0]

    shoulder_vector = frame[4, :]-frame[8, :]
    spine_vector = frame[0, :]-frame[12, :]

    spine_vector /= np.linalg.norm(spine_vector)
    shoulder_vector /= np.linalg.norm(shoulder_vector)

    shoulder_angle_list = [np.dot(torso_pca[:,i], shoulder_vector) for i in range(3)]
    np.argmax(shoulder_angle_list)

    u_x_1 = torso_pca[:, np.argmax(shoulder_angle_list)]
    print 'My u_x: ', u_x
    print 'Pradys u_x: ', u_x_1
    '''

    angle = []
    for e in j:
        for i in e:
            v1 = (frame[i[0][1]] - frame[i[0][0]])
            v1_unit_vector = (v1 / (np.linalg.norm(v1))**2)
            dot_product = np.dot(u_x ,v1_unit_vector)

            vector_1 = u_x - (v1*dot_product)

            v2 = (frame[i[1][0]] - frame[i[1][1]])
            vector_2 = v2 - (v1 * np.dot(v2, v1_unit_vector))

            vector_1 /= np.linalg.norm(vector_1)
            vector_2 /= np.linalg.norm(vector_2)
            angle.append(np.degrees(np.arccos(np.dot(vector_1, vector_2))))

    return angle



def calculate_azimuth_angles_window_frame(frame):
    j = get_joint_combinations()
    return np.vstack([calculate_azimuth_angle(f, j) for f in frame])


def calculate_azimuth_angles_video(video):
    return [calculate_azimuth_angles_window_frame(f) for f in video]


def calculate_azimuth_angle_dataset(data_list):
    return [calculate_azimuth_angles_video(data) for data in data_list]

