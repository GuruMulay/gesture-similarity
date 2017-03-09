import numpy as np
import math
from core.PCA import *


def calculate_basis_vectors(frame):
    torso_joints = frame[[0, 1, 4, 8, 12], :]
    torso_pca = PCA(torso_joints.T)[0]  # this will give a (3,3) matrix with u_x, u_y and u_z


    spine_vector = frame[0, :] - frame[12, :]
    shoulder_vector = frame[8, :] - frame[4, :]

    spine_vector /= np.linalg.norm(spine_vector)
    shoulder_vector /= np.linalg.norm(shoulder_vector)
    torso_vector = np.cross(shoulder_vector, spine_vector)
    #print 'shape of torso vector is: ', torso_vector.shape

    #PCA generates column eigen-vectors
    shoulder_angle_list = [np.dot(torso_pca[:,i], shoulder_vector) for i in range(3)]
    spine_angle_list = [np.dot(torso_pca[:,i], spine_vector) for i in range(3)]
    torso_angle_list = [np.dot(torso_pca[:,i], torso_vector) for i in range(3)]

    u_x_index = np.argmax(np.abs(shoulder_angle_list))
    u_y_index = np.argmax(np.abs(spine_angle_list))
    u_z_index = np.argmax(np.abs(torso_angle_list))

    u_x = torso_pca[u_x_index]
    u_y = torso_pca[u_y_index]
    u_z = torso_pca[u_z_index]

    return u_x, u_y, u_z


def calculate_bending_angles(frame):
    frame = frame.reshape(-1,3)
    u_x, u_y, u_z = calculate_basis_vectors(frame)

    bending_angles_x = [math.degrees(math.acos(np.dot(u_x, joint) / np.linalg.norm(joint))) for joint in frame]
    bending_angles_y = [math.degrees(math.acos(np.dot(u_y, joint) / np.linalg.norm(joint))) for joint in frame]
    bending_angles_z = [math.degrees(math.acos(np.dot(u_z, joint) / np.linalg.norm(joint))) for joint in frame]
    bending_angles = np.hstack([bending_angles_x[i], bending_angles_y[i], bending_angles_z[i]] for i in range(len(bending_angles_x)))

    return bending_angles


def calculate_bending_angles_window_frame(frame):
    return np.vstack([calculate_bending_angles(f) for f in frame])


def calculate_bending_angles_video(video):
    return [calculate_bending_angles_window_frame(f) for f in video]


def calculate_bending_angles_dataset(data_list):
    return [calculate_bending_angles_video(data) for data in data_list]



