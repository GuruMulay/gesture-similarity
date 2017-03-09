import numpy as np


#Spine-Base Normalization
def normalize_spine_base(data):
    b = np.copy(data)
    spine = np.zeros(3)
    for i in range(3):
        spine[i] = np.mean(b[:, i])
    m, n = b.shape

    for i in range(n):
        b[:, i] -= spine[i % 3]

    return b


def normalize_spine_base_dataset(data_list):
    return [normalize_spine_base(data) for data in data_list]


#Normalization by other joints
def normalize_by_joint(data, joint_index):
    print joint_index
    b = np.copy(data)
    joint = np.zeros(3)
    for i in range(3):
        joint[i] = np.mean(b[:, joint_index[i]])
    m, n = b.shape

    for i in range(n):
        b[:, i] -= joint[i % 3]

    return b


def normalize_joint_dataset(data_list, joint_index):
    return [normalize_by_joint(data, joint_index) for data in data_list]

