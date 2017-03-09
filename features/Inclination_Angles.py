from itertools import product, combinations
import numpy as np

dictionary ={
    1: [0,12],
    2: [3,12],
    4: [12,5],
    5: [4,6],
    6: [5,7,14],
    7: [6,13],
    8: [12,9],
    9: [8,10],
    10: [9,11,16],
    11: [10,15],
    12: [1,2,8,4]
}


def process_joints(d):
    for j in d:
        yield j, d[j]


def get_joint_combinations():
    a = [item for item in process_joints(dictionary)]
    g = [[i for i in product([item[0]], item[1])] for item in a]

    j = [[c for c in combinations(k, 2)] for k in g]

    return j


def calculate_inclination_angle(frame, j):
    frame = frame.reshape(-1, 3)

    angle = []
    for e in j:
        for i in e:
            v1 = (frame[i[0][0]] - frame[i[0][1]])
            v2 = (frame[i[1][0]] - frame[i[1][1]])
            v1 /= np.linalg.norm(v1)
            v2 /= np.linalg.norm(v2)

            angle.append(np.dot(v1,v2))

    return angle


def calculate_inclination_angle_dataset(data_list, remove_frames=2):
    inclination_angle = []
    j = get_joint_combinations()


    for data in data_list:
        window = []
        for win in data:
            alpha = []
            for frame in win:
                alpha.append(calculate_inclination_angle(frame, j))
            window.append(np.vstack(alpha))
        #print 'Length of wndow is: ', len(window)
        inclination_angle.append(window)

    return inclination_angle

