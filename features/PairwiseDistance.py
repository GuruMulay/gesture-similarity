import numpy as np

def calculate_pairwise_distance(frame):
#Removes the joints which form the bones, dimension is 120
    temp = [(0,1),(2,3),(2,12),(8,12),(4,12),(1,12),(4,5),(5,6),(6,7),(7,13),(6,14),(8,9),(9,10),(10,11),(11,15),(10,16)]
    frame = frame.reshape(-1,3)
    joints = frame.shape[0]
    pairwise_distance = []

    for i in xrange(joints-1):
        for j in xrange(i+1, joints):
            if (i,j) not in temp:
                pairwise_distance.append(np.linalg.norm(frame[i,:]-frame[j,:]))

    return pairwise_distance


def calculate_pairwise_distance_frame(frame):
#Includes all the combinations of joints, dimension is 136
    frame = frame.reshape(-1,3)
    joints = frame.shape[0]
    pairwise_distance = []

    for i in xrange(joints-1):
        for j in xrange(i+1, joints):
            pairwise_distance.append(np.linalg.norm(frame[i,:]-frame[j,:]))

    return pairwise_distance


def calculate_pairwise_distance_window_frame(frame):
    return np.vstack([calculate_pairwise_distance(f) for f in frame])


def calculate_pairwise_distance_video(video):
    return [calculate_pairwise_distance_window_frame(f) for f in video]


def calculate_pairwise_distance_dataset(data_list):
    return [calculate_pairwise_distance_video(data) for data in data_list]
