import numpy as np


def calculate_velocity(data):
    samples, dimensions = data.shape
    velocity = np.zeros((samples-2, dimensions))
    for k in range(1, samples-1):
        velocity[k-1, :] = (data[k+1, :] - data[k-1, :])/2

    return velocity

# added for single video case
def calculate_for_video_single(d, remove_frames=1):
    return [calculate_velocity(d)[remove_frames:-remove_frames, :]]  # remove_frames:-remove_frames


def calculate_for_video(data, remove_frames=1):
    return [calculate_velocity(d)[remove_frames:-remove_frames, :] for d in data]


def calculate_velocity_dataset(data_list):
    return [calculate_for_video(data) for data in data_list]


