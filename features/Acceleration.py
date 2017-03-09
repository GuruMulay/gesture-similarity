import numpy as np
from Velocity import calculate_velocity


def calculate_acceleration(data):
    return calculate_velocity(calculate_velocity(data))


def calculate_for_video(data):
    return [calculate_acceleration(d) for d in data]

def calculate_acceleration_dataset(data_list):
    return [calculate_for_video(data) for data in data_list]
    #return [calculate_acceleration(data) for data in data_list]