import numpy as np
from sklearn.preprocessing import normalize
from liblinearutil import train

def time_varying_mean(data):
    frames = data.shape[0]
    cumulative_sum = np.cumsum(data, axis=0)
    mean = cumulative_sum/np.arange(1.0, frames+1)[:, np.newaxis]
    return mean


def non_linearity(data):
    return np.sign(data)*np.sqrt(np.abs(data))


def rank_pooling(data):
    #mean = time_varying_mean(data)
    #non_linear_mean = non_linearity(mean)
    normalized_data = normalize(data, axis=1, norm='l2')

    total_frames = normalized_data.shape[0]
    labels = list(range(1, total_frames+1))
    data = normalized_data.tolist()
    model = train(labels, data,'-s 11 -q')
    return np.array(model.get_decfun()[0])



def rank_pooling_dataset(data_list):
    return np.array([rank_pooling(data) for data in data_list])