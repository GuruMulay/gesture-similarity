__author__ = 'Prady'
import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, '../')

def PCA(matrix):
    rows = matrix.shape[0]
    cols = matrix.shape[1]

    if rows>cols:
        # print "Snapshot Method"
        u,s,v = svd(np.dot(matrix.T,matrix))
        u = normalize(np.dot(matrix,u),axis=0)
    else:
        # print "Normal Method"
        u,s,v = svd(np.dot(matrix,matrix.T))
    return u,s

# added for a single video
def pca_video_eigvecs_single(video, dims=5):
    return [PCA(video.T)[0][:,:dims]]
# ----------------------------------------------------

def pca_video_eigvecs(data, dims=5):
    return [PCA(video.T)[0][:,:dims] for video in data]


def pca_dataset_eigvecs(data_list):
    return [pca_video_eigvecs(g) for g in data_list]


def pca_video_eigvals(data, dims=5):
    return [PCA(video.T)[1][:dims] for video in data]


def pca_dataset_eigvals(data_list):
    return [pca_video_eigvals(g) for g in data_list]