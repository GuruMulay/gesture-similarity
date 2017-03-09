import numpy as np
import os
from core.Normalization import normalize_spine_base_dataset
from core.PCA import pca_dataset_eigvals, pca_dataset_eigvecs

home = os.path.expanduser('~');
print home, " <- home directory's path for user "

data_dir_with_sound = os.path.join(home, 'gesture_data/with sound/')
data_dir_without_sound = os.path.join(home, 'gesture_data/without sound/')
des_dir = os.path.join(home, 'gesture_data/')
print data_dir_with_sound, data_dir_without_sound, des_dir

def load_skeleton_data(filename):
    data = np.loadtxt(filename, dtype='float', delimiter=', ', skiprows=1, usecols=(9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 45, 46, 47, 54, 55, 56, 63, 64, 65, 72, 73, 74, \
                                                                                    81, 82, 83, 90, 91, 92, 99, 100, 101, 108, 109, 110, 189, 190, 191, 198,199,200,\
                                                                                    207,208,209,216,217,218,225,226,227))
    return data


def data_initialization():
    with_sound_data = [[load_skeleton_data(os.path.join((data_dir_with_sound + gesture), f)) for f in
                        os.listdir(os.path.join(data_dir_with_sound, gesture))] for gesture in
                       os.listdir(data_dir_with_sound)]
    # print len(with_sound_data), len(with_sound_data[0]), with_sound_data[0][0].shape

    without_sound_data = [[load_skeleton_data(os.path.join((data_dir_without_sound + gesture), f)) for f in
                           os.listdir(os.path.join(data_dir_without_sound, gesture))] for gesture in
                          os.listdir(data_dir_without_sound)]
    # print len(without_sound_data), len(without_sound_data[0]), without_sound_data[0][0].shape

    np.save((des_dir + 'with_sound.npy'), with_sound_data)
    np.save((des_dir + 'without_sound.npy'), without_sound_data)

    # a = np.load(des_dir+'with_sound.npy')
    # b = np.load(des_dir+'without_sound.npy')
    # print len(a), len(b), len(a[0]), len(b[0]), a[0][0].shape, b[0][0].shape


def eigenvalue_eigenvector_data_initialization():
    data_with_sound, data_without_sound = get_data()
    # spine base normalize the data
    data_with_sound = [normalize_spine_base_dataset(d) for d in data_with_sound]
    data_without_sound = [normalize_spine_base_dataset(d) for d in data_without_sound]

    # calculate pca to get eigenvectors and eigenvalues of dimensions (51,5) for eigvecs and 5 for eigvals
    # with sound
    eigvecs_with_sound = pca_dataset_eigvecs(data_with_sound)
    eigvals_with_sound = pca_dataset_eigvals(data_with_sound)
    # without sound
    eigvecs_without_sound = pca_dataset_eigvecs(data_without_sound)
    eigvals_without_sound = pca_dataset_eigvals(data_without_sound)

    #print len(eigvecs_with_sound), eigvecs_with_sound[0][0].shape, len(eigvals_with_sound[0][0])
    #print len(eigvecs_without_sound), eigvecs_without_sound[0][0].shape, len(eigvals_without_sound[0][0])

    # saving the eigenvectors and eigenvalues to file for further references
    # with sound
    np.save((des_dir + 'eigenvector_with_sound.npy'), eigvecs_with_sound)
    np.save((des_dir + 'eigenvalue_with_sound.npy'), eigvals_with_sound)

    # without sound
    np.save((des_dir + 'eigenvector_without_sound.npy'), eigvecs_without_sound)
    np.save((des_dir + 'eigenvalue_without_sound.npy'), eigvals_without_sound)



def get_data():
    # Load the data from the directory
    # Data format is : [[[]], [[]]] [category[video]]
    data_with_sound = np.load(open(os.path.join(des_dir, "with_sound.npy"), 'rb'))
    data_without_sound = np.load(open(os.path.join(des_dir, "without_sound.npy"), 'rb'))
    return data_with_sound, data_without_sound


def get_eigenvector_data():
    eigvecs_with_sound = np.load(open(os.path.join(des_dir, "eigenvector_with_sound.npy"), 'rb'))
    eigvecs_without_sound = np.load(open(os.path.join(des_dir, "eigenvector_without_sound.npy"), 'rb'))
    return eigvecs_with_sound, eigvecs_without_sound


def get_eigenvalue_data():
    #5 eigenvalues for every video's PCA are returned.
    eigvals_with_sound = np.load(open(os.path.join(des_dir, "eigenvalue_with_sound.npy"), 'rb'))
    eigvals_without_sound = np.load(open(os.path.join(des_dir, "eigenvalue_without_sound.npy"), 'rb'))
    return eigvals_with_sound, eigvals_without_sound
