import numpy as np
import os
from core.Normalization import normalize_spine_base_dataset, normalize_spine_base
from core.PCA import pca_dataset_eigvals, pca_dataset_eigvecs, pca_video_eigvecs_single
from features.Velocity import calculate_velocity_dataset, calculate_for_video_single
from features.Acceleration import calculate_acceleration_dataset

from itertools import chain


home = os.path.expanduser('~');
print home, " <- home directory's path for user "

data_dir_with_sound = os.path.join(home, 'gesture_data/with sound/')
data_dir_without_sound = os.path.join(home, 'gesture_data/without sound/')
des_dir_ns = os.path.join(home, 'gesture_data/all_npy/ns/')
des_dir_eigen = os.path.join(home, 'gesture_data/all_npy/eigen/')
des_dir_velocity = os.path.join(home, 'gesture_data/all_npy/velocity/')
des_dir_acceleration = os.path.join(home, 'gesture_data/all_npy/acceleration/')

print data_dir_with_sound, data_dir_without_sound, des_dir_eigen, des_dir_velocity

p5 = "p5/"
p8 = "p8/"

p22 = "p22/"
p30 = "p30/"

p_list_with_sound = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 33, 34, 37, 38, 41, 42]
p_list_without_sound = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 31, 32, 35, 36, 39, 40]

# for testing 1-1-1-1 or 10-10-10-10 cases
# p_list_with_sound = [22, 30]
# p_list_without_sound = [5, 8]



def load_skeleton_data(filename):
    data = np.loadtxt(filename, dtype='float', delimiter=', ', skiprows=1, usecols=(9, 10, 11, 18, 19, 20, 27, 28, 29, 36, 37, 38, 45, 46, 47, 54, 55, 56, 63, 64, 65, 72, 73, 74, \
                                                                                    81, 82, 83, 90, 91, 92, 99, 100, 101, 108, 109, 110, 189, 190, 191, 198,199,200,\
                                                                                    207,208,209,216,217,218,225,226,227))
    return data


# removes videos for which n_frames < 9 because velocity calculation removes 4 frames => 9-4 = 5 (5 required for PCA)
def remove_small_videos():
    for p in p_list_with_sound:
        print "person is =========================================", p

        for gesture in os.listdir(data_dir_with_sound):
            print "gesture is ----------------------- ", gesture
            for f in os.listdir(os.path.join(data_dir_with_sound, gesture, ("p" + str(p)))):
                n_frames = sum(1 for line in open(os.path.join(data_dir_with_sound, gesture, ("p" + str(p)), f)))
                # print "file name and n_frames = ", f, n_frames
                if n_frames - 1 < 9:
                    print "file name and n_frames = ", f, n_frames
                    os.remove(os.path.join(data_dir_with_sound, gesture, ("p" + str(p)), f))

    for p in p_list_without_sound:
        print "person is =========================================", p

        for gesture in os.listdir(data_dir_without_sound):
            print "gesture is ----------------------- ", gesture
            for f in os.listdir(os.path.join(data_dir_without_sound, gesture, ("p" + str(p)))):
                n_frames = sum(1 for line in open(os.path.join(data_dir_without_sound, gesture, ("p" + str(p)), f)))
                if n_frames - 1 < 9:
                    print "file name and n_frames = ", f, n_frames
                    os.remove(os.path.join(data_dir_without_sound, gesture, ("p" + str(p)), f))


def data_initialization():

    for p in p_list_with_sound:
        print p
        with_sound_data_p = [[load_skeleton_data(os.path.join((data_dir_with_sound + gesture), ("p" + str(p)), f)) for f in
                            os.listdir(os.path.join(data_dir_with_sound, gesture, ("p" + str(p))))] for gesture in
                            os.listdir(data_dir_with_sound)]

        np.save((des_dir_ns + 'with_sound_p' + str(p) + '.npy'), with_sound_data_p)
        # print len(with_sound_data_p22)
        # print len(with_sound_data_p22[0])
        # print len(with_sound_data_p22[1])
        # print len(with_sound_data_p22[0][0])
        # print with_sound_data_p22[0][0].shape
        # print with_sound_data_p22[4] # , sum(with_sound_data_p22[4])

    for p in p_list_without_sound:
        print p
        without_sound_data_p = [[load_skeleton_data(os.path.join((data_dir_without_sound + gesture), ("p" + str(p)), f)) for f in
                                  os.listdir(os.path.join(data_dir_without_sound, gesture, ("p" + str(p))))] for gesture in
                                 os.listdir(data_dir_without_sound)]

        np.save((des_dir_ns + 'without_sound_p' + str(p) + '.npy'), without_sound_data_p)

    # with_sound_data_p30 = [[load_skeleton_data(os.path.join((data_dir_with_sound + gesture), p30, f)) for f in
    #                     os.listdir(os.path.join(data_dir_with_sound, gesture, p30))] for gesture in
    #                     os.listdir(data_dir_with_sound)]

    # print len(with_sound_data), len(with_sound_data[0]), with_sound_data[0][0].shape
    # print " path p5 for loading ", os.path.join((data_dir_without_sound + gesture), p5, f)
    # print " path p5 for f in ", os.path.join(data_dir_without_sound, gesture, p5)

    # without_sound_data_p8 = [[load_skeleton_data(os.path.join((data_dir_without_sound + gesture), p8, f)) for f in
    #                        os.listdir(os.path.join(data_dir_without_sound, gesture, p8))] for gesture in
    #                        os.listdir(data_dir_without_sound)]

    # print len(without_sound_data), len(without_sound_data[0]), without_sound_data[0][0].shape

    # np.save((des_dir + 'with_sound_p30.npy'), with_sound_data_p30)
    # np.save((des_dir + 'without_sound_p8.npy'), without_sound_data_p8)

    # a = np.load(des_dir+'with_sound.npy')
    # b = np.load(des_dir+'without_sound.npy')
    # print len(a), len(b), len(a[0]), len(b[0]), a[0][0].shape, b[0][0].shape

def print_npy_data_sanity_check():
    data_participant = get_data()
    data_participant = np.array(data_participant)
    print "data_participant.shape", data_participant.shape
    # data_participant shape: [[p1], [p2], ... , p[40]]
    # [pi] = [[g1], [g2], ... , g[5]]
    # [gi] i.e., d  = []
    print "len(data_participant)", len(data_participant)
    p = 0  # => p12

    for p in range(0, 39) : #, p_list_with_sound]:
        # print os.listdir(os.path.join(des_dir_ns, os.listdir(des_dir_ns)[p]) )
        print "person is os.listdir(des_dir_ns)[p]", os.listdir(des_dir_ns)[p]

        i = 0
        for d in data_participant[p]:
            # if i==4:
            print "ith gesture, i ==================================================================", i
            # print "os.listdir(des_dir_ns)[p] and data p", data_participant[p]
            # print "data d from data_participant", d
            d = np.array(d)
            print "d shape ", d.shape  # number of videos for gesture i
            n = d.shape[0] - 1  # n number of videos in ith gesture
            # sanity check! confirmed that every row (frame) has a 51 elements => till pca everything is fine
            for ni in range(0, n+1):  # for every video for gi for pi
                # print "d[n] ", len(d[n])  # number of frames in a video
                if d[n].size / len(d[n]) != 51:
                    print "ERROR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"

            i += 1


    print os.listdir(data_dir_without_sound)
    gesture_list_p_armsmovedown = os.listdir(os.path.join(data_dir_without_sound, "arms move down", os.listdir(des_dir_ns)[p][-7:-4]))
    print os.listdir(os.path.join(data_dir_without_sound, "arms move down", os.listdir(des_dir_ns)[p][-7:-4]))
    print " len(gesture_list_p_armsmovedown)", len(gesture_list_p_armsmovedown)

    data_participant = np.array(data_participant)
    print "data participant list shape", data_participant.shape  # 40 p and 5 gestures


def print_spine_norm_data_sanity_check():
    data_participant = get_data()
    print "len(data_participant)", len(data_participant)

    # data_participant shape: [[p1], [p2], ... , p[40]] (40, 5)
    # [pi] = [[g1], [g2], ... , g[5]] for d in data_participant[p] => d is g1 g2 g3 g4 g5
    # [gi] i.e., d  = []

    # spine normalization
    for p in range(len(data_participant)):
        data_p_norm = [normalize_spine_base_dataset(d) for d in data_participant[p]]

        # VELOCITY INSERTED !
        # --------------------------------------------------------------------------------------
        data_p_vel = calculate_velocity_dataset(data_p_norm)

        # Acceleration INSERTED !
        # --------------------------------------------------------------------------------------
        data_p_acc = calculate_acceleration_dataset(data_p_norm)

        data_p_check = data_p_vel #############################################################################

        if p!=100:
            data_norm = np.array(data_p_check)
            print "data_p_check.shape", data_norm.shape

            # data_norm shape: [[g1], [g2], ... , g[5]] (5, )
            # [gi] = (n_gestures, )
            for l in range(len(data_norm)):  # lth gestures
                print "",  # data_norm[l]
                data_norm[l] = np.array(data_norm[l])
                print "l = ", l, data_norm[l].shape

                for gi in range(0, data_norm[l].shape[0]):
                    # print data_norm[l][gi][:].size/len(data_norm[l][gi][:])
                    if data_norm[l][gi][:].size/len(data_norm[l][gi][:]) != 51:
                        print "Error!--------------------------------------------------"


    pn=0
    print os.listdir(data_dir_without_sound)
    pi = os.listdir(des_dir_ns)[pn][-7:-4]
    print pi
    gesture_list_p_armsmovedown = os.listdir(os.path.join(data_dir_without_sound, "arms move down", pi ))
    print os.path.join(data_dir_without_sound, "arms move down", os.listdir(des_dir_ns)[pn][-7:-4])
    print os.listdir(os.path.join(data_dir_without_sound, "arms move down", os.listdir(des_dir_ns)[pn][-7:-4]))
    print " len(gesture_list_p_armsmovedown)", len(gesture_list_p_armsmovedown)

    data_participant = np.array(data_participant)
    print "data participant list shape", data_participant.shape  # 40 p and 5 gestures


def print_pca_data_sanity_check():
    data_participant = get_data()
    print "len(data_participant)", len(data_participant)

    # data_participant shape: [[p1], [p2], ... , p[40]] (40, 5)
    # [pi] = [[g1], [g2], ... , g[5]] for d in data_participant[p] => d is g1 g2 g3 g4 g5
    # [gi] i.e., d  = [[[51 vals], [], []... number of frames], [], []... number of videos]

    # spine normalization
    for p in range(len(data_participant)):
        data_p_norm = [normalize_spine_base_dataset(d) for d in data_participant[p]]

        # VELOCITY INSERTED !
        # --------------------------------------------------------------------------------------
        data_p_vel = calculate_velocity_dataset(data_p_norm)

        # Acceleration INSERTED !
        # --------------------------------------------------------------------------------------
        data_p_acc = calculate_acceleration_dataset(data_p_norm)

        data_p_check = data_p_vel #############################################################################

        eigvecs_p = pca_dataset_eigvecs(data_p_vel)
        eigvals_p = pca_dataset_eigvals(data_p_vel)

        if p==0:
            # eigvecs_p = np.array(eigvecs_p)
            # print "eigen vect p: ", eigvecs_p.shape  # (5, )
            # in each of 5, we have [[5 elements], [], ... 51 of these]
            for l in range(len(eigvecs_p)):  # lth gestures
                # eigvecs_p[l] = np.array(eigvecs_p[l])
                # print "l = ", l, eigvecs_p[l].shape
                print "eigen vect---", l, eigvecs_p[l]

        if p==0:
            data_norm = np.array(data_p_check)
            print "data_p_check.shape", data_norm.shape

            # data_norm shape: [[g1], [g2], ... , g[5]] (5, )
            # [gi] = (n_gestures, )
            for l in range(len(data_norm)):  # lth gestures
                print "",  # data_norm[l]
                data_norm[l] = np.array(data_norm[l])
                print "l = ", l, data_norm[l].shape

                for gi in range(0, data_norm[l].shape[0]):
                    # print data_norm[l][gi][:].size/len(data_norm[l][gi][:])
                    if data_norm[l][gi][:].size/len(data_norm[l][gi][:]) != 51:
                        print "Error!--------------------------------------------------"


    pn=0
    print os.listdir(data_dir_without_sound)
    pi = os.listdir(des_dir_ns)[pn][-7:-4]
    print pi
    gesture_list_p_armsmovedown = os.listdir(os.path.join(data_dir_without_sound, "arms move down", pi ))
    print os.path.join(data_dir_without_sound, "arms move down", os.listdir(des_dir_ns)[pn][-7:-4])
    print os.listdir(os.path.join(data_dir_without_sound, "arms move down", os.listdir(des_dir_ns)[pn][-7:-4]))
    print " len(gesture_list_p_armsmovedown)", len(gesture_list_p_armsmovedown)

    data_participant = np.array(data_participant)
    print "data participant list shape", data_participant.shape  # 40 p and 5 gestures



def eigenvalue_eigenvector_data_initialization():
    # data_with_sound_p22, data_with_sound_p30, data_without_sound_p5, data_without_sound_p8
    # data_with_sound_p22, data_with_sound_p30, data_without_sound_p5, data_without_sound_p8 = get_data()
    data_participant = get_data()
    print "len(data_participant)", len(data_participant)
    print "in init", os.listdir(des_dir_ns)

    for p in range(len(data_participant)):
        data_p_norm = [normalize_spine_base_dataset(d) for d in data_participant[p]]

        # VELOCITY INSERTED !
        # --------------------------------------------------------------------------------------
        data_p_vel = calculate_velocity_dataset(data_p_norm)
        np.save((des_dir_velocity + 'velocity_' + os.listdir(des_dir_ns)[p]), data_p_vel)
        # np.save((des_dir_velocity + 'eigenvalue_' + os.listdir(des_dir_ns)[p]), eigvals_p)
        # --------------------------------------------------------------------------------------

        # Acceleration INSERTED !
        # --------------------------------------------------------------------------------------
        data_p_acc = calculate_acceleration_dataset(data_p_norm)
        np.save((des_dir_acceleration + 'acceleration_' + os.listdir(des_dir_ns)[p]), data_p_acc)
        # --------------------------------------------------------------------------------------

        eigvecs_p = pca_dataset_eigvecs(data_p_vel)
        eigvals_p = pca_dataset_eigvals(data_p_vel)
        np.save((des_dir_eigen + 'eigenvector_' + os.listdir(des_dir_ns)[p]), eigvecs_p)
        np.save((des_dir_eigen + 'eigenvalue_' + os.listdir(des_dir_ns)[p]), eigvals_p)

    # # spine base normalize the data
    # data_with_sound_p22 = [normalize_spine_base_dataset(d) for d in data_participant[0]]
    # data_with_sound_p30 = [normalize_spine_base_dataset(d) for d in data_participant[1]]
    # data_without_sound_p8 = [normalize_spine_base_dataset(d) for d in data_participant[2]]
    # data_without_sound_p5 = [normalize_spine_base_dataset(d) for d in data_participant[3]]
    #
    # print "after norm -----------------"
    # print len(data_with_sound_p22)
    # print len(data_with_sound_p22[0])
    # print len(data_with_sound_p22[1])
    # print len(data_with_sound_p22[0][0])
    # print data_with_sound_p22[0][0].shape
    # print data_with_sound_p22[4]
    #
    # # calculate pca to get eigenvectors and eigenvalues of dimensions (51,5) for eigvecs and 5 for eigvals
    # # with sound
    # eigvecs_with_sound_p22 = pca_dataset_eigvecs(data_with_sound_p22)
    # eigvals_with_sound_p22 = pca_dataset_eigvals(data_with_sound_p22)
    # eigvecs_with_sound_p30 = pca_dataset_eigvecs(data_with_sound_p30)
    # eigvals_with_sound_p30 = pca_dataset_eigvals(data_with_sound_p30)
    #
    # print "after pca -----------------"
    # print len(eigvecs_with_sound_p22)
    # print len(eigvecs_with_sound_p22[0])
    # print len(eigvecs_with_sound_p22[1])
    # print len(eigvecs_with_sound_p22[0][0])
    # print eigvecs_with_sound_p22[0][0].shape
    # print eigvecs_with_sound_p22[4]
    #
    # # without sound
    # eigvecs_without_sound_p5 = pca_dataset_eigvecs(data_without_sound_p5)
    # eigvals_without_sound_p5 = pca_dataset_eigvals(data_without_sound_p5)
    # eigvecs_without_sound_p8 = pca_dataset_eigvecs(data_without_sound_p8)
    # eigvals_without_sound_p8 = pca_dataset_eigvals(data_without_sound_p8)
    #
    # #print len(eigvecs_with_sound), eigvecs_with_sound[0][0].shape, len(eigvals_with_sound[0][0])
    # #print len(eigvecs_without_sound), eigvecs_without_sound[0][0].shape, len(eigvals_without_sound[0][0])
    #
    # # saving the eigenvectors and eigenvalues to file for further references
    # # with sound
    # np.save((des_dir_eigen + 'eigenvector_with_sound_p22.npy'), eigvecs_with_sound_p22)
    # np.save((des_dir_eigen + 'eigenvalue_with_sound_p22.npy'), eigvals_with_sound_p22)
    # np.save((des_dir_eigen + 'eigenvector_with_sound_p30.npy'), eigvecs_with_sound_p30)
    # np.save((des_dir_eigen + 'eigenvalue_with_sound_p30.npy'), eigvals_with_sound_p30)
    #
    # # without sound
    # np.save((des_dir_eigen + 'eigenvector_without_sound_p5.npy'), eigvecs_without_sound_p5)
    # np.save((des_dir_eigen + 'eigenvalue_without_sound_p5.npy'), eigvals_without_sound_p5)
    # np.save((des_dir_eigen + 'eigenvector_without_sound_p8.npy'), eigvecs_without_sound_p8)
    # np.save((des_dir_eigen + 'eigenvalue_without_sound_p8.npy'), eigvals_without_sound_p8)


def get_data():
    # Load the data from the directory
    # Data format is : [[[]], [[]]] [category[video]]
    # data_with_sound_p22 = np.load(open(os.path.join(des_dir_ns, "with_sound_p22.npy"), 'rb'))
    # data_with_sound_p30 = np.load(open(os.path.join(des_dir_ns, "with_sound_p30.npy"), 'rb'))
    # data_without_sound_p5 = np.load(open(os.path.join(des_dir_ns, "without_sound_p5.npy"), 'rb'))
    # data_without_sound_p8 = np.load(open(os.path.join(des_dir_ns, "without_sound_p8.npy"), 'rb'))
    print " in get data --->", os.listdir(des_dir_ns)
    # for file in os.listdir(des_dir_ns): print file
    # return data_with_sound_p22, data_with_sound_p30, data_without_sound_p5, data_without_sound_p8
    return [np.load(open(os.path.join(des_dir_ns, file), 'rb')) for file in os.listdir(des_dir_ns)]


def get_eigenvector_data():
    # for p in os.listdir(os.path.join(data_dir_with_sound, gesture)) for gesture in
    #                        os.listdir(data_dir_with_sound)]

    # eigvecs_with_sound_p22 = np.load(open(os.path.join(des_dir_eigen, "eigenvector_with_sound_p22.npy"), 'rb'))
    #
    # print "eigen vects -----------------------"
    # print len(eigvecs_with_sound_p22)
    # print len(eigvecs_with_sound_p22[0])
    # print len(eigvecs_with_sound_p22[1])
    # print len(eigvecs_with_sound_p22[0][0])
    # print eigvecs_with_sound_p22[0][0].shape
    # # print eigvecs_with_sound_p22[4]
    # # print sum(sum(sum(eigvecs_with_sound_p22)))
    # # print len(sum(sum(eigvecs_with_sound_p22[0])))
    # # print sum(sum(sum(chain(*eigvecs_with_sound_p22))))
    #
    # eigvecs_with_sound_p30 = np.load(open(os.path.join(des_dir_eigen, "eigenvector_with_sound_p30.npy"), 'rb'))
    # eigvecs_without_sound_p5 = np.load(open(os.path.join(des_dir_eigen, "eigenvector_without_sound_p5.npy"), 'rb'))
    # eigvecs_without_sound_p8 = np.load(open(os.path.join(des_dir_eigen, "eigenvector_without_sound_p8.npy"), 'rb'))
    # return [22, 30, 5, 8], [eigvecs_with_sound_p22, eigvecs_with_sound_p30, eigvecs_without_sound_p5, eigvecs_without_sound_p8]

    print "print os.listdir(des_dir_ns)---------->", os.listdir(des_dir_ns)
    print "Important ---------->>>>"
    print "get eigvec --->", os.listdir(des_dir_eigen)
    p_number_array = [int(filter(str.isdigit, f)) for f in os.listdir(des_dir_eigen) if f.startswith("eigenvector")]
    print "p_number_array for eigen vectors ", p_number_array

    p_number_array_vals = [int(filter(str.isdigit, f)) for f in os.listdir(des_dir_eigen) if f.startswith("eigenvalue")]
    print "p_number_array for eigen values:::::", p_number_array_vals

    # print "sanity check on p list"
    # for file in os.listdir(des_dir_eigen):
    #     if file.startswith("eigenvector"):
    #         print file
    return p_number_array, [np.load(open(os.path.join(des_dir_eigen, file), 'rb')) for file in os.listdir(des_dir_eigen) if file.startswith("eigenvector")]


def get_eigenvalue_data():
    #5 eigenvalues for every video's PCA are returned.
    # eigvals_with_sound_p22 = np.load(open(os.path.join(des_dir_eigen, "eigenvalue_with_sound_p22.npy"), 'rb'))
    # eigvals_with_sound_p30 = np.load(open(os.path.join(des_dir_eigen, "eigenvalue_with_sound_p30.npy"), 'rb'))
    # eigvals_without_sound_p5 = np.load(open(os.path.join(des_dir_eigen, "eigenvalue_without_sound_p5.npy"), 'rb'))
    # eigvals_without_sound_p8 = np.load(open(os.path.join(des_dir_eigen, "eigenvalue_without_sound_p8.npy"), 'rb'))

    print "print os.listdir(des_dir_ns)---------->", os.listdir(des_dir_ns)
    print "Important ---------->>>>"
    print "get eigvec --->", os.listdir(des_dir_eigen)
    p_number_array = [int(filter(str.isdigit, f)) for f in os.listdir(des_dir_eigen) if f.startswith("eigenvalue")]
    print "p_number_array for eigen values", p_number_array

    return p_number_array, [np.load(open(os.path.join(des_dir_eigen, file), 'rb')) for file in os.listdir(des_dir_eigen) if file.startswith("eigenvalue")]


def test_video_on_pca_methods():

    video_path = "/home/guru/gesture_data/without sound/arms move down/p12/Gesture968_Session 6-Participant 12-Block 1_1507000000_Skeleton.txt"

    data = load_skeleton_data(video_path)
    print data

    data = normalize_spine_base(data)
    print "normalized data", data

    data = calculate_for_video_single(data)
    # data = np.array(data)
    print "velocity data ", data[0]  # to extract the data from [array([[ ]])  ]

    eigvecs_p = pca_video_eigvecs_single(data[0])
    print "eigen vectors of data", eigvecs_p
    eigvecs_p = eigvecs_p[0]
    print "eigen vector 1st", eigvecs_p[:,0]


def process_video_get_eigenvectors(video_path):

    data = load_skeleton_data(video_path)
    # print data

    data = normalize_spine_base(data)
    # print "normalized data", data

    data = calculate_for_video_single(data)
    # data = np.array(data)
    # print "velocity data ", data[0]  # to extract the data from [array([[ ]])  ]

    eigvecs_p = pca_video_eigvecs_single(data[0])
    # print "eigen vectors of data", eigvecs_p
    eigvecs_p = eigvecs_p[0]
    # print "eigen vector 1st", eigvecs_p[:, 0]

    return eigvecs_p[:, 0]


# Comment out when done with data generation!
# remove_small_videos()
print "initializing and generating the eigen data....."
# data_initialization();
# eigenvalue_eigenvector_data_initialization();


# SANITY checks----------------------------------
# printing and verifying numpy array data
# print_npy_data_sanity_check()
# print_spine_norm_data_sanity_check()
# print_pca_data_sanity_check()
# test_video_on_pca_methods()
# -----------------------------------------------

# get eigen vectors and values
# p_arr, eigv_arr = get_eigenvector_data()
# arr1, arr2 = get_eigenvalue_data()

# print len(arr1), len(arr2)
# print "shapes", len(arr2[0][2]), len(arr2[1][2]), len(arr2[2][2]), len(arr2[3][2])