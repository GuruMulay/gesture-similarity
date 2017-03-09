import numpy as np
np.set_printoptions(precision=4)

from dataset.Data import get_eigenvalue_data, get_eigenvector_data, process_video_get_eigenvectors
from callib.Projection import get_index_combinations, get_index_combinations_a_equals_b
from core.PCA import PCA
import os
from dataset.group_data import write_matrix, write_matrix_2d
from participantAnalysis.participant_analysis import get_p_to_same_p_PA
from participantAnalysis.p_mat_mean_sd import get_col_mean_sd
from participantAnalysis.p_mat_col_gaussian import gaussian_of_mean_pa_per_p
from dataset.matrix_plot import matrix_heatmap

# clustering
from clustering.hca import hca_clustering
from scipy.spatial.distance import pdist

import seaborn as sns
import matplotlib as plt
sns.set(context="paper", font="monospace")

home = os.path.expanduser('~');
data_dir = os.path.join(home, 'gesture_data/')
g_list_n = os.listdir(os.path.join(data_dir,'without sound/'))
print "without sound dir list --->", os.listdir(os.path.join(data_dir,'without sound/'))
print "with sound dir list --->", os.listdir(os.path.join(data_dir,'with sound/'))
#        [1  2  3  4  5  6  7  8  9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40]
# mat0 graph indexing below ==================
#        [0  1  2  3  4  5  6  7  8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39]
p_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 31, 32, 35, 36, 39, 40, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 33, 34, 37, 38, 41, 42]

# [eigvecs_with_sound_p22, eigvecs_with_sound_p30, eigvecs_without_sound_p5, eigvecs_without_sound_p8] = get_eigenvector_data()
eigen_vector_p_number, eigen_vector_array = get_eigenvector_data()
# print "empty gesture array ", eigen_vector_array[0][4]  # 0th person, 4th gesture

if eigen_vector_array[0][4]  == []:
    print "empty !!!!! "
# eigvecs_with_sound_p22[4]
# print len(eigen_vector_array), eigen_vector_p_number

eigen_value_p_number, eigen_value_array = get_eigenvalue_data()

# print "len(eigen_vector_array[0])", len(eigen_vector_array[0]), len(eigvals_with_sound_p22[0][0])
# print "len(eigval_with_sound_p30)[][]", len(eigvals_with_sound_p30), len(eigvals_with_sound_p30[0][0])
# print "len(eigvec_with_sound)[][]", len(eigvecs_with_sound_p22), len(eigvecs_with_sound_p22[0]), eigvecs_with_sound_p22[0][0].shape

# def p_to_p_similarity():

def p_to_p_similarity_matrix():

    p_matrix = np.zeros((5, 40, 40))
    # print p_matrix[0][:][:]
    gcount = 0;
    iter = 0;

    for i in range(len(eigen_vector_array[0])):
        gcount += 1
        print "gcount ------------------------------- ", gcount
        # generate the indeices and then the videos to get the projections
        # For SS and NN, the combinations will give us nC2 while for SN we get mXn
        # Each entry will be a tuple with each entry of size (51,5)

        # eigen_vector_p_number[1 to 4] = [22, 30, 5, 8]
        for pni in range(len(eigen_vector_p_number)):
            # print "pni", pni
            # print eigen_vector_p_number[pni]
            for pnj in range(pni, len(eigen_vector_p_number)): # pni + 1 => no self comparison
                iter += 1
                # print "pnj", pnj
                # print "pni, index of pni", eigen_vector_p_number[pni], p_list.index(eigen_vector_p_number[pni])
                # print "pnj, index of pnj", eigen_vector_p_number[pnj], p_list.index(eigen_vector_p_number[pnj])

                # if sum(sum(sum(chain(*eigen_vector_array[pni][i])))) == 0:
                if eigen_vector_array[pni][i] == [] or eigen_vector_array[pnj][i] == []:  # pni-th person, i-th gesture
                    # assigning -1 to non-existing gestures
                    p_matrix[i][p_list.index(eigen_vector_p_number[pni])][p_list.index(eigen_vector_p_number[pnj])] = -0.0001
                    p_matrix[i][p_list.index(eigen_vector_p_number[pnj])][p_list.index(eigen_vector_p_number[pni])] = -0.0001

                else:
                    # if pni == pnj and pni == 15:
                    #     write_matrix(eigen_vector_array[pni][i], "eigen_v_arr.txt")
                    #     print "pni -------------", pni

                    # if pni == 2:
                        # print eigen_vector_array[pni][2]
                        # write_matrix(eigen_vector_array[pni][2], "eigen_v_p2_g2.txt")

                    if pni == pnj:
                        # print len(eigen_vector_array[pni][][i])
                        # comment if you want to assign zero values along the diagonal => self-PA is zero
                        pp = get_index_combinations_a_equals_b(eigen_vector_array[pni][i], eigen_vector_array[pnj][i])

                    else:
                        pp = get_index_combinations(eigen_vector_array[pni][i], eigen_vector_array[pnj][i])

                    pp_angles = [np.degrees(np.arccos(PCA(np.dot(p[0].T, p[1]))[1][0])) for p in pp]

                    # print "participants: ", eigen_vector_p_number[pni], eigen_vector_p_number[pnj]
                    # print 'PP: ', len(pp_angles), pp_angles[0]
                    # print "PP mean, median, and sd ", np.mean(pp_angles), np.median(pp_angles), np.std(pp_angles)
                    p_matrix[i][p_list.index(eigen_vector_p_number[pni])][
                        p_list.index(eigen_vector_p_number[pnj])] = np.mean(pp_angles)
                    p_matrix[i][p_list.index(eigen_vector_p_number[pnj])][
                        p_list.index(eigen_vector_p_number[pni])] = np.mean(pp_angles)

                    # ss = get_index_combinations(eigen_vector_array[0][i], eigen_vector_array[1][i])
                    # sn1 = get_index_combinations(eigen_vector_array[0][i], eigen_vector_array[2][i])
                    # sn2 = get_index_combinations(eigen_vector_array[1][i], eigen_vector_array[3][i])
                    # sn3 = get_index_combinations(eigen_vector_array[0][i], eigen_vector_array[3][i])
                    # sn4 = get_index_combinations(eigen_vector_array[1][i], eigen_vector_array[2][i])
                    # nn = get_index_combinations(eigen_vector_array[2][i], eigen_vector_array[3][i])
                    #
                    # # if count == 1 :
                    #     # print "ss, sn, nn -> ", ss, sn, nn
                    # #dot product of (51,5) and (5,51) gives (5,5), taking the PCA gives eigenvector of size (5,5)
                    # #and eigenvalues 5. we only consider the first eigenvalue. Find its cosine, and convert to
                    # #degree measure.
                    # ss_angles = [np.degrees(np.arccos(PCA(np.dot(s[0].T, s[1]))[1][0])) for s in ss]
                    # sn1_angles = [np.degrees(np.arccos(PCA(np.dot(s[0].T, s[1]))[1][0])) for s in sn1]
                    # sn2_angles = [np.degrees(np.arccos(PCA(np.dot(s[0].T, s[1]))[1][0])) for s in sn2]
                    # sn3_angles = [np.degrees(np.arccos(PCA(np.dot(s[0].T, s[1]))[1][0])) for s in sn3]
                    # sn4_angles = [np.degrees(np.arccos(PCA(np.dot(s[0].T, s[1]))[1][0])) for s in sn4]
                    # nn_angles = [np.degrees(np.arccos(PCA(np.dot(s[0].T, s[1]))[1][0])) for s in nn]
                    #
                    # print 'length of angles arrays:'
                    # print 'SS:  ', len(ss_angles), ss_angles[0]
                    # print 'SN1: ', len(sn1_angles), sn1_angles[0]
                    # print 'SN2: ', len(sn2_angles), sn2_angles[0]
                    # print 'SN3: ', len(sn3_angles), sn3_angles[0]
                    # print 'SN4: ', len(sn4_angles), sn4_angles[0]
                    # print 'NN:  ', len(nn_angles), nn_angles[0]
                    #
                    # print "SS  mean, median, and sd", np.mean(ss_angles), np.median(ss_angles), np.std(ss_angles)
                    # print "SN1 mean, median, and sd", np.mean(sn1_angles), np.median(sn1_angles), np.std(sn1_angles)
                    # print "SN2 mean, median, and sd", np.mean(sn2_angles), np.median(sn2_angles), np.std(sn2_angles)
                    # print "SN3 mean, median, and sd", np.mean(sn3_angles), np.median(sn3_angles), np.std(sn3_angles)
                    # print "SN4 mean, median, and sd", np.mean(sn4_angles), np.median(sn4_angles), np.std(sn4_angles)
                    # print "NN  mean, median, and sd", np.mean(nn_angles), np.median(nn_angles), np.std(nn_angles)

    print "iter", iter

    # print p_matrix[0][:][:]
    # ========================================================= Choose which matrix SS20 NN20 or ALL40
    p_matrix = np.nan_to_num(p_matrix) # replaces all the NANs with zeros? (here, it's only diagonal nans) NAN occur when the number of samples is just one and we are calculation self PA and not cross PA
    # p_matrix_nn = p_matrix[:][0:19][0:19]
    p_matrix_nn = p_matrix[:, :20, :20]
    p_matrix_ss = p_matrix[:, 20:, 20:]

    p_matrix_final = p_matrix # p_matrix
    # ======================================================================================================

    write_matrix(p_matrix_final, "p_mat_same_pp_noNan_40.txt")
    print p_matrix_final[0][:][:]

    # gives PA between same Ps for nC2 on different example of the same P
    diagonal_mean, diagonal_sd, non_diagonal_mean, non_diagonal_sd = get_p_to_same_p_PA(p_matrix_final, g_list=g_list_n)
    print "diagonal_mean", diagonal_mean
    print "diagonal_sd", diagonal_sd
    print "non_diagonal_mean", non_diagonal_mean
    print "non_diagonal_sd", non_diagonal_sd

    # find column-wise mean and sd for all 40 Ps
    col_mean = np.zeros((len(g_list_n), len(p_matrix_final[1])))
    col_sd = np.zeros((len(g_list_n), len(p_matrix_final[1])))
    col_mean, col_sd = get_col_mean_sd(p_matrix_final, g_list_n, col_mean, col_sd)

    # find the gaussian distribution for 40 means (mean of every column PA)
    gaussian_of_mean_pa_per_p(col_mean, col_sd, g_list_n, p_list)



    # p_to_p_similarity()
    ## ax = sns.heatmap(p_matrix_final[1][:][:])
    ## matrix_heatmap(p_matrix_final[1][:][:], 0, 50)

    ax = sns.heatmap(p_matrix_final[4][:][:], vmin=0.0001, cmap="YlGnBu", square=True)
    sns.plt.show()

eigen_value_p_number = np.array(eigen_value_p_number)

def get_distance_matrix(X):
    y = np.zeros((len(X), len(X)))
    print "y shape = ", y.shape

    for i, v1 in enumerate(X):
        # if i<3:
        #     print i, "i----------------------", X[i,:]
        for j in range(i, len(X)):
            # if i<3:
            #     print j, "j----------------", X[j,:]
            angle = np.degrees(np.arccos(np.minimum(1, np.dot(X[i,:], X[j,:].T)) ))
            y[i,j] = 360-angle if angle>180 else angle
            y[j,i] = y[i,j]


    # write_matrix_2d(y, "distance_mat_50_angles180.txt")
    # ax = sns.heatmap(y, cmap="YlGnBu", square=True)
    # sns.plt.show()

    return y

def get_eigen_vector_matrix(gesture):
    data_dir_with_sound = os.path.join(home, 'gesture_data_original/with sound/')
    data_dir_without_sound = os.path.join(home, 'gesture_data_original/without sound/')

    n = 0
    x = []
    video_list = []

    print "gesture is ----------------------- ", gesture
    for f in os.listdir(os.path.join(data_dir_with_sound, gesture)):
        print "n & file name = ", n, f
        n += 1
        x.append(process_video_get_eigenvectors(os.path.join(data_dir_with_sound, gesture, f)))
        video_list.append(f)

    print "n gestures with sound", n

    print "gesture is ----------------------- ", gesture
    for f in os.listdir(os.path.join(data_dir_without_sound, gesture)):
        # n_frames = sum(1 for line in open(os.path.join(data_dir_without_sound, gesture, f)))
        print "n & file name = ", n, f
        n += 1
        x.append(process_video_get_eigenvectors(os.path.join(data_dir_without_sound, gesture, f)))
        video_list.append(f)

    print "total gestures ", n

    return x, video_list


def cluster_eigen_vectors_values():
    gcount = 0;
    iter = 0;
    X = []

    # for i in [2]: # range(len(eigen_vector_array[0]) - 4):  # for every gesture
    gesture_list = os.listdir(os.path.join(data_dir, 'without sound/'))

    ############# change i to change plots gestures ######################
    i = 4  # ['head nod', 'hands into claw down', 'LA move down', 'RA move down', 'arms move down']

    gcount += 1
    print "gcount ------------------------------- ", gcount

    print "eigen_vector_p_number", eigen_vector_p_number

    for pni in range(len(eigen_vector_p_number)):  # for every participant (40)
        print "pni is ", eigen_vector_p_number[pni]
        eigen_vector_array_pni = eigen_vector_array[pni][i]

        if eigen_vector_p_number[pni] == 12:
            print "eigen_vector_array_pni for 12 ", eigen_vector_array_pni

        print len(eigen_vector_array_pni)
        # if len(eigen_vector_array_pni) == 1:
            # continue
            # print eigen_vector_array_pni
            # print eigen_vector_array_pni[:][0]
        # else:
        eigen_vector_array_pni = np.array(eigen_vector_array_pni)

        # print "len(eigen_vector_array_pni)", len(eigen_vector_array_pni)
        # print "pni shape", eigen_vector_array_pni.shape

        if eigen_vector_array[pni][i] != []:  # if not null
            for video in range(eigen_vector_array_pni.shape[0]):  # for every video by participant pni
                # print "1st eigen vecs for video number", video, "-->", eigen_vector_array_pni[video][:, 0]
                X.append(eigen_vector_array_pni[video][:, 0])

        # print "eigen vects", eigen_vector_p_number[pni], "=====", eigen_vector_array[pni][i]
        print "eigen vals:::::", eigen_value_array[np.where(eigen_value_p_number == eigen_vector_p_number[pni])[0][0]][i]

    print "eigen_value_p_number", eigen_value_p_number
    for pni in range(len(eigen_value_p_number)):
        print "eigen vals", eigen_value_p_number[pni], "=====", eigen_value_array[pni][i]

    X = np.array(X)
    print "X shape ", X.shape
    # print X
    write_matrix_2d(X, "p_data_x_amd")

    # getting eigen vectors from non-p-divided data (gesture_data_original)
    X, video_list = get_eigen_vector_matrix(gesture_list[i])
    X = np.array(X)
    write_matrix_2d(X, "original_data_x_amd")

    # truncated for testing
    X_trunc = X[0:50,:]
    print "X_trunc shape ", X_trunc.shape


    # X_dist is a diagonally symmetric matrix of the principal angle between two 1st eigen vectors
    X_dist = get_distance_matrix(X)
    # print "X_dist", X_dist
    print "X_dist shape ", X_dist.shape

    hca_clustering(X_dist, plot_title=gesture_list[i])

    print "video list --------", len(video_list), video_list



# plot the matrix showing p to p similarity matrix
# p_to_p_similarity_matrix()

# print eigen vecs and vals
cluster_eigen_vectors_values()

# X1 = get_eigen_vector_matrix("arms move down")
# X1 = np.array(X1)
# print X1.shape
