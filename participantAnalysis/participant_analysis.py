import numpy as np


def get_p_to_same_p_PA(matrix3d, g_list):
    print g_list
    diagonal_mean = []
    diagonal_sd = []
    non_diagonal_mean = []
    non_diagonal_sd = []

    len_g_list = len(g_list)
    mat_reshaped = matrix3d.reshape(len_g_list, len(matrix3d[1])*len(matrix3d[1]) )

    for i in range(len(g_list)):
        print "diagonal elements", g_list[i], ["%.2f" % fl for fl in matrix3d[i][:][:].diagonal()]

        diag_array = [fl for fl in matrix3d[i][:][:].diagonal()]
        diag_array = np.array(diag_array)
        diagonal_mean.append( np.true_divide(diag_array.sum(0) , (diag_array > 0).sum(0) ) )
        diagonal_sd.append( np.apply_along_axis(lambda v: np.std(v[v > 0]), 0, diag_array) )

        non_diagonal_mean.append( np.true_divide( matrix3d[i][:][:].sum() - diag_array.sum(0), (matrix3d[i][:][:] > 0).sum() - (diag_array > 0).sum(0) ) )

        # removes diagonal elements and zero elements
        itemsToRemove = set(diag_array)
        sd_array = mat_reshaped[i]
        sd_array_1 = filter(lambda x: x not in itemsToRemove, sd_array)
        print "sd_array_1", len(sd_array_1)
        non_diagonal_sd.append( np.apply_along_axis(lambda v: np.std(v[v > 0]), 0, sd_array_1) )

    return diagonal_mean, diagonal_sd, non_diagonal_mean, non_diagonal_sd

