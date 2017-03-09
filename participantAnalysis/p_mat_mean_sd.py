import numpy as np
from numpy import inf


def get_col_mean_sd(matrix3d, g_list, col_mean, col_sd):
    for i in range(len(g_list)):
        col_mean[i][:] = np.true_divide(matrix3d[i][:][:].sum(1) , (matrix3d[i][:][:] > 0).sum(1) ) # if (matrix3d[i][:][:] > 0).sum(1) > 0.1 else 0.1 ) # if else handles -inf problem
        col_mean[i][:][col_mean[i][:] == -inf] = 0. # happens when all elements are zero

        col_sd[i][:] = np.apply_along_axis(lambda v: np.std(v[v > 0]), 0, matrix3d[i][:][:])
        col_sd[i][:][np.isnan(col_sd[i][:])] = 0.;
        # ans

        # Wrong method that also includes the zero elements in the row. These zero elements represent either self PA or 1 sample of the gesture
        # col_mean[i][:] = matrix3d[i][:][:].mean(1)
        # col_sd[i][:] = matrix3d[i][:][:].std(1)


        print "-----------------------------------------"
        print "mean ", g_list[i], ["%.2f" % fl for fl in col_mean[i][:]]
        print "std- ", g_list[i], ["%.2f" % fl for fl in col_sd[i][:]]
        print "-----------------------------------------"

    return col_mean, col_sd