import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np


def gaussian_of_mean_pa_per_p(col_mean, col_sd, g_list, p_list):
    colours = ['r', 'g', 'b', 'k', 'y']
    # print "col_mean.dtype", col_mean.dtype
    # for i in range(len(g_list)):
        # col_mean[i][:]

    # change this i to change gesture
    i = 4
    print "positive values in col_mean", col_mean[i][col_mean[i][:] > 0]
    col_mean_separated = np.array(col_mean[i][:])
    # print col_mean_separated
    index_array = np.where(col_mean_separated > 0)[0]
    gauss_array = col_mean[i][col_mean[i][:] > 0]
    # print "gauss_array.dtype", gauss_array.dtype

    # print "gauss_array_original", gauss_array
    print "index array", index_array

    gauss_array.sort()
    # print "gauss_array_sorted", gauss_array
    gauss_array_original = col_mean[i][col_mean[i][:] > 0]
    sd_array_original = col_sd[i][col_sd[i][:] > 0]
    print "gauss_array_original", gauss_array_original  # for printing purpose
    print "sd_array_original", sd_array_original  # for printing purpose

    arr_mean = np.mean(gauss_array)
    arr_sd = np.std(gauss_array)
    pdf = stats.norm.pdf(gauss_array, arr_mean, arr_sd)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # # major ticks every 20, minor ticks every 5
    # major_ticks = np.arange(0, 31, 1)
    # minor_ticks = np.arange(0, 31, .5)
    # ax.set_xticks(major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)
    # # and a corresponding grid
    # ax.grid(which='both')
    # ax.grid(which='minor', alpha=0.01)

    # ax.grid(alpha = 0.001, color = 'k')

    plt.title(g_list[i])
    plt.plot(gauss_array, pdf, colours[i])
    plt.xlim(arr_mean - 2.5*arr_sd, arr_mean + 2.5*arr_sd)
    plt.axvline(x=arr_mean, color = 'b')
    plt.axvline(x=arr_mean + arr_sd, color = 'r')
    plt.axvline(x=arr_mean - arr_sd, color = 'r')
    plt.axvline(x=arr_mean + 2*arr_sd, color = 'r')
    plt.axvline(x=arr_mean - 2*arr_sd, color = 'r')
    plt.axvline(x=arr_mean + 3*arr_sd, color = 'r')
    plt.axvline(x=arr_mean - 3*arr_sd, color = 'r')
    # ax = plt.plot()
    plt.plot(gauss_array_original, np.zeros(len(gauss_array_original)), 'r.')
    xerr =0.01
    # plt.errorbar(gauss_array_original, np.resize( [0.0075, 0.005, 0.0025, 0, -0.0025, -0.005, -0.0075], len(gauss_array_original)), xerr=0.2, fmt='.')
    plt.errorbar(gauss_array_original, pdf, xerr=sd_array_original, fmt='.', color= 'b')

    # IMPORTANT This indexing matches with the graph (starting form 0). It doesn't match with the p_list indexing
    for i, txt in enumerate(np.array(index_array)):
        ax.annotate(txt, ( gauss_array_original[i], np.resize( [0.0075, 0.005, 0.0025, 0, -0.0025, -0.005, -0.0075], len(gauss_array_original))[i] ), fontsize = 7)
        # ax.annotate(txt, ( sd_array_original[i], np.resize( [0.0035, -0.0035], len(sd_array_original))[i] ), fontsize = 7)
    # for i, txt in enumerate( np.around(np.array(sd_array_original), decimals=1) ):
    #     ax.annotate(txt, ( gauss_array_original[i], np.resize([0.005, -0.0005 -0.005], len(gauss_array_original))[i] ), fontsize = 7)
    # plt.grid()
    plt.show()
    # _ = raw_input("Press [enter] to continue.")


    print "done"
