import numpy as np


def slice(a, scale):
    temp=[]
    count=0
    while count<len(a):
        temp.append(a[count, :])
        count+=scale+1

    return np.vstack(temp)


def sliding_window(array, window_size, scale):
    rows, cols = array.shape[0], array.shape[1]
    temp = []
    if(rows<window_size):
        #print 'Window size greater than number of samples!'
        return 0
    else:
        for i in range((rows-window_size)+1):
            #print 'window range',(i, i+window_size)
            window = array[i:i + window_size, :]
            window = slice(window,scale)
            #print 'window size shape is: ', window.shape
            #window = window[window.shape[0]/2, :].reshape(1,-1)
            temp.append(window)

    #print 'Length of temp is: ', len(temp), temp[0].shape
    #print '\nSize of temp all window size is: ', np.vstack(temp).shape
    return temp
    #return np.vstack(temp)


def sliding_window_dataset(data_list, window_size, scale):
    return [sliding_window(data, window_size, scale) for data in data_list]


#a = np.arange(45).reshape(15,3)
#sliding_window(a,5,0)