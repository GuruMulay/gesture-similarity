import os
import numpy as np
import shutil

home = os.path.expanduser('~');
data_dir = os.path.join(home, 'gesture_data/')
data_dir_with_sound = os.path.join(home, 'gesture_data/with sound/')
data_dir_without_sound = os.path.join(home, 'gesture_data/without sound/')

# gesture = "arms move down"
p = 'Participant'
b = '-Block'

p_list_with_sound = [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30, 33, 34, 37, 38, 41, 42]
p_list_without_sound = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 31, 32, 35, 36, 39, 40]


def distribute_data():

    for dir_ns in [data_dir_with_sound, data_dir_without_sound]:

        for gesture in os.listdir(dir_ns):

            gest_dir = os.path.join(dir_ns, gesture)

            count = 0
            for f in os.listdir(gest_dir):
                count += 1
                # print "f.find(p) ",  f.find(p)

                if f.find(p) == -1:
                    # print "not found"
                    continue
                else:
                    p_number = f[f.find(p)+len(p)+1 : f.find(b)]

                # print "p_num", p_number, f
                # print [f.find(b)]

                p_dir_abs = os.path.join(dir_ns, gesture, ("p" + str(p_number)))
                if not os.path.exists(p_dir_abs):
                    if (0 < int(p_number) < 43):
                        print "creating directory..."
                        os.makedirs(p_dir_abs)

                shutil.move(os.path.join(gest_dir, f), os.path.join(p_dir_abs, f))

            # print os.path.join(gest_dir, f), os.path.join(p_dir_abs, f)
            # print gest_dir
            print count

def create_empty_folders_for_participants():
    dir_ns = data_dir_with_sound
    for gesture in os.listdir(dir_ns):
        gest_dir = os.path.join(dir_ns, gesture)
        for p_number in p_list_with_sound:
            print p_number

            p_dir_abs = os.path.join(gest_dir, ("p" + str(p_number)))
            print p_dir_abs
            if not os.path.exists(p_dir_abs):
                if (0 < int(p_number) < 43):
                    print "creating directory..."
                    os.makedirs(p_dir_abs)

    dir_ns = data_dir_without_sound
    for gesture in os.listdir(dir_ns):
        gest_dir = os.path.join(dir_ns, gesture)
        for p_number in p_list_without_sound:
            print p_number

            p_dir_abs = os.path.join(gest_dir, ("p" + str(p_number)))
            print p_dir_abs
            if not os.path.exists(p_dir_abs):
                if (0 < int(p_number) < 43):
                    print "creating directory..."
                    os.makedirs(p_dir_abs)


def write_matrix(data, filename):

    with file(filename, 'w') as outfile:
        # line starting with "#" will be ignored by numpy.loadtxt
        # outfile.write('# Array shape: {0}\n'.format(data.shape))
        outfile.write('# Array shape: ')

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in data:

            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')


def write_matrix_2d(data, filename):

    with file(filename, 'w') as outfile:
        np.savetxt(outfile, data, fmt='%-7.4f')

# gets the skeleton files and distributes them according to which participant performed that gesture
# distribute_data()

# creates empty folders for the participants that do not have any gesture in the gesture categories
# create_empty_folders_for_participants()