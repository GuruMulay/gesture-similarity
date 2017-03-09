import seaborn as sns
import matplotlib as plt
sns.set()

# gesture_mat1 = sns.load_dataset("flights")


def matrix_heatmap(data, vmin, vmax):
    ax = sns.heatmap(data, vmin, vmax)
    print "printing heatmap....."
