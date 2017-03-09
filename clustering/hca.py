import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


def hca_clustering(data, plot_title = 'Hierarchical Clustering Dendrogram (truncated)'):
    Z = linkage(data, method='centroid', metric='euclidean')  # method='single', 'centroid', metric='euclidean'
    clusters_f = fcluster(Z, t=700, depth=10, criterion="distance")
    print "clusters_f, shape ", clusters_f, len(clusters_f)
    c, coph_dists = cophenet(Z, pdist(data))
    print "c, coph_d", c, coph_dists


    # Plot dendrogram...
    plt.title(plot_title)
    plt.xlabel('sample index or (cluster size)')
    plt.ylabel('distance')
    dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=275,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=4.,
        show_leaf_counts=True,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )
    plt.show()