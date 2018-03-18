import numpy as np
from numpy import genfromtxt
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')


def visualization(results, cluster_count, doc_vecs):
    """
    results: {
        'cluster id': [vector id, ...],
        ...
    }
    """
    print(">>>>>>> TSNE")
    vis_data = TSNE(n_components=2, perplexity=10).fit_transform(doc_vecs)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    removed_list = []
    for idx, i in enumerate(vis_x):
        if i > 100 or i < -100:
            removed_list.append(idx)
    for idx, j in enumerate(vis_y):
        if j > 100 or j < -100:
            if idx not in removed_list:
                removed_list.append(idx)

    data = []
    for i in range(cluster_count):
        data.append([[], []])
    for i in range(len(results)):
        if i not in removed_list:
            data[results[i]][0].append(vis_x[i])
            data[results[i]][1].append(vis_y[i])
    data = np.array(data)

    sc_list = []
    for i in range(cluster_count):
        sc = plt.scatter(data[i][0], data[i][1], marker='.', alpha=0.7, edgecolors='none')
        sc_list.append(sc)

    cluster_list = []
    for i in range(cluster_count):
        cluster_list.append('Cluster_' + str(i))

    import pdb; pdb.set_trace()
    # plt.legend(tuple(sc_list),
    #     iter(cluster_list),
    #     loc='lower left',
    #     ncol=1,
    #     fontsize=8)

    plt.show()
    fig = plt.gcf()
    fig.savefig('tsne_' + str(cluster_count) + '.png', dpi=300)
