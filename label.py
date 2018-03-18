import os
import csv
import operator
from tqdm import tqdm
import numpy as np
import math


def label(clusters, files, terms, tfidf_path):
    print(">>>>>>> Get tfs")
    tf_files = os.listdir('for_label')
    collection_tf = dict()
    for i in tqdm(tf_files):
        with open('for_label/' + i, 'r') as file:
            tfs = list(csv.reader(file))
            for tf in tfs:
                if tf[0] not in collection_tf.keys():
                    collection_tf[tf[0]] = int(tf[1])
                else:
                    collection_tf[tf[0]] += int(tf[1])

    print(">>>>>>> Get dfs")
    collection_df = dict()
    with open('dfs.csv', 'r') as file:
        dfs = list(csv.reader(file))
        for df in dfs:
            collection_df[df[0]] = int(df[1])

    labels_list = []
    for k, cluster in clusters.items():
        print(">>>>>>> Labeling cluster", k)
        term_dict = dict()

        """
        {term: [
            df in cluster,
            tf in cluster,
            df in collection,
            tf in collection,
            fre_pre
        ]}
        """

        for c in cluster:
            tf_of_page = dict()
            with open('for_label/' + c + '.csv', 'r') as file:
                for t in list(csv.reader(file)):
                    tf_of_page[t[0]] = int(t[1])

            with open(tfidf_path + '/' + c + '.csv', 'r') as file:
                tmp = list(csv.reader(file))
                for t in tmp:
                    if t[0] not in term_dict.keys():
                        term_dict[t[0]] = [1]
                        term_dict[t[0]].append(tf_of_page[t[0]])
                    else:
                        term_dict[t[0]][0] += 1
                        term_dict[t[0]][1] += tf_of_page[t[0]]

        for k in term_dict.keys():
            term_dict[k].append(collection_df[k])
            term_dict[k].append(collection_tf[k])
            fre_pre = 2 * math.log(term_dict[k][1]) - math.log(term_dict[k][3])
            term_dict[k].append(fre_pre)

        # by_cluster_df = sorted(term_dict.items(), key=lambda term: term[1][0], reverse=True)
        # by_cluster_tf = sorted(term_dict.items(), key=lambda term: term[1][1], reverse=True)
        by_fre_pre = sorted(term_dict.items(), key=lambda term: term[1][4], reverse=True)

        total_labels = []
        def get_labels(values):
            labels = []
            i = 0
            j = 0
            while i < 10:
                try:
                    j += 1
                    if len(values[j][0]) >= 2 and values[j][0] not in ['http', 'https']:
                        print(values[j][0])
                        labels.append(values[j][0])
                        i += 1
                except IndexError:
                    break
            return labels

        # total_labels.append(get_labels(by_cluster_df))
        # total_labels.append(get_labels(by_cluster_tf))
        total_labels.append(get_labels(by_fre_pre))
        labels_list.append(total_labels)

    return labels_list
