import os
import sys
import csv
import math
import numpy as np
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
from label import label
from collections import OrderedDict
from visualization import visualization


tfidf_path = sys.argv[1]
terms_path = sys.argv[2]
files = os.listdir(tfidf_path)

with open(terms_path, 'r') as file:
	terms = list(csv.reader(file))
terms = terms[0]
print('Term count', len(terms))

doc_vecs = []
for i in tqdm(files):
    with open(tfidf_path + '/' + i, 'r') as file:
        d = dict()
        tmp = list(csv.reader(file))
        for j in tmp:
            d[j[0]] = float(j[1])
        tmp_vec = []
        for t in terms:
            if t in d:
                tmp_vec.append(d[t])
            else:
                tmp_vec.append(0.0)
        tmp_vec = np.array(tmp_vec, dtype=float)
        norm = np.linalg.norm(tmp_vec)
        if norm == 0.0:
            tmp_vec += math.sqrt(1.0 / float(len(terms)))
            norm = 1.0
        tmp_vec /= norm
        doc_vecs.append(tmp_vec)

doc_vecs = np.array(doc_vecs, dtype=float)
print('Doc vecs shape', doc_vecs.shape)

print(">>>>> Clustering")
cluster_count = 16
result = AgglomerativeClustering(
    n_clusters=cluster_count, affinity='cosine', linkage='complete').fit(doc_vecs).labels_

output = dict()
for i in range(len(result)):
    if result[i] not in output:
        output[result[i]] = []
    output[result[i]].append(files[i].split('.')[0])
output = OrderedDict(sorted(output.items(), key=lambda t: t[0]))

# for k, v in output.items():
#     print(len(v))

# with open('output.csv', 'w') as f:
#     writer = csv.writer(f)
#     for i in output:
#         writer.writerow(output[i])

print(">>>>> Labeling")
labels_list = label(output, files, terms, tfidf_path)
with open('label.csv', 'w') as f:
    writer = csv.writer(f)
    for labels in labels_list:
        for label in labels:
            writer.writerow(label)
        writer.writerow(['\n'])

#unique, counts = np.unique(result, return_counts=True)
#d = dict(zip(unique, counts))
#for i in d:
#    print(str(i) + ' : ' + str(d[i]))

#print(result)

visualization(result, cluster_count, doc_vecs)
