import os
import sys
import csv
import math
from tqdm import tqdm
from operator import itemgetter


post_tfs_path = sys.argv[1]
files = os.listdir(post_tfs_path)

total_term_count = 0
tfs = []
dfs = dict()
doc_names = []

for f in tqdm(files):
    with open(post_tfs_path + '/' + f, 'r') as file:
        doc_names.append(f)
        tmp = list(csv.reader((line.replace('\0', '') for line in file)))
        tmp = [[i[0], int(i[1])] for i in tmp if int(i[1]) >= 20]
        term_sum = 0

        # count df
        for i in tmp:
            term_sum += i[1]
            if i[0] not in dfs:
                dfs[i[0]] = [1]
                dfs[i[0]].append([f])
            else:
                dfs[i[0]][0] += 1
                dfs[i[0]][1].append(f)
        total_term_count += term_sum
        # tmp = [[i[0], float(i[1]) / float(term_sum)] for i in tmp]
        tfs.append(tmp)

print('total term count: ', total_term_count)

# count idf and tfidf
keys = list(dfs.keys())
idfs = dict()
for k in keys:
    idfs[k] = math.log(float(len(tfs)) / float(1 + dfs[k][0]))
tfidf = [[[j[0], j[1] * idfs[j[0]], j[1]] for j in i] for i in tfs]

# write files
if not os.path.isdir('tfidfs'):
    os.makedirs('tfidfs')
if not os.path.isdir('for_label'):
    os.makedirs('for_label')

terms = []
for i in range(len(doc_names)):
    tmp = tfidf[i]
    tmp = sorted(tmp, key=itemgetter(1), reverse=True)
    tmp = tmp[:250]
    with open('tfidfs/' + doc_names[i], 'w') as file:
        writer = csv.writer(file)
        for j in range(len(tmp)):
            terms.append(tmp[j][0])
            writer.writerow(tmp[j][:2])
    with open('for_label/' + doc_names[i], 'w') as file:
        writer = csv.writer(file)
        for j in range(len(tmp)):
            writer.writerow([tmp[j][0], tmp[j][2]])

terms = {}.fromkeys(terms).keys()

with open('terms.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerow(terms)

with open('dfs.csv', 'w') as file:
    writer = csv.writer(file)
    for term, df in dfs.items():
        row = [term, df[0]]
        row.extend(df[1])
        writer.writerow(row)
