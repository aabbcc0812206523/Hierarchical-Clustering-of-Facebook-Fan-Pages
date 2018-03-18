import os
import sys
import csv
from tqdm import tqdm
import re
import jieba


posts_path = sys.argv[1]
dirs = os.listdir(posts_path)
dir_count = len(dirs)

for idx, d in enumerate(dirs):
    print('Progress: {}/{}'.format(idx + 1, dir_count))
    pages_path = posts_path + '/' + d
    pages = os.listdir(pages_path)

    # create folder for tf files
    if not os.path.isdir('tfs'):
        os.makedirs('tfs')

    for i in tqdm(pages):
        tfs = dict()
        with open(pages_path + '/' + i, 'r') as file:
            r = list(csv.reader((line.replace('\0', '') for line in file)))
            if len(r) >= 2:
                page_name = r[1][0]
                r = r[1:]

                # clear blanks and useless words
                r = [list(jieba.cut(re.sub(r'^https?:\/\/.*[\r\n]*', '', i[1], flags=re.MULTILINE)))
                     for i in r if len(i) >= 2]

                r = [j for i in r for j in i]
                for term in r:
                    if term not in tfs:
                        tfs[term] = 0
                    tfs[term] += 1

                # write files
                with open('tfs/' + page_name.replace('/', '') + '.csv', 'w') as w:
                    writer = csv.writer(w)
                    for term in tfs:
                        writer.writerow([term, tfs[term]])
