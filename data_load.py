import pandas as pd
import random
import csv

df = pd.read_csv("labelled_newscatcher_dataset.csv", encoding='utf-8', sep=';')
print(df.columns.values)
print(df.iloc[0])
'''
topic                                                       SCIENCE
link              https://www.eurekalert.org/pub_releases/2020-0...
domain                                               eurekalert.org
published_date                                  2020-08-06 13:59:45
title             A closer look at water-splitting's solar fuel ...
lang                                                             en
'''
labels = set(df['topic'])
contents = df['title']
count = {}
cal = {}
for p in df['topic']:
    cal[p] = 0
    try:
        count[p] += 1
    except KeyError:
        count[p] = 1
print(count)
"""
{
	'SCIENCE': 3774, 'TECHNOLOGY': 15000,
	'HEALTH': 15000, 'WORLD': 15000,
	'ENTERTAINMENT': 15000, 'SPORTS': 15000,
	'BUSINESS': 15000, 'NATION': 15000
}
"""
print(set(df['topic']))
# # 以下是按照70：15：15的比例来均分数据集为训练集train.csv、验证集dev.csv、测试集test.csv
# train, val, test = [], [], []
# for i, label in enumerate(df['topic']):
#     if cal[label] < count[label] * 0.7:
#         train.append({'label': label, 'content': contents[i]})
#     elif cal[label] < count[label] * 0.85:
#         val.append({'label': label, 'content': contents[i]})
#     else:
#         test.append({'label': label, 'content': contents[i]})
#     cal[label] += 1
# random.shuffle(train)
# random.shuffle(val)
# random.shuffle(test)
#
# # with open('train.csv', 'a', newline='', encoding='utf-8') as f:
# #     writer = csv.DictWriter(f, ['label','content'],delimiter=';')
# #     writer.writeheader()
# #     writer.writerows(train)
# # with open('valid.csv', 'a', newline='', encoding='utf-8') as f:
# #     writer = csv.DictWriter(f, ['label','content'],delimiter=';')
# #     writer.writeheader()
# #     writer.writerows(val)
# # with open('test.csv', 'a', newline='', encoding='utf-8') as f:
# #     writer = csv.DictWriter(f, ['label','content'],delimiter=';')
# #     writer.writeheader()
# #     writer.writerows(test)
#
# # 去除重复标签
# unique_labels = set(labels)
# # 将标签转换为字符串，每个标签占一行
# label_str = '\n'.join(unique_labels)
# # 将字符串写入到 class.txt 文件中
# with open('class.txt', 'w') as f:
#     f.write(label_str)