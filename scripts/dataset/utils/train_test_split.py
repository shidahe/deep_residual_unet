import numpy as np
import json

def prepare_points_xy(points_raw):
    points = []
    for j in points_raw:
        if j != 'Skip':
            res = {}
            for key in j:
                polygons = []
                try:
                    type(j[key][0])
                except:
                    print(j)

                if type(j[key][0]) == list:
                    for pp in j[key]:
                        polygons.append({'x': np.array([h['x'] for h in pp]), 'y': np.array([h['y'] for h in pp])})
                else:
                    polygons.append({'x': np.array([h['x'] for h in j[key]]), 'y': np.array([h['y'] for h in j[key]])})
                res[key] = polygons
            points.append(res)
    return points

import sys
import argparse

parser = argparse.ArgumentParser(description='Augment photos')

parser.add_argument('p', metavar='--p', type=float,
                    help='path to pascal-voc directory try (Spot)')

parser.add_argument('path', metavar='--path', type=str,
                    help='times to make augmentation per image try (1)')

args = parser.parse_args()
a, path = args.p, args.path


JPEGImages = 'JPEGImages'
# Annotations
ImageSets = 'ImageSets'

from sklearn.utils import shuffle as sh
import os

print('reading names')

file_name = None
for i in os.listdir(path):
    if i.endswith('.json'):
        file_name = i
if not file_name:
    raise Exception('json file not found')

with open(os.path.join(path, file_name)) as f:
    data = json.load(f)
import os
names = [i['External ID'] for i in data]
data = sh(names)
level = int(len(data) * a)

print('spliting')
x_train, x_test = data[:level], data[level:]

print('writing')
f_train = open(os.path.join(path, ImageSets, 'train.txt'), 'w')
f_test = open(os.path.join(path, ImageSets, 'test.txt'), 'w')


f_train.writelines('\n'.join(x_train))
f_test.writelines('\n'.join(x_test))

f_train.close()
f_test.close()

print('done:')
print('train: ' + str(len(x_train)), 'test: ' + str(len(x_test)))