##############################################################################
"""Convert the csv of fashionai data to coco format, so that we can easily use
Detectron to train for keypoints estimation.
"""
import json
import tablib
import datetime
from PIL import Image
import os
import numpy as np

# can't run due to error: if isinstance(pickle[0], list):
json_file = "/data/wangli/tmp/fashion_output/output.json"
csv_file = "/data/wangli/tmp/fashion_output/output.csv"
imported_data = json.load(open(json_file, 'r'))
out_data = tablib.Dataset()

data = imported_data['images']

points = []
for img in data:
    line = []
    line.append(img['file_name'])
    line.append(img['category'])
    kps = img['keypoints']
    for x,y,v in zip(kps[0::3],kps[1::3],kps[2::3]):
        p = '{}_{}_{}'.format(x,y,v)
        line.append(p)
    print(line)
    out_data.append(line)

with open(csv_file, 'w') as f:
    f.write(out_data.export('csv'))
