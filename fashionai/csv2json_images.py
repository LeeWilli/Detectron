##############################################################################
"""Convert the csv of fashionai data to coco format, so that we can easily use
Detectron to train for keypoints estimation.
"""
from pycocotools.coco import COCO
import json
import tablib
import datetime
from PIL import Image
import os
import numpy as np

def to_coco_keypoints(raw_pts):
    coco_pts = []
    for pt in raw_pts:
        p = pt.split('_')
        for val in p[:2]:
            v = int(val)
            coco_pts.append(0 if v < 0 else v)
        coco_pts.append(int(p[2])+1)
    return coco_pts, len(coco_pts)/3


"""compute a box containing all points and the box's area.

    return xywh and area
"""
def find_convex_box(kps):
    kp = np.array(kps)
    x = kp[0::3]  # 0-indexed x coordinates
    y = kp[1::3]  # 0-indexed y coordinates
    x = x[x>0] #filter unexisted points
    y = y[y>0] #filter unexisted points
    xmin = int(np.min(x))
    xmax = int(np.max(x))
    ymin = int(np.min(y))
    ymax = int(np.max(y))
    area = (xmax-xmin)*(ymax-ymin)
    return [xmin, ymin, xmax-xmin, ymax-ymin], area

def box2segment(bbox):
    segs = []
    seg = []
    #bbox[] is x,y,w,h
    #left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    #left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    #right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    #right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    segs.append(seg)
    return segs


coco=COCO()
file_path = "/data/wangli/fashionai_keypoint/test/test.csv"
json_path = "/data/wangli/fashionai_keypoint/test/test.json"
image_path = "/data/wangli/fashionai_keypoint/test"

INFO = {
    "description": "FashionAI keypoints Dataset",
    "url": "https://tianchi.aliyun.com/competition/information.htm?raceId=231648",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "yscz",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://yscz/"
    }
]

keypoints_name = "neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out"
kp_name = keypoints_name.split(',')
CATEGORIES = [
    {
        'id': 1,
        'name': 'blouse',
        'supercategory': 'cloth',
        'skeleton':[[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[10,11],[12,13],[13,14],[14,15]],
        'keypoints': kp_name,
    },
    {
        'id': 2,
        'name': 'outwear',
        'supercategory': 'cloth',
        'skeleton': [[1, 2], [4, 5], [5, 6], [6, 7], [10, 11], [12, 13], [13, 14], [14, 15]],
        'keypoints': kp_name,
    },
    {
        'id': 3,
        'name': 'trousers',
        'supercategory': 'cloth',
        'skeleton': [[16, 17], [20, 21], [22, 23], [23, 24]],
        'keypoints': kp_name,
    },
    {
        'id': 4,
        'name': 'skirt',
        'supercategory': 'cloth',
        'skeleton':[[16,17],[17,18],[18,19]],
        'keypoints': kp_name,
    },
    {
        'id': 5,
        'name': 'dress',
        'supercategory': 'cloth',
        'skeleton': [[1, 2], [2,3], [3,4], [4, 5], [5, 6], [6, 7], [10, 11], [11,12],[12, 13], [18, 19]],
        'keypoints': kp_name,
    },
]
dataset = dict()
dataset['info'] = INFO
dataset['categories'] = CATEGORIES
dataset['licenses'] = LICENSES

fashion_type = ['blouse','outwear','trousers','skirt','dress']
categories = dict()
type_num = 0
for type in fashion_type:
    type_num += 1
    categories[type] = type_num

imported_data = tablib.Dataset().load(open(file_path).read())
#print(imported_data.export('json'))

annotations = list()
images = list()
image_id = 0
for data in imported_data:
    img = dict()
    img['id'] = image_id
    img['file_name'] = data[0]
    filename = os.path.join(image_path, data[0])
    im = Image.open(filename)
    img['width'] = im.size[0]
    img['height'] = im.size[1]
    images.append(img)

dataset['images'] = images

json.dump(dataset, open(json_path,'w'))
