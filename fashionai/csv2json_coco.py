from pycocotools.coco import COCO
import json
import tablib
import datetime

def to_coco_keypoints(raw_pts):
    coco_pts = []
    for pt in raw_pts:
        p = pt.split('_')
        for val in p:
            v = int(val)
            coco_pts.append(0 if v < 0 else v)
    return coco_pts

coco=COCO()
file_path = "/data/wangli/fashionai_keypoint/train/Annotations/annotations.csv"
json_path = "/data/wangli/fashionai_keypoint/train/Annotations/annotations.json"

fashion_type = ['blouse','outwear','trousers','skirt','dress']
categories = dict()
type_num = 0
for type in fashion_type:
    type_num += 1
    categories[type] = type_num

imported_data = tablib.Dataset().load(open(file_path).read())
#print(imported_data.export('json'))
dataset = dict()
annotations = list()
images = list()
image_id = 0
for data in imported_data:
    print(data)
    ann = dict()
    image_id = image_id + 1
    ann['image_id'] = categories[data[1]]
    ann['keypoints'] = to_coco_keypoints(data[2:])
    ann['iscrowd'] = 0
    ann['id'] = image_id
    annotations.append(ann)
    img = dict()
    img['id'] = image_id
    img['file_name'] = data[0]
    images.append(img)

dataset['annotations'] = annotations
dataset['images'] = images

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

CATEGORIES = [
    {
        'id': 1,
        'name': 'blouse',
        'supercategory': 'cloth',
    },
    {
        'id': 2,
        'name': 'outwear',
        'supercategory': 'cloth',
    },
    {
        'id': 3,
        'name': 'trousers',
        'supercategory': 'cloth',
    },
    {
        'id': 4,
        'name': 'skirt',
        'supercategory': 'cloth',
    },
    {
        'id': 5,
        'name': 'dress',
        'supercategory': 'cloth',
    },
]

dataset['info'] = INFO
dataset['categories'] = CATEGORIES
dataset['licenses'] = LICENSES

json.dump(dataset, open(json_path,'w'))
