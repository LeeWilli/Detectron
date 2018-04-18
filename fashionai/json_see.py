import pytablereader as ptr
import pytablewriter as ptw
import io
from pycocotools.coco import COCO
import numpy as np
#import skimage.io as io
import matplotlib.pyplot as plt
import pylab

coco=COCO()
file_path = "/data/wangli/fashionai_keypoint/train/Annotations/annotations.csv"
loader = ptr.CsvTableFileLoader(file_path)

for table_data in loader.load():
    print("\n".join([
        "load from file",
        "==============",
        "{:s}".format(ptw.dump_tabledata(table_data)),
    ]))

writer = ptw.TableWriterFactory.create_from_format_name("rst")
writer.stream = io.open("load_url_result.rst", "w", encoding=loader.encoding)
for table_data in loader.load():
    writer.from_tabledata(table_data)
    writer.write_table()