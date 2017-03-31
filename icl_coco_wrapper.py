from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import icl_util as util
from LogUtil import LogUtil
from os.path import abspath, expanduser
import random
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

def get_captions(coco_obj, img_id):
    annIds = coco_obj.getAnnIds(imgIds=[img_id])
    anns = coco_obj.loadAnns(annIds)
    captions = list()
    for ann in anns:
        captions.append(ann['caption'])
    return captions
#enddef

def show_image(coco_obj, img):
    I = io.imread('http://mscoco.org/images/%d' % (img['id']))
    plt.imshow(I); plt.axis('off')
    annIds = coco_obj.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco_obj.loadAnns(annIds)
    coco_obj.showAnns(anns)
    plt.show()
#enddef

dataDir = abspath(expanduser("~/source/data/mscoco/"))
dataType = 'train2014'
instance_file = '%s/annotations/instances_%s.json' % (dataDir, dataType)
caption_file = '%s/annotations/captions_%s.json' % (dataDir, dataType)

log = LogUtil(lvl='debug', delay=45)

# initialize COCO api for instance and caption annotations
coco = COCO(instance_file)
coco_caps = COCO(caption_file)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
log.info("Coco Categories")
nms = [cat['name'] for cat in cats]
cat_rows = util.list_to_rows(nms, 4)
print util.rows_to_str(cat_rows, False)
log.info("Super Categories")
nms = set([cat['supercategory'] for cat in cats])
cat_rows = util.list_to_rows(nms, 4)
print util.rows_to_str(cat_rows, False)
cat_dict = dict()
for cat in cats:
    cat_dict[cat['id']] = cat

# load images with specific categories
log.info("Loading images")
use_rand = False
img_ids = list()
if use_rand:
    total_imgs = len(coco.getImgIds())
    cat_ids = coco.getCatIds(catNms=['person', 'animal'])
    cat_img_ids = coco.getImgIds(catIds=cat_ids)
    log.info(None, "%d images (%.2f%%) with [person] or [animal]",
         len(cat_img_ids), 100.0 * len(cat_img_ids) / total_imgs)
    img_ids = random.sample(cat_img_ids, 250)
else:
    with open('coco_sub_train_imgs_orig.txt', 'r') as f:
        for line in f.xreadlines():
            line = line.replace("COCO_" + dataType + "_", "")
            line = line.replace(".jpg", "")
            img_ids.append(int(line))
        #endfor
    #endwith
#endif

imgs = coco.loadImgs(img_ids)

# write the img IDs, captions, and bounding boxes to a file
sub_img_ids = list()
ll_captions = list()
ll_bbox = list()

for img in imgs:
    #store the image IDs
    img_id_str = "COCO_" + dataType + "_%012d.jpg" % (img['id'])
    sub_img_ids.append(img_id_str + "\n")

    #captions
    captions = get_captions(coco_caps, img['id'])
    for i in range(0, len(captions)):
        ll_captions.append(img_id_str + "#" + str(i) + "\t" + captions[i].strip() + "\n")

    #and bounding boxes
    annIDs = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIDs)
    for j in range(0, len(anns)):
        ann = anns[j]
        line = img_id_str + "," + str(j) + "," + str(ann['id'])
        line += "," + cat_dict[ann['category_id']]['name']
        line += "," + cat_dict[ann['category_id']]['supercategory']
        for dim in ann['bbox']:
            line += "," + str(dim)
        line += "\n"
        ll_bbox.append(line)
    #endfor
#endfor

with open('coco_sub_train_imgs.txt', 'w') as f:
    f.writelines(sub_img_ids)
    f.close()
with open('coco_sub_train_caps.txt', 'w') as f:
    f.writelines(ll_captions)
    f.close()
with open('coco_sub_train_bbox.csv', 'w') as f:
    f.writelines(ll_bbox)
    f.close()