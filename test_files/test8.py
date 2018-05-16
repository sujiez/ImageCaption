import PIL
from PIL import Image
import numpy as np
import cv2

# image_data = Image.open('./COCO_val2014_000000581726.jpg')
#
# image_data = image_data.resize([224, 224], PIL.Image.LANCZOS)
#
# print np.array(image_data).shape
#
# image_data.save('./e.jpg')
# path_list = ['./COCO_val2014_000000290833.jpg', 'COCO_val2014_000000290951.jpg', 'COCO_val2014_000000581929.jpg']
# save_listA = ['./1a.jpg','./2a.jpg','./3a.jpg']
# save_listB = ['./1b.jpg','./2b.jpg','3b.jpg']
#
# for i, im_path in enumerate(path_list):
#     image_data = cv2.imread(im_path)
#     image_data = cv2.resize(image_data, (224, 224), interpolation=cv2.INTER_AREA)
#     print type(image_data)
#     cv2.imwrite(save_listA[i], image_data)
#
#     image_data = Image.open(im_path)
#     image_data = image_data.resize([224, 224], PIL.Image.NEAREST)
#     image_data.save(save_listB[i])

im_path = './data/MScoco/raw-data/train2014/COCO_train2014_000000210175.jpg'

image_data = cv2.imread(im_path)
image_data = cv2.resize(image_data, (224, 224), interpolation=cv2.INTER_AREA)
print type(image_data)
print image_data.shape
cv2.imwrite('a.jpg', image_data)

# print type(image_data)
# print image_data.shape
