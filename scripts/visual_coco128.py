import os, cv2
import numpy as np

img_base_path = '../dataset/coco128-seg/images/train2017'
lab_base_path = '../dataset/coco128-seg/labels/train2017'

label_path_list = [i.split('.')[0] for i in os.listdir(img_base_path)]
for path in label_path_list:
    image = cv2.imread(f'{img_base_path}/{path}.jpg')
    h, w, c = image.shape
    label = np.zeros((h, w), dtype=np.uint8)
    with open(f'{lab_base_path}/{path}.txt') as f:
        mask = np.array(list(map(lambda x:np.array(x.strip().split()), f.readlines())))
    for i in mask:
        i = np.array(i, dtype=np.float32)[1:].reshape((-1, 2))
        i[:, 0] *= w
        i[:, 1] *= h
        label = cv2.fillPoly(label, [np.array(i, dtype=np.int32)], color=255)
    image = cv2.bitwise_and(image, image, mask=label)
    cv2.imshow('Pic', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()