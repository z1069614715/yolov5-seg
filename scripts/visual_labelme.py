import os, cv2, json
import numpy as np

base_path = '../dataset/labelme_dataset'
path_list = [i.split('.')[0] for i in os.listdir(base_path) if 'json' in i]
for path in path_list:
    image = cv2.imread(f'{base_path}/{path}.jpg')
    h, w, c = image.shape
    label = np.zeros((h, w), dtype=np.uint8)
    with open(f'{base_path}/{path}.json') as f:
        mask = json.load(f)['shapes']
    for i in mask:
        i = np.array([np.array(j) for j in i['points']])
        label = cv2.fillPoly(label, [np.array(i, dtype=np.int32)], color=255)
    image = cv2.bitwise_and(image, image, mask=label)
    cv2.imshow('Pic', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()