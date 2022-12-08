import os, cv2, json
import numpy as np

classes = ['square', 'triangle']

base_path = '../dataset/labelme_dataset'
path_list = [i.split('.')[0] for i in os.listdir(base_path)]
for path in path_list:
    image = cv2.imread(f'{base_path}/{path}.jpg')
    h, w, c = image.shape
    with open(f'{base_path}/{path}.json') as f:
        masks = json.load(f)['shapes']
    with open(f'{base_path}/{path}.txt', 'w+') as f:
        for idx, mask_data in enumerate(masks):
            mask_label = mask_data['label']
            if '_' in mask_label:
                mask_label = mask_label.split('_')[0]
            mask = np.array([np.array(i) for i in mask_data['points']], dtype=np.float)
            mask[:, 0] /= w
            mask[:, 1] /= h
            mask = mask.reshape((-1))
            if idx != 0:
                f.write('\n')
            f.write(f'{classes.index(mask_label)} {" ".join(list(map(lambda x:f"{x:.6f}", mask)))}')