import cv2
import json
import os

file_names = os.listdir('../data/images')

f = open('../data/labels/labels.json')
labels = json.load(f)

if not os.path.exists('../classification_data'):
    os.mkdir('../classification_data')

cnt = 0

for label in labels:
    if label['name'] in file_names:
        image = cv2.imread('../data/images/' + label['name'])
        for box in label['labels']:
            if 'box2d' in box:
                x1 = int(box['box2d']['x1'])
                y1 = int(box['box2d']['y1'])
                x2 = int(box['box2d']['x2'])
                y2 = int(box['box2d']['y2'])
                crop_image = image[y1:y2+1, x1:x2+1]

                if not os.path.exists('../classification_data/' + box['category']):
                    os.mkdir('../classification_data/' + box['category'])

                # cv2.imwrite('../classification_data/' + box['category'] + '/' + str(cnt) + '.jpg', crop_image)
                cv2.imwrite(f'../classification_data/{box["category"]}/{str(cnt)}.jpg', crop_image)
                cnt += 1
