import json
import cv2

file_name = input()
image = cv2.imread('../data/images/' + file_name)

f = open('../data/labels/labels.json')
labels = json.load(f)

for label in labels:
    if label['name'] == file_name:
        for l in label['labels']:
            if 'box2d' in l:
                # print(l['category'], l['box2d'])
                # print(l['box2d']['x1'], l['box2d']['y1'], l['box2d']['x2'], l['box2d']['y2'])
                x1 = int(l['box2d']['x1'])
                y1 = int(l['box2d']['y1'])
                x2 = int(l['box2d']['x2'])
                y2 = int(l['box2d']['y2'])
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
                cv2.putText(image, l['category'], (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

cv2.imshow('image', image)
cv2.waitKey(0)
