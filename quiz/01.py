import json

file_name = input()

f = open('../data/labels/labels.json')
labels = json.load(f)

for label in labels:
    if label['name'] == file_name:
        for l in label['labels']:
            if 'box2d' in l:
                print(l['category'], l['box2d'])
                print(l['box2d']['x1'], l['box2d']['y1'], l['box2d']['x2'], l['box2d']['y2'])
