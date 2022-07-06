import json

f = open('../data/labels/labels.json')
label = json.load(f)

print(label)
print(type(label))
print(label[11])
print(len(label))

print(type(label[0]))
print(label[0]['name'])
print(label[0]['attributes']['scene'])

print(label[0].keys())
print(label[0]['name'])
print(label[0]['attributes'])
print(label[0]['timestamp'])
print(label[0]['labels'])

for l in label[0]['labels']:
    print(l)


for l in label[0]['labels']:
    print(l.keys())

print(label[0]['labels'][0]['category'])
print(label[0]['labels'][0]['box2d'])
