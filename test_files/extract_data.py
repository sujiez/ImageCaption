import json

caption = json.load(open("./data/raw-data/annotations/captions_train2014.json"))

#info = caption['info']
reverse = dict([(i['id'], i) for i in caption['images']])

part = caption['annotations'][:500]

image_info = [reverse[j['image_id']] for j in part]

result = {'info':caption['info'], 'images':image_info, 'licenses':caption['licenses'], 'annotations':part}


with open('./sample_annotation.json', 'w') as f:
	json.dump(result, f)

check = json.load(open('./sample_annotation.json'))


print "images: \n"
for i in check['images']:
	print i, "\n"
print "\n", "annotations: \n"
for i in check['annotations']:
	print i, "\n"

print len(check['images']), "\n"
print len(check['annotations']), "\n"