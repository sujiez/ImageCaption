import json
import configuration as conf

with open(conf.coco_val_caption_path) as f:
    val_raw = json.load(f)
    pass

imid = set()
[imid.add(im['id']) for im in val_raw['images']]
capid = set()
[capid.add(c['id']) for c in val_raw['annotations']]
print len(imid)
print len(capid)