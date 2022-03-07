import tensorflow as tf

import collections
from collections import Counter
import random
import numpy as np
import os
import time
import json
import re
import glob
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg



# ## Download and prepare dataset



# Download training image files
train_image_folder = '/train2014/'
if not os.path.exists(os.path.abspath('.') + train_image_folder):
    train_image_zip = tf.keras.utils.get_file('train2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin='http://images.cocodataset.org/zips/train2014.zip',
                                      extract=True)
    train_path = os.path.dirname(train_image_zip) + train_image_folder
    os.remove(train_image_zip)
else:
    train_path = os.path.abspath('.') + train_image_folder


# Download validation image files
val_image_folder = '/val2014/'
if not os.path.exists(os.path.abspath('.') + val_image_folder):
    val_image_zip = tf.keras.utils.get_file('val2014.zip',
                                      cache_subdir=os.path.abspath('.'),
                                      origin='http://images.cocodataset.org/zips/val2014.zip',
                                      extract=True)
    val_path = os.path.dirname(val_image_zip) + val_image_folder
    os.remove(val_image_zip)
else:
    val_path = os.path.abspath('.') + val_image_folder

# Download caption annotation files
annotation_folder = '/annotations/'
if not os.path.exists(os.path.abspath('.') + annotation_folder):
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                           cache_subdir=os.path.abspath('.'),
                                           origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                           extract=True)
    train_annot = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
    val_annot = os.path.dirname(annotation_zip)+'/annotations/captions_val2014.json'
    os.remove(annotation_zip)


train_path = 'train2014/'

val_path = 'val2014/'

train_annot_path = 'annotations/captions_train2014.json'
val_annot_path = 'annotations/captions_val2014.json'

with open(train_annot_path, 'r') as f:
    train_annot = json.load(f)


with open(val_annot_path, 'r') as f:
    val_annot = json.load(f)


# train images
train_images = train_annot['images']
train_captions = train_annot['annotations']


# group train annotations by image
train_dict = collections.defaultdict(list)
for x in train_captions:
    caption = f"<start> {x['caption']} <end>"
    image_path = train_path + 'COCO_train2014_' + '%012d.jpg' % (x['image_id'])
    train_dict[image_path].append(caption)


# validation
val_images = val_annot['images']
val_captions = val_annot['annotations']


# group all val annotations by image
val_dict = collections.defaultdict(list)
for x in val_captions:
    
    caption = f"<start> {x['caption']} <end>"
    val_image_path = val_path + 'COCO_val2014_' + '%012d.jpg' % (x['image_id'])
    val_dict[val_image_path].append(caption)


# filter images that have exactly 5 captions
def filter_dict(dict_):
    new_dict = dict()
    # Iterate over all the items in dictionary
    for key, value in dict_.items():
      if len(value) == 5:
        new_dict[key] = value
    return new_dict


final_train_dict = filter_dict(train_dict)
print('preprocessed captions lengths %d -> %d' % (len(train_dict.keys()), len(final_train_dict.keys())))

final_val_dict = filter_dict(val_dict)
print('preprocessed captions lengths %d -> %d' % (len(val_dict.keys()), len(final_val_dict.keys())))

print(len(list(final_train_dict.keys())))
print(len(list(final_val_dict.keys())))


train_keys = list(final_train_dict.keys())


valid_image_keys = list(final_val_dict.keys())
random.shuffle(valid_image_keys)

# Select the first 5000 image_paths for validation
val_keys = valid_image_keys[:5000]
print(len(val_keys))

# Select the last 5000 image_paths for test
test_keys = valid_image_keys[-5000:]
print(len(test_keys))


# save as json for future use
with open('train_keys.json', 'w') as f:
    json.dump(train_keys, f)



with open('val_keys.json', 'w') as f:
    json.dump(val_keys, f)


with open('test_keys.json', 'w') as f:
    json.dump(test_keys, f)


def subset(data_dict, image_names):
    subset_dict = {image_name:captions for image_name, captions in data_dict.items() if image_name in image_names}
    return subset_dict



train_data = dict(final_train_dict)
valid_data = subset(final_val_dict, val_keys)
test_data = subset(final_val_dict, test_keys)


# save dict as json for future use
with open('train_data.json', 'w') as f:
    json.dump(train_data, f)


with open('valid_data.json', 'w') as f:
    json.dump(valid_data, f)


with open('test_data.json', 'w') as f:
    json.dump(test_data, f)


print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))
print("Number of testing samples: ", len(test_data))
