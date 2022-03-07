import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm

from tensorflow import keras
from tensorflow.keras import layers


import numpy as np
import os
import time
import json
import re
import pickle



# preprocess and prepare captions
# open train_keys json 
with open('train_keys.json', 'r') as f:
    train_keys = json.load(f)

# open val_keys json 
with open('val_keys.json', 'r') as f:
    val_keys = json.load(f)

train_paths = list(train_data.keys())
val_paths = list(valid_data.keys())

# open saved train and val dicts
# open train_data json 
with open('train_data.json', 'r') as f:
    train_data = json.load(f)

# open valid_data json 
with open('valid_data.json', 'r') as f:
    valid_data = json.load(f)

train_c = []
train_i = []

for i in train_paths:
    train_caption_list = train_data[i]
    train_c.extend(train_caption_list)
    train_i.extend([i] * len(train_caption_list))

val_c = []
val_i = []

for i in val_paths:
    val_caption_list = valid_data[i]
    val_c.extend(val_caption_list)
    val_i.extend([i] * len(val_caption_list))

all_c = train_c + val_c
all_i = train_i + val_i


# with Tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<unk>", lower=True, split=' ',
                                                  filters='!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~')
tokenizer.fit_on_texts(all_c)
seqs = tokenizer.texts_to_sequences(all_c)
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(seqs, padding='post')

# save tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Find the maximum length of any caption in the dataset
def max_caption_length(captions):
    return max(len(caption) for caption in captions)

# Calculate the max_length, which is used to store the attention weights
max_length = max_caption_length(seqs)

merge_dict = collections.defaultdict(list)
for image, caption in zip(all_i, cap_vector):
    merge_dict[image].append(caption)


train_slice = []
val_slice = []

for i in all_i:
    if i[:i.index("/")] == 'train2014':
        train_slice.append(i)
    else:
        val_slice.append(i)

train_slice = set(train_slice)
val_slice = set(val_slice)
print('train length:', len(train_slice))
print('val length:', len(val_slice))


train_images = []
train_captions = []
for imgt in train_slice:
    capt_len = len(merge_dict[imgt])
    train_images.extend([imgt] * capt_len)
    train_captions.extend(merge_dict[imgt])

val_images = []
val_captions = []
for imgv in val_slice:
    capv_len = len(merge_dict[imgv])
    val_images.extend([imgv] * capv_len)
    val_captions.extend(merge_dict[imgv])

len(train_images), len(train_captions), len(val_images), len(val_captions)






