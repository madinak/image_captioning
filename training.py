import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from tqdm import tqdm
from dataset import make_dataset, custom_standardization, reduce_dataset_dim, valid_test_split

from transformer_model import *
from image_preprocessing import load_image_captions
import numpy as np
import os
import time
import json
import re

# set hyperparameters
num_layers = 4
d_model = 512
dff = 2048
num_heads = 8
vocabulary_size = tokenizer.num_words + 1
dropout_rate = 0.3

max_length = 52
BUFFER_SIZE=1000
BATCH_SIZE=64

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


# create df.dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_captions))
train_dataset = train_dataset.map(load_image_captions, 
                       num_parallel_calls=tf.data.experimental.AUTOTUNE) 
train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_captions))
val_dataset = val_dataset.map(load_image_captions, 
                       num_parallel_calls=tf.data.experimental.AUTOTUNE) 
val_dataset = val_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)



# define model
transformer = Transformer(num_layers, d_model, num_heads, dff, vocab_size=vocabulary_size, 
                          pe_input=49, pe_target=vocabulary_size, rate=dropout_rate)

# define custom schedule
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

# define optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='val_accuracy')

train_loss_plot = []
train_acc_plot = []

val_loss_plot = []
val_acc_plot = []

# create checkpoints
try:
    os.makedirs(os.path.join('checkpoints', 'trans'))
except FileExistsError:
    pass
checkpoint_path = "./checkpoints/trans"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print('Latest checkpoint restored!!')

# define custom training
@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


@tf.function
def test_step(val_inp, val_tar):
    val_tar_inp = val_tar[:, :-1]
    val_tar_real = val_tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(val_inp, val_tar_inp)
    val_predictions, _ = transformer(val_inp, val_tar_inp, False, enc_padding_mask,
                                     combined_mask, dec_padding_mask)
    v_loss = loss_function(val_tar_real, val_predictions)

    val_loss(v_loss)
    val_accuracy(val_tar_real, val_predictions)



EPOCHS = 20
patience = 5
wait = 0
best = 0

for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))
    start_time = time.time()

    # Iterate over the batches of the dataset.
    for batch, (inp, tar) in enumerate(train_dataset):
        train_step(inp, tar)

        # Log every 1000 batches.
        if batch % 1000 == 0:
            print("Training loss (for one batch) at batch %d: %.4f" % (batch, float(train_loss.result())))
            print("Training acc (for one batch) at batch %d: %.4f" % (batch, float(train_accuracy.result())))

    
    # Display metrics at the end of each epoch.
    #train_loss = train_loss.result()
    print("Training loss over epoch: %.4f" % (float(train_loss.result()),))

    #train_acc = train_accuracy.result()
    print("Training acc over epoch: %.4f" % (float(train_accuracy.result()),))

    train_loss_plot.append(train_loss.result())
    train_acc_plot.append(train_accuracy.result())

    # Reset training metrics at the end of each epoch
    train_loss.reset_states()
    train_accuracy.reset_states()

    # Run a validation loop at the end of each epoch.
    for val_inp, val_tar in val_dataset:
        test_step(val_inp, val_tar)

    #val_loss = val_loss.result()
    #val_acc = val_accuracy.result()
    
    print("Validation loss: %.4f" % (float(val_loss.result()),))
    print("Validation acc: %.4f" % (float(val_accuracy.result()),))

    val_loss_plot.append(val_loss.result())
    val_acc_plot.append(val_accuracy.result())

    if (epoch + 1) % 5 == 0:
      checkpoint_path = ckpt_manager.save()
      print(f'Saving checkpoint for epoch {epoch+1} at {checkpoint_path}')

    print("Time taken: %.2fs" % (time.time() - start_time))
    val_loss.reset_states()
    val_accuracy.reset_states()

     # early stopping 
    wait += 1
    if val_loss > best:
      best = val_loss
      wait = 0
    if wait >= patience:
      break


plt.plot(train_loss_plot)
plt.plot(val_loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Model Loss')
plt.legend(['training', 'validation'], loc='upper right')
plt.grid()
plt.show()






