from __future__ import absolute_import, division, print_function
from tqdm import tqdm
from numpy.random import randn

import pathlib
import random
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from matplotlib.image import imread

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

AUTOTUNE = tf.data.experimental.AUTOTUNE


data_dir = tf.keras.utils.get_file('flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz', untar=True)
data_dir = pathlib.Path(data_dir)


all_images = list(data_dir.glob('*/*'))
all_images = [str(path) for path in all_images]
random.shuffle(all_images)

image_count = len(all_images)

label_names={'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}


def _process_image(path):
    image = open(path, 'rb').read()

    text = pathlib.Path(filename).parent.name
    label = label_names[text]

    return image, text, label


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(image_buffer, label, text):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': _int64_feature(label),
        'text': _bytes_feature(tf.compat.as_bytes(text)),
        'encoded': _bytes_feature(tf.compat.as_bytes(image_buffer))}))
    return example

with tf.io.TFRecordWriter('flower.tfrecords') as writer:
  for filename in tqdm(all_images):
    image_buffer,text,label = _process_image(filename)
    example = _convert_to_example(image_buffer, label,text)
    writer.write(example.SerializeToString())

image_dataset = tf.data.TFRecordDataset('flower.tfrecords')

IMG_SIZE = 224
# Create a dictionary describing the features.
image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'text': tf.io.FixedLenFeature([], tf.string),
    'encoded': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    # Parse the input tf.Example proto using the dictionary above.
    feature = tf.io.parse_single_example(example_proto, image_feature_description)

    image = feature['encoded']
    image = tf.image.decode_jpeg(image, channels=3)
    #   image = tf.image.resize_images(image, [224, 224])
    #   image /= 255.0  # normalize to [0,1] range

    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))

    return image, feature['label']


dataset = image_dataset.map(_parse_image_function)

BATCH_SIZE = 1

ds = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=image_count))
ds = dataset.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

# for image, label, text in ds.take(1):
#   # plt.title(text.numpy())
#   plt.imshow(image)


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
VGG16_MODEL=tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
VGG16_MODEL.trainable=False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(len(label_names),activation=tf.nn.softmax)

VGG16_MODEL.summary()

model = tf.keras.Sequential([
  VGG16_MODEL,
  global_average_layer,
  prediction_layer
])

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(ds,
                    steps_per_epoch=5,
                    epochs=20)

