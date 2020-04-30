import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

import os
from matplotlib.image import imread

import numpy as np

import math
print(tf.__version__)
AUTO = tf.data.experimental.AUTOTUNE # used in tf.data.Dataset API

nb_images=len(tf.io.gfile.glob('C:/Users/masou/Downloads/train/train/*.jpg'))
SHARDS = 16
shared_size = math.ceil(1.0 * nb_images / SHARDS)
print(shared_size)

TARGET_SIZE=160
CLASSES = [b'cat', b'dog'] # do not change, maps to the labels in the data (folder names)


def read_image_and_label(img_path):
    bits = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(bits)
    #   image = tf.image.resize_images(image, [TARGET_SIZE, TARGET_SIZE])

    label = tf.strings.split(img_path, sep='/')
    label = tf.strings.split(label[-1], sep='.')

    return image, label[0]


dataset = tf.data.Dataset.list_files('C:/Users/masou/Downloads/train/train/*.jpg', seed=10000)  # This also shuffles the images
dataset = dataset.map(read_image_and_label)


def resize_and_crop_image(image, label):
    w = tf.shape(image)[0]
    h = tf.shape(image)[1]
    tw = TARGET_SIZE
    th = TARGET_SIZE
    resize_crit = (w * th) / (h * tw)
    image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(image, [w * tw / w, h * tw / w]),  # if true
                    lambda: tf.image.resize(image, [w * th / h, h * th / h])  # if false
                    )
    nw = tf.shape(image)[0]
    nh = tf.shape(image)[1]
    image = tf.image.crop_to_bounding_box(image, (nw - tw) // 2, (nh - th) // 2, tw, th)
    return image, label


dataset = dataset.map(resize_and_crop_image, num_parallel_calls=AUTO)

def recompress_image(image, label):
  image = tf.cast(image, tf.uint8)
  image = tf.image.encode_jpeg(image, optimize_size=True, chroma_downsampling=False)
  return image, label

dataset = dataset.map(recompress_image, num_parallel_calls=AUTO)
dataset = dataset.batch(shared_size)

def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))

def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))

def _float_feature(list_of_floats): # float32a
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(img_bytes, label):
    class_num = np.argmax(np.array(CLASSES) == label)
    feature = {
        "image": _bytestring_feature([img_bytes]),  # one image in the list
        "class": _int_feature([class_num]),  # one class in the list
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


for shard, (image, label) in enumerate(dataset):
    shard_size = image.numpy().shape[0]
    filename = "cat_dog" + "{:02d}-{}.tfrec".format(shard, shard_size)

    with tf.io.TFRecordWriter(filename) as out_file:
        for i in range(shard_size):
            example = to_tfrecord(image.numpy()[i], label.numpy()[i])
            out_file.write(example.SerializeToString())
        print("Wrote file {} containing {} records".format(filename, shard_size))

import glob
import os, json, random

VALIDATION_SPLIT = 0.19

BATCH_SIZE = 32

filenames=tf.io.gfile.glob('cat*.tfrec')

random.shuffle(filenames)
split = int(len(filenames) * VALIDATION_SPLIT)

training_filenames = filenames[split:]
validation_filenames = filenames[:split]

validation_steps = int(3670 // len(filenames) * len(validation_filenames)) // BATCH_SIZE
steps_per_epoch = int(3670 // len(filenames) * len(training_filenames)) // BATCH_SIZE


def read_tfrecord(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [TARGET_SIZE, TARGET_SIZE, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label


def get_batched_dataset(filenames):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)

    dataset = dataset.cache()  # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(AUTO)  #

    return dataset


def get_training_dataset():
    return get_batched_dataset(training_filenames)


def get_validation_dataset():
    return get_batched_dataset(validation_filenames)


model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu',
                           input_shape=[TARGET_SIZE, TARGET_SIZE, 3]),
    tf.keras.layers.Conv2D(kernel_size=3, filters=32, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(kernel_size=3, filters=64, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(kernel_size=3, filters=128, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=2),

    tf.keras.layers.Conv2D(kernel_size=3, filters=256, padding='same', activation='relu'),

    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(1, 'sigmoid')
])

model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(get_training_dataset(), steps_per_epoch=steps_per_epoch, epochs=10,
                      validation_data=get_validation_dataset(), validation_steps=validation_steps)