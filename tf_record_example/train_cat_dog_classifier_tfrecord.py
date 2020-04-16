import numpy as np
import os
import tensorflow as tf
import math
import glob
import json
import random
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.image import imread


def read_image_and_label(_image_path):

    bits = tf.io.read_file(_image_path)
    image = tf.image.decode_jpeg(bits)

    label = tf.strings.split(_image_path, sep='/')
    label = tf.strings.split(label[-1], sep='.')

    return image, label[0]


def resize_and_crop_image(_image, _label, _target_size):

    w = tf.shape(_image)[0]
    h = tf.shape(_image)[1]
    tw = _target_size
    th = _target_size
    resize_crit = (w * th) / (h * tw)
    _image = tf.cond(resize_crit < 1,
                    lambda: tf.image.resize(_image, [w * tw / w, h * tw / w]),  # if true
                    lambda: tf.image.resize(_image, [w * th / h, h * th / h])  # if false
                    )
    nw = tf.shape(_image)[0]
    nh = tf.shape(_image)[1]
    _image = tf.image.crop_to_bounding_box(_image, (nw - tw) // 2, (nh - th) // 2, tw, th)

    return _image, _label


def recompress_image(_image, _label):

    _image = tf.cast(_image, tf.uint8)
    _image = tf.image.encode_jpeg(_image, optimize_size=True, chroma_downsampling=False)

    return _image, _label


def load_image_dataset(_image_path, _target_size):

    images, labels = read_image_and_label(_image_path=_image_path)
    images, labels = resize_and_crop_image(_image=images, _label=labels, _target_size=_target_size)
    images, labels = recompress_image(_image=images, _label=labels)

    return images, labels


def _bytestring_feature(list_of_bytestrings):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _int_feature(list_of_ints): # int64
  return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def _float_feature(list_of_floats): # float32
  return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(img_bytes, label, _classes):

    class_num = np.argmax(np.array(_classes) == label)
    feature = {
        "image": _bytestring_feature([img_bytes]),  # one image in the list
        "class": _int_feature([class_num]),  # one class in the list
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_tfrecord_files(_dataset, _classes, _path):

    try:
        path_tfrecord = _path + 'tfrecord_files'
        os.makedirs(path_tfrecord)
    except:
        pass

    for shard, (image, label) in enumerate(_dataset):
        shard_size = image.numpy().shape[0]
        filename = f"cat_dog{shard:02d}-{shard_size}.tfrecords"
        full_filename = os.path.join(path_tfrecord, filename)
        if not os.path.isfile(full_filename):

            with tf.io.TFRecordWriter(full_filename) as out_file:
                for i in range(shard_size):
                    example = to_tfrecord(image.numpy()[i], label.numpy()[i], _classes=_classes)
                    out_file.write(example.SerializeToString())
                print(f"Wrote file: {full_filename} containing {shard_size} records")
        else:
            print(f"file: {full_filename} containing {shard_size} records, already exist")


def read_tfrecord(example, _target_size):

    features = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string = bytestring (not text string)
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
    }
    # decode the TFRecord
    example = tf.io.parse_single_example(example, features)

    image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [_target_size, _target_size, 3])

    class_label = tf.cast(example['class'], tf.int32)

    return image, class_label


def get_batched_dataset(_filenames, _target_size, _batch_size, _num_parallel_calls=None):

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.Dataset.list_files(_filenames)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=_num_parallel_calls)
    dataset = dataset.map(lambda x: read_tfrecord(x, _target_size=_target_size), num_parallel_calls=_num_parallel_calls)

    dataset = dataset.take(200).cache()  # This dataset fits in RAM
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(_batch_size, drop_remainder=False)  # drop_remainder will be needed on TPU
    dataset = dataset.prefetch(_num_parallel_calls)  #

    return dataset


def get_training_dataset(_training_filenames, _batch_size, _target_size, _num_parallel_calls=None):

    dataset = get_batched_dataset(_filenames=_training_filenames,
                                  _target_size=_target_size,
                                  _batch_size=_batch_size,
                                  _num_parallel_calls=_num_parallel_calls)

    return dataset


def get_validation_dataset(_validation_filenames, _target_size, _batch_size, _num_parallel_calls=None):

    dataset = get_batched_dataset(_filenames=_validation_filenames,
                                  _target_size=_target_size,
                                  _batch_size=_batch_size,
                                  _num_parallel_calls=_num_parallel_calls)

    return dataset


def create_model(_target_size):

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu',
                               input_shape=[_target_size, _target_size, 3]),
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

    return model


def predict_model(model):

    fig = plt.figure(figsize=(12, 28))

    cnt = 1

    ds = get_validation_dataset()

    for imgs, lbls in ds.take(1):
        predicted_classes = model.predict_classes(imgs)
        for img, lbl, cl in zip(imgs, lbls, predicted_classes):
            fig.add_subplot(8, 4, cnt)
            plt.title('{}/{}'.format(cl[0], lbl))
            plt.imshow(img)
            cnt = cnt + 1


def main():

    print(tf.version)
    AUTO = tf.data.experimental.AUTOTUNE  # used in tf.data.Dataset API
    SHARDS = 64
    TARGET_SIZE = 160
    path_train_image = 'C:/Users/masou/Downloads/train/train/'
    VALIDATION_SPLIT = 0.19
    BATCH_SIZE = 16
    EPOCHS = 10

    num_images = len(tf.io.gfile.glob(f'{path_train_image}*.jpg'))
    shared_size = math.ceil(1.0 * num_images / SHARDS)
    print(f"Shared size for {num_images} number of images is {shared_size}")
    CLASSES = [b'cat', b'dog']  # do not change, maps to the labels in the data (folder names)

    # load images and resize it to the target size
    dataset = tf.data.Dataset.list_files(f'{path_train_image}*.jpg', seed=10000)  # This also shuffles the images
    dataset = dataset.map(lambda x: load_image_dataset(x, _target_size=TARGET_SIZE), num_parallel_calls=AUTO)
    dataset = dataset.batch(shared_size)

    # Next, we should stuff data in a protocol buffer called Example.
    # Example protocol buffer contains Features.
    # The feature is a protocol to describe the data and could have three types: bytes, float, and int64.
    create_tfrecord_files(_dataset=dataset, _classes=CLASSES, _path=path_train_image)

    filenames = tf.io.gfile.glob(f'{path_train_image}tfrecord_files/cat*.tfrecords')
    random.shuffle(filenames)
    split = int(len(filenames) * VALIDATION_SPLIT)

    training_filenames = filenames[split:]
    validation_filenames = filenames[:split]
    validation_steps = int(3670 // len(filenames) * len(validation_filenames)) // BATCH_SIZE
    steps_per_epoch = int(3670 // len(filenames) * len(training_filenames)) // BATCH_SIZE

    model = create_model(_target_size=TARGET_SIZE)
    model.compile(optimizer='Adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(get_training_dataset(_training_filenames=training_filenames,
                                             _target_size=TARGET_SIZE,
                                             _batch_size=BATCH_SIZE,
                                             _num_parallel_calls=AUTO),
                        steps_per_epoch=steps_per_epoch,
                        epochs=EPOCHS,
                        validation_data=get_validation_dataset(_validation_filenames=validation_filenames,
                                                               _target_size=TARGET_SIZE,
                                                               _batch_size=BATCH_SIZE,
                                                               _num_parallel_calls=AUTO),
                        validation_steps=validation_steps)


if __name__ == '__main__':
    main()







