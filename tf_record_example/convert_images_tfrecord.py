import numpy as np
import glob
import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(image, label, image_shape):
    feature = {
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
        'height': _int64_feature(image_shape[0]),
        'width': _int64_feature(image_shape[1]),
        'depth': _int64_feature(image_shape[2]),
         }

    #  Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord(_tfrecord_path, _image_paths, _labels):

    with tf.io.TFRecordWriter(_tfrecord_path) as writer:

        for image_path, label in zip(_image_paths, _labels):
            print(f"wirting image {image_path} in tf record")
            img = tf.keras.preprocessing.image.load_img(image_path)
            img_array = tf.keras.preprocessing.image.img_to_array(img)

            img_array = tf.keras.preprocessing.image.random_zoom(img_array, (0.5, 0.5),
                                                                 row_axis=0,
                                                                 col_axis=1,
                                                                 channel_axis=2)

            img_bytes = tf.io.serialize_tensor(img_array)
            image_shape = img_array.shape

            example = serialize_example(img_bytes, label, image_shape)
            writer.write(example)


def read_tfrecord(serialized_example):
    feature_description = {
        'image': tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64)
         }

    example = tf.io.parse_single_example(serialized_example, feature_description)

    image = tf.io.parse_tensor(example['image'], out_type=float)
    image_shape = [example['height'], example['width'], example['depth']]
    image = tf.reshape(image, image_shape)

    return image, example['label']


def main():

    # parse the input argument
    parser = argparse.ArgumentParser(description="Parsing input argument")
    parser.add_argument("-i", "--input_dir", required=True, help="input argument of images")
    parser.add_argument("-o", "--output_dir", required=False, default="tf_record",
                        help="output directory for saving tfrecord file")
    args = parser.parse_args()

    data_dir = args.input_dir
    image_paths = glob.glob(data_dir + '/*.jpg')

    # make some labels
    num_image_category = len(image_paths)//2
    labels = np.append(np.zeros(num_image_category, dtype=int), np.ones(num_image_category, dtype=int))

    # make a directory to store tfrecord files
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # make tfrecord files
    tf_record_filename = os.path.join(args.output_dir, "tfrecord_data.tfrecords")
    write_tfrecord(_tfrecord_path=tf_record_filename,
                   _image_paths=image_paths,
                   _labels=labels)

    # read tfrecord files
    tfrecord_dataset = tf.data.TFRecordDataset(tf_record_filename)
    parsed_dataset = tfrecord_dataset.map(read_tfrecord)

    # some some sample images loaded from tfrecord
    plt.figure(figsize=(10, 10))
    for i, data in enumerate(parsed_dataset.take(9)):
        image = tf.keras.preprocessing.image.array_to_img(data[0])
        plt.subplot(3, 3, i + 1)
        plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    main()