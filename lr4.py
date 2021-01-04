import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from datetime import datetime
import tensorflow as tf
from pathlib import Path
from keras.layers import Conv2D, InputLayer, Conv2DTranspose
from tensorflow.keras import layers
from keras.models import Sequential
import tensorflow_io as tfio
import numpy as np


BATCH_SIZE = 512
#NUM_CLASSES = 6
#PARALLEL_CALLS = 4
RESIZE_TO = 224
#TRAINSET_SIZE = 14034
TRAIN_FOLDER = 'various_tagged_images_67k_tf'
# TRAIN_FOLDER = 'video_capture_5frame_1000_shard_tf'
# TRAIN_FOLDER = 'validation_tf'
# VALIDATION_FOLDER = 'val'
VALIDATION_FOLDER = 'validation_shard_tf'

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)


def generator_train():
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomCrop(112, 112),
        layers.experimental.preprocessing.Resizing(224, 224),
        layers.experimental.preprocessing.RandomRotation(factor=0.55),
        layers.experimental.preprocessing.RandomFlip(mode='horizontal')
    ])

    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(current_dir + f"/{TRAIN_FOLDER}")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    i = 0
    for fn in file_list_train:
        for dataset in create_dataset(fn, BATCH_SIZE):
            dataset = data_augmentation(dataset)
            #x = tf.image.grayscale_to_rgb(dataset[:, :, :, 0])
            #x = tf.keras.layers.Concatenate(axis=2)([dataset[:, :, :, 1], dataset[:, :, :, 1]])
            #x = tf.keras.layers.Concatenate(axis=2)([x, dataset[:, :, :, 1]])
            #x = np.reshape(x, (-1, RESIZE_TO, RESIZE_TO, 3))
            x = np.reshape(dataset[:, :, :, 0], (-1, RESIZE_TO, RESIZE_TO, 1))
            # x = tf.image.grayscale_to_rgb(x)
            y = np.reshape(dataset[:, :, :, 1:], (-1, RESIZE_TO, RESIZE_TO, 2))
            i = i + 1
            # print(f'\nIteration: {i}. Train')
            yield (x, y)


def generator_valid():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(current_dir + f"/{VALIDATION_FOLDER}")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    i = 0
    for fn in file_list_train:
        for dataset in create_dataset(fn, BATCH_SIZE):
            #x = tf.image.grayscale_to_rgb(dataset[:, :, :, 0])
            #x = tf.keras.layers.Concatenate(axis=2)([dataset[:, :, :, 1], dataset[:, :, :, 1]])
            #x = tf.keras.layers.Concatenate(axis=2)([x, dataset[:, :, :, 1]])
            #x = np.reshape(x, (-1, RESIZE_TO, RESIZE_TO, 3))
            x = np.reshape(dataset[:, :, :, 0], (-1, RESIZE_TO, RESIZE_TO, 1))
            # x = tf.image.grayscale_to_rgb(x)
            y = np.reshape(dataset[:, :, :, 1:], (-1, RESIZE_TO, RESIZE_TO, 2))
            i = i + 1
            # print(f'\nIteration {i}. Validation')
            yield (x, y)


def visualize_images(epoch, model, dataset, writer):
    item = iter(dataset).next()

    l_channel = item[0]
    target_ab = item[1]
    target_image = np.zeros((l_channel.shape[0], l_channel.shape[1], l_channel.shape[2], 3))
    target_image[:, :, :, 1:] = target_ab
    target_image[:, :, :, 0] = np.reshape(l_channel, (-1, 224, 224))

    predicted_ab = model(np.reshape(l_channel, (-1, 224, 224, 1)))
    predicted_image = np.zeros((l_channel.shape[0], l_channel.shape[1], l_channel.shape[2], 3))
    predicted_image[:, :, :, 0] = np.reshape(l_channel, (-1, 224, 224))
    predicted_image[:, :, :, 1:] = predicted_ab

    with writer.as_default():
        tf.summary.image('Target Lab', np.reshape(target_image, (-1, 224, 224, 3)), step=epoch)
        tf.summary.image('Result Lab', np.reshape(predicted_image, (-1, 224, 224, 3)), step=epoch)


def visualize_images_augmented(epoch, dataset, writer):
    item = iter(dataset).next()
    #item = dataset[2]
    l_channel = item[0]
    target_ab = item[1]
    target_image = np.zeros((l_channel.shape[0], l_channel.shape[1], l_channel.shape[2], 3))
    target_image[:, :, :, 1:] = target_ab
    target_image[:, :, :, 0] = np.reshape(l_channel, (-1, 224, 224))

    with writer.as_default():
        tf.summary.image('Augmented', target_image, step=epoch)


def parse_proto_example(proto):
    keys_to_features = {
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value='')
    }
    example = tf.io.parse_single_example(proto, keys_to_features)
    example['image'] = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    example['image'] = tf.image.convert_image_dtype(example['image'], dtype=tf.float32)
    example['image'] = tf.image.resize(example['image'], tf.constant([RESIZE_TO, RESIZE_TO]))
    return example['image']


def create_dataset(filenames, batch_size):
    """Create dataset from tfrecords file
    :tfrecords_files: Mask to collect tfrecords file of dataset
    :returns: tf.data.Dataset
    """
    return tf.data.TFRecordDataset(filenames) \
        .map(parse_proto_example) \
        .batch(batch_size) \
        .prefetch(batch_size)


def display_image_count():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    train_dir = Path(current_dir + f"/{TRAIN_FOLDER}")
    file_list_train = [str(pp) for pp in train_dir.glob("*")]
    #file_list_train = tf.random.shuffle(file_list_train)
    c = 0
    for fn in file_list_train:
        for record in tf.data.TFRecordDataset(fn):
            c += 1
   # c = len(file_list_train)
    print(f'Count of train images: {c}')

    valid_dir = Path(current_dir + f"/{VALIDATION_FOLDER}")
    file_list_valid = [str(pp) for pp in valid_dir.glob("*")]
    #file_list_valid = tf.random.shuffle(file_list_valid)
    v = 0
    for fn in file_list_valid:
        for record in tf.data.TFRecordDataset(fn):
            v += 1
    #v = len(file_list_valid)
    print(f'Count of validation images: {v}')
    return c, v

def build_model():
    INPUT_SHAPE = (224, 224, 3)
    model = Sequential()
    model.add(InputLayer(input_shape=INPUT_SHAPE))
    model.add(tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1))
    model.add(tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE,
                                               include_top=False,
                                               weights='imagenet', training=False))
    # output shape (None, 7, 7, 1280)
    model.add(Conv2DTranspose(128, 5, strides=4, activation='relu', padding='same'))
    model.add(Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same'))
    model.add(Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same'))
    model.add(Conv2DTranspose(2, 3, strides=2, activation='relu', padding='same'))

    return model


def main():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    log_dir = current_dir + '/log_lab3/' + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)

    display_image_count()

    train = tf.data.Dataset.from_generator(
        generator_train,
        (tf.float32, tf.float32))

    valid = tf.data.Dataset.from_generator(
        generator_valid,
        (tf.float32, tf.float32))
    IMG_SHAPE = (RESIZE_TO, RESIZE_TO, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False

    #prediction_layer = tf.keras.layers.Dense(2)
    inputs = tf.keras.Input(shape=(RESIZE_TO, RESIZE_TO, 1))
    x = tf.image.grayscale_to_rgb(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.Conv2DTranspose(128, 5, strides=4, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2DTranspose(2, 3, strides=2, activation='relu', padding='same')(x)
    #outputs = prediction_layer(x)
    outputs = x
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.optimizers.SGD(lr=0.1, momentum=0.9),
        loss=tf.keras.losses.mean_squared_error
    )

    print(model.summary())

    model.fit(
        train,
        epochs=50,
        validation_data=valid,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir),
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: visualize_images(epoch, model, valid, file_writer)
            ),
            # tf.keras.callbacks.LambdaCallback(
            #     on_epoch_end=lambda epoch, logs: visualize_images_augmented(epoch, train, file_writer)
            # )
        ]
    )




if __name__ == '__main__':
    main()
