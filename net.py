import numpy as np
np.random.seed(2016)
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
# from keras.layers import merge
from keras.layers import concatenate
from keras.optimizers import Adam


def create_fcn(input_size):
    # tensorflowに書き換え
    inputs = Input((input_size[0], input_size[1], 1))

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv5) # 元は(2, 2)

    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format="channels_last")(pool5)
    conv6 = Conv2D(1024, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv6), conv5], axis=3) # 元は(2, 2)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(up7)
    conv7 = Conv2D(512, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv7), conv4], axis=3)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(up8)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv8), conv3], axis=3)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(up9)
    conv9 = Conv2D(128, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv9)

    up10 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv9), conv2], axis=3)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(up10)
    conv10 = Conv2D(64, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv10)

    up11 = concatenate([UpSampling2D(size=(2, 2), data_format="channels_last")(conv10), conv1], axis=3)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(up11)
    conv11 = Conv2D(32, (3, 3), activation='relu', padding='same', data_format="channels_last")(conv11)

    conv12 = Conv2D(3, (1, 1), activation='tanh', data_format="channels_last")(conv11)

    fcn = Model(input=inputs, output=conv12)

    return fcn