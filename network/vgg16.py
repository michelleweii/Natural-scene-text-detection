from __future__ import print_function
from keras.layers import Dense, Dropout, Flatten, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D, add
from keras.models import Model


def network_cnn(inputs):
    # inputs = Input(shape=x_train.shape[1:])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x = Conv2D(1, (1, 1), activation='relu', padding='same', name='bottleneck_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Classification block
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model.
    # model = Model(inputs, x, name='vgg16')
    return x


def network_cnn_16(inputs):
    # inputs = Input(shape=x_train.shape[1:])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x = Conv2D(1, (1, 1), activation='relu', padding='same', name='bottleneck_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Classification block
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model.
    # model = Model(inputs, x, name='vgg16')
    return x


def network_cnn_8(inputs):
    # inputs = Input(shape=x_train.shape[1:])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x = Conv2D(1, (1, 1), activation='relu', padding='same', name='bottleneck_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Classification block
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model.
    # model = Model(inputs, x, name='vgg16')
    return x


def network_cnn_4(inputs):
    # inputs = Input(shape=x_train.shape[1:])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x = Conv2D(1, (1, 1), activation='relu', padding='same', name='bottleneck_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Classification block
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model.
    # model = Model(inputs, x, name='vgg16')
    return x


def network_cnn_8_16(inputs):
    # inputs = Input(shape=x_train.shape[1:])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x = Conv2D(1, (1, 1), activation='relu', padding='same', name='bottleneck_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Classification block
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model.
    # model = Model(inputs, x, name='vgg16')
    return x


def network_cnn_8_16_dilated(inputs):
    # inputs = Input(shape=x_train.shape[1:])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((1, 2), strides=(1, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), dilation_rate=(1, 3), activation='relu', padding='same', name='block3_conv1',
               kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), dilation_rate=(1, 4), activation='relu', padding='same', name='block3_conv2',
               kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), dilation_rate=(1, 5), activation='relu', padding='same', name='block3_conv3',
               kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), dilation_rate=(1, 3), activation='relu', padding='same', name='block4_conv1',
               kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(1, 4), activation='relu', padding='same', name='block4_conv2',
               kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(1, 5), activation='relu', padding='same', name='block4_conv3',
               kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), dilation_rate=(1, 3), activation='relu', padding='same', name='block5_conv1',
               kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(1, 4), activation='relu', padding='same', name='block5_conv2',
               kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(1, 5), activation='relu', padding='same', name='block5_conv3',
               kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x = Conv2D(1, (1, 1), activation='relu', padding='same', name='bottleneck_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Classification block
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model.
    # model = Model(inputs, x, name='vgg16')
    return x


def network_cnn_8_8_dilated(inputs):
    # inputs = Input(shape=x_train.shape[1:])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), dilation_rate=(1, 3), activation='relu', padding='same', name='block3_conv1',
               kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), dilation_rate=(1, 4), activation='relu', padding='same', name='block3_conv2',
               kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), dilation_rate=(1, 5), activation='relu', padding='same', name='block3_conv3',
               kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), dilation_rate=(1, 3), activation='relu', padding='same', name='block4_conv1',
               kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(1, 4), activation='relu', padding='same', name='block4_conv2',
               kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(1, 5), activation='relu', padding='same', name='block4_conv3',
               kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), dilation_rate=(1, 3), activation='relu', padding='same', name='block5_conv1',
               kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(1, 4), activation='relu', padding='same', name='block5_conv2',
               kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), dilation_rate=(1, 5), activation='relu', padding='same', name='block5_conv3',
               kernel_initializer='he_normal')(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # x = Conv2D(1, (1, 1), activation='relu', padding='same', name='bottleneck_1', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Classification block
    # x = Flatten(name='flatten')(x)
    # x = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model.
    # model = Model(inputs, x, name='vgg16')
    return x


def network_fcn_8(inputs):
    # inputs = Input(shape=x_train.shape[1:])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal')(x)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal')(x)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x5)

    # x = Conv2D(1, (1, 1), activation='relu', padding='same', name='bottleneck_1', kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    # Classification block
    # x = Flatten(name='flatten')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2', kernel_initializer='he_normal')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal')(x)
    x = add([x, x5])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal')(x)
    x = add([x, x4])
    # x = UpSampling2D(size=(2, 2))(x)
    # x = Conv2D(256, (1, 1), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal')(x)
    # # x3 = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal')(x3)
    # x = add([x, x3])
    x = BatchNormalization()(x)

    # x = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model.
    # model = Model(inputs, x, name='vgg16')
    return x



def network_fcn_4(inputs):
    # inputs = Input(shape=x_train.shape[1:])
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', kernel_initializer='he_normal')(
        inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', kernel_initializer='he_normal')(x)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', kernel_initializer='he_normal')(x)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x4)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', kernel_initializer='he_normal')(x)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x5)

    # x = Conv2D(1, (1, 1), activation='relu', padding='same', name='bottleneck_1', kernel_initializer='he_normal')(x)
    # x = BatchNormalization()(x)
    # Classification block
    # x = Flatten(name='flatten')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv1', kernel_initializer='he_normal')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block6_conv2', kernel_initializer='he_normal')(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv1', kernel_initializer='he_normal')(x)
    x = add([x, x5])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv2', kernel_initializer='he_normal')(x)
    x = add([x, x4])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(256, (1, 1), activation='relu', padding='same', name='deconv3', kernel_initializer='he_normal')(x)
    # x3 = Conv2D(512, (1, 1), activation='relu', padding='same', name='deconv4', kernel_initializer='he_normal')(x3)
    x = add([x, x3])

    x = BatchNormalization()(x)

    # x = Dense(4096, activation='relu', name='fc1', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(4096, activation='relu', name='fc2', kernel_initializer='he_normal')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(10, activation='softmax', name='predictions')(x)

    # Create model.
    # model = Model(inputs, x, name='vgg16')
    return x
