from keras.layers import Dense, Dropout, BatchNormalization
from keras.layers import Input, core, Lambda, Activation, Conv2D
from keras import backend as K


def network_classification(inputs):
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='cls_out')(x)
    return x

def network_centerness(inputs):
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='cnt_out')(x)
    return x


def network_regression(inputs):
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(4, activation='sigmoid', kernel_initializer='he_normal', name='reg_out')(x)
    return x


def cls_slice(x, index):
    return x[:, :, :, :index]

def cnt_slice(x, index):
    return x[:, :, :, index]

def bbox_slice(x, index):
    return x[:, :, :, index:]


# 去除全连接层
def network_out(inputs, act):
    inputs = Conv2D(10, (1, 1), activation='relu', padding='same', name='bottleneck_1',
                    kernel_initializer='he_normal')(inputs)
    # cls 分支
    cls = Lambda(cls_slice, arguments={'index': 2}, name='slice_cls')(inputs)
    # cls = BatchNormalization()(cls)
    # cls = Conv2D(11, (1, 3), activation='relu', padding='same', name='bottleneck_cls',
    #            kernel_initializer='he_normal')(cls)
    # cls = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_cls')(cls)
    # cls = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_cls')(cls)
    cls = BatchNormalization()(cls)
    cls = Activation('softmax', name='cls_out')(cls)

    # bbox 分支
    bbox = Lambda(bbox_slice, arguments={'index': 2}, name='slice_bbox')(inputs)
    bbox = BatchNormalization()(bbox)
    bbox = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_bbox',
                  kernel_initializer='he_normal')(bbox)
    # bbox = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_bbox')(bbox)
    # bbox = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_bbox')(bbox)
    bbox = BatchNormalization()(bbox)
    bbox = Activation(act, name='reg_out')(bbox)

    return cls, bbox


def network_out_no_FC(inputs):
    # cls 分支
    cls = Lambda(cls_slice, arguments={'index': 1}, name='slice_cls')(inputs)
    # cls = BatchNormalization()(cls)
    # cls = Conv2D(11, (1, 3), activation='relu', padding='same', name='bottleneck_cls',
    #            kernel_initializer='he_normal')(cls)
    # cls = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_cls')(cls)
    # cls = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_cls')(cls)
    cls = BatchNormalization()(cls)
    cls = Activation('sigmoid', name='cls_out')(cls)

    # centerness 分支
    cnt = Lambda(cnt_slice, arguments={'index': 1}, name='slice_cnt')(inputs)
    # cls = BatchNormalization()(cls)
    # cls = Conv2D(11, (1, 3), activation='relu', padding='same', name='bottleneck_cls',
    #            kernel_initializer='he_normal')(cls)
    # cls = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_cls')(cls)
    # cls = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_cls')(cls)
    cnt = BatchNormalization()(cnt)
    cnt = Activation('sigmoid', name='cnt_out')(cnt)

    # bbox 分支
    bbox = Lambda(bbox_slice, arguments={'index': 2}, name='slice_bbox')(inputs)
    bbox = BatchNormalization()(bbox)
    # bbox = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_bbox',
    #               kernel_initializer='he_normal')(bbox)
    # bbox = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_bbox')(bbox)
    # bbox = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_bbox')(bbox)
    # bbox = BatchNormalization()(bbox)
    bbox = Activation('sigmoid', name='reg_out')(bbox)

    return cls, cnt, bbox
