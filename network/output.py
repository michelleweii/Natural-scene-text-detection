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


def network_classification_location(inputs):
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(5, activation='softmax', kernel_initializer='he_normal', name='loc_out')(x)
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


def network_regression_no_scaler(inputs):
    x = Dense(2048, activation='relu', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    # x = Dropout(0.5)(x)
    x = Dense(4, kernel_initializer='he_normal', name='reg_out')(x)
    return x


def cls_slice(x, index):
    return x[:, :, :, :index]


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

    # bbox 分支
    bbox = Lambda(bbox_slice, arguments={'index': 1}, name='slice_bbox')(inputs)
    bbox = BatchNormalization()(bbox)
    # bbox = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_bbox',
    #               kernel_initializer='he_normal')(bbox)
    # bbox = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_bbox')(bbox)
    # bbox = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_bbox')(bbox)
    # bbox = BatchNormalization()(bbox)
    bbox = Activation('sigmoid', name='reg_out')(bbox)

    return cls, bbox


def network_out_no_FC_scaler(inputs):
    # cls 分支
    cls = Lambda(cls_slice, arguments={'index': 1}, name='slice_cls')(inputs)
    # cls = BatchNormalization()(cls)
    # cls = Conv2D(11, (1, 3), activation='relu', padding='same', name='bottleneck_cls',
    #            kernel_initializer='he_normal')(cls)
    # cls = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_cls')(cls)
    # cls = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_cls')(cls)
    cls = BatchNormalization()(cls)
    cls = Activation('sigmoid', name='cls_out')(cls)

    # bbox 分支
    bbox = Lambda(bbox_slice, arguments={'index': 1}, name='slice_bbox')(inputs)
    # bbox = BatchNormalization()(bbox)
    # bbox = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_bbox',
    #               kernel_initializer='he_normal')(bbox)
    # bbox = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_bbox')(bbox)
    # bbox = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_bbox')(bbox)
    # bbox = BatchNormalization()(bbox)
    # bbox = Activation('sigmoid', name='reg_out')(bbox)

    return cls, bbox

def network_out_location(inputs):
    # cls 分支
    cls = Lambda(cls_slice, arguments={'index': 1}, name='slice_cls')(inputs)
    # cls = BatchNormalization()(cls)
    # cls = Conv2D(11, (1, 3), activation='relu', padding='same', name='bottleneck_cls',
    #            kernel_initializer='he_normal')(cls)
    # cls = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_cls')(cls)
    # cls = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_cls')(cls)
    cls = BatchNormalization()(cls)
    cls = Activation('sigmoid', name='cls_out')(cls)

    # location 分支
    loc = Lambda(bbox_slice, arguments={'index': 1}, name='slice_bbox')(inputs)
    loc = BatchNormalization()(loc)
    # bbox = Conv2D(8, (3, 3), activation='relu', padding='same', name='conv_bbox',
    #               kernel_initializer='he_normal')(bbox)
    # bbox = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 3, 1)), name='transpose_bbox')(bbox)
    # bbox = Lambda(lambda x: K.squeeze(x, axis=-1), name='reshape_bbox')(bbox)
    # bbox = BatchNormalization()(bbox)
    loc = Activation('softmax', name='loc_out')(loc)
    print("zhu")
    return cls, loc