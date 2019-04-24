import pickle
import numpy as np
import os
from network import vgg16 as cnn
from network import lstm as rnn
from network import output as out
from network import resnet
from keras.models import Model
from keras.utils import plot_model, np_utils
from keras.layers import Input, Lambda
from keras import metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras import backend as K
from help import sample_resize as sr
from help import handle_model as hm


# 导入事先存好的样本
def loadSample(samples):
    with open(samples, 'rb') as f:
        all_images = pickle.load(f)
        class_mapping = pickle.load(f)
        classes_count = pickle.load(f)
    return all_images, class_mapping, classes_count


def getBatch(X_path, batch_size, label_path, val=False):
    FMAP_w = ResizeW // DSR_x
    FMAP_h = ResizeH // DSR_y
    X_path = np.array(X_path)
    while 1:
        # 打乱数据非常重要
        permutation = np.random.permutation(X_path.shape[0])
        X_path = X_path[permutation]
        # label_c = label_c[permutation, :, :, :]
        # label_bbox = label_bbox[permutation, :, :, :]
        # X, label_c, label_bbox = shuffleTogether(X_path, label_c, label_bbox)
        print(X_path[0])  # 用于确认是否打乱
        if val:
            Epoch_len = X_path.shape[0]
        else:
            Epoch_len = X_path.shape[0]  # Todo:如果要每次用部分训练样本，设置这里
        for i in range(0, Epoch_len, batch_size):
            x = sr.sampleResize(x_path=X_path[i:i + batch_size], short_side=ShortSide,
                                long_side=LongSide, resizeh=ResizeH, resizew=ResizeW, resizeX=True)
            Y_number = len(X_path[i:i + batch_size])
            cls = np.zeros((Y_number, FMAP_h, FMAP_w, 1))
            loc = np.zeros((Y_number, FMAP_h, FMAP_w, 1))
            for j in range(Y_number):
                row = 0
                Y_path = label_path + str(os.path.basename(X_path[i + j]).split('.')[0]) + '.txt'
                with open(Y_path, 'r') as f:
                    for line in f:
                        line_split = line.strip().split(' ')
                        cls[j][row // FMAP_w][row % FMAP_w][0] = int(float(line_split[0]))
                        loc[j][row // FMAP_w][row % FMAP_w][0] = int(float(line_split[1]))
                        row += 1
            # cls = label_c[i:i + batch_size, :, :, :]
            # bbox = label_bbox[i:i + batch_size, :, :, :]
            # 最重要的就是这个yield，它代表返回，返回以后循环还是会继续，然后再返回。
            # 就比如有一个机器一直在作累加运算，但是会把每次累加中间结果告诉你一样，直到把所有数加完
            yield ({'input': x}, {'cls_out': cls,
                                  'loc_out': np_utils.to_categorical(loc, num_classes=5)})


def getModel():
    inputs = Input(shape=(None, None, CHANNEL), name='input')
    # 卷积提取部分
    if backbone == 'vgg16':
        if DSR_x == DSR_y == 32:
            x = cnn.network_cnn(inputs)
        elif DSR_x == DSR_y == 16:
            x = cnn.network_cnn_16(inputs)
        elif DSR_x == DSR_y == 8:
            if FCN:
                x = cnn.network_fcn_8(inputs)
            else:
                x = cnn.network_cnn_8(inputs)
        elif DSR_y == 8 and DSR_x == 16:
            if DilatedConvolution:
                x = cnn.network_cnn_8_16_dilated(inputs)
            else:
                x = cnn.network_cnn_8_16(inputs)
        elif DSR_x == DSR_y == 4:
            if FCN:
                x = cnn.network_fcn_4(inputs)
            else:
                x = cnn.network_cnn_4(inputs)
    else:
        x = resnet.resnet_graph_original(inputs, backbone, stage5=True)
    # print(x.shape)
    # x = core.Reshape((x_train.shape[1] // DSR, x_train.shape[2] // DSR))(x)
    # print(x.shape)
    # x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)), name='transpose_1')(x)
    # print(x.shape)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_1')(x)
    print(x.shape)
    x = rnn.network_cudnnlstm(x, output_size=256, name='vertical_lstm', mode='concat')
    print(x.shape)
    x = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1, 3)), name='transpose_2')(x)
    print(x.shape)
    x = rnn.network_cudnnlstm(x, output_size=256, name='horizontal_lstm1', mode='concat')
    if FC:
        cls = out.network_classification(x)
        loc = out.network_classification_location(x)
        # print(cls.shape)
        # if NoScale:
        #     bbox = out.network_regression_no_scaler(x)
        # else:
        #     bbox = out.network_regression(x)
        # print(bbox.shape)
    else:
        x = rnn.network_cudnnlstm(x, output_size=6, name='horizontal_lstm2', mode='sum')
        cls, loc = out.network_out_location(x)
        # if NoScale:
        #     cls, bbox = out.network_out_no_FC_scaler(x)
        # else:
        #     cls, bbox = out.network_out_no_FC(x)

    model = Model(inputs=inputs, outputs=[cls, loc])
    return model


# def bgIgnoredLoss(y_true, y_pred):
#     c1 = K.cast(K.equal(y_true[:, :, 0], 0), 'float32')
#     c2 = K.cast(K.equal(y_true[:, :, 1], 0), 'float32')
#     c3 = K.cast(K.equal(y_true[:, :, 2], 0), 'float32')
#     c4 = K.cast(K.equal(y_true[:, :, 3], 0), 'float32')
#     check = 1 - c1 * c2 * c3 * c4
#     normalizer = K.sum(check) + K.pow(10.0, -9)
#     loss_x = K.sum(K.pow((y_true[:, :, 0] - y_pred[:, :, 0]) * check, 2))
#     loss_y = K.sum(K.pow((y_true[:, :, 1] - y_pred[:, :, 1]) * check, 2))
#     loss_w = K.sum(K.pow((K.sqrt(y_true[:, :, 2]) - K.sqrt(y_pred[:, :, 2])) * check, 2))
#     loss_h = K.sum(K.pow((K.sqrt(y_true[:, :, 3]) - K.sqrt(y_pred[:, :, 3])) * check, 2))
#     return 1 * (loss_x + loss_y + loss_w + loss_h) / normalizer


# 训练模型
def train(model, x_train, x_val):
    # if NoBgLoss:
    #     reg_loss = bgIgnoredLoss
    # else:
    #     reg_loss = 'mean_squared_error'
    # 设置模型的参数
    model.compile(loss={'cls_out': 'binary_crossentropy', 'loc_out': 'categorical_crossentropy'},
                  loss_weights={'cls_out': 1.0, 'loc_out': 1.0}, optimizer='adam',
                  metrics={'cls_out': [metrics.binary_accuracy], 'loc_out': [metrics.categorical_accuracy]})
    # set callbacks
    # 设置模型按什么标准进行保存。比如：acc,loss
    CP = ModelCheckpoint(ModelFile, monitor='val_cls_out_binary_accuracy',
                         verbose=Verbose, save_best_only=False, mode='auto')
    # 设置如果性能不上升，停止学习
    ES = EarlyStopping(monitor='val_cls_out_binary_accuracy', patience=Patience)
    callbacks_list = [CP, ES]
    # 训练开始
    # model.fit(x_train, {'cls_out': cls_train, 'reg_out': bbox_train}, shuffle=True,
    #           batch_size=BatchSize, epochs=Epochs,
    #           verbose=Verbose, callbacks=callbacks_list,
    #           validation_data=(x_val, {'cls_out': cls_val, 'reg_out': bbox_val}))
    num_train_sample = len(x_train)  # Todo:如果要每次用部分训练样本，设置这里
    num_val_sample = len(x_val)
    model.fit_generator(generator=getBatch(x_train, TrainBatchSize, train_label),
                        steps_per_epoch=num_train_sample / TrainBatchSize, shuffle=True,
                        epochs=Epochs, verbose=Verbose,
                        validation_data=getBatch(x_val, ValBatchSize, val_label, val=True),
                        validation_steps=num_val_sample / ValBatchSize,
                        callbacks=callbacks_list)


def getInformation(data):
    X_path = []
    for i in range(len(data)):
        X_path.append(data[i]['filepath'])
        # y.append(data[i]['bboxes'])
        # size.append([data[i]['height'], data[i]['width'], data[i]['channel']])
    return X_path  # , y, size


def loadModel():
    # 先检查是否指定，再检查在指定路径下检查模型是否存在，如果存在，导入。如果不存在，创建。
    if LoadModel:
        print('Load model named by {}...'.format(WeightFile))
        if backbone == 'resnet50' or backbone == 'resnet101':
            model = load_model(WeightFile, custom_objects={'BatchNorm': resnet.BatchNorm})
        else:
            model = load_model(WeightFile)
    else:
        FLAG, MN = hm.checkModel(ModelPath)
        if FLAG:
            if backbone == 'resnet50' or backbone == 'resnet101':
                model = load_model(MN, custom_objects={'BatchNorm': resnet.BatchNorm})
            else:
                model = load_model(MN)
            print(MN, 'is loaded.')
        else:
            print('Build a new model...')
            model = getModel()
    # 打印模型
    print(model.summary())
    # 绘制模型图
    # plot_model(model, to_file='./model/model.png')
    return model


def main():
    # 准备样本
    print('All samples are from pickle files...')
    train_images, class_mapping, classes_count = loadSample(train_pickle)
    syn_images, class_mapping, classes_count = loadSample(train_syn)
    train_images.extend(syn_images)
    val_images, class_mapping, classes_count = loadSample(val_pickle)
    # train_images = [s for s in all_images if s['imageset'] == 'train']
    # val_images = [s for s in all_images if s['imageset'] == 'validation']
    print('The total training samples are {}.'.format(len(train_images)))
    print('The total val samples are {}.'.format(len(val_images)))

    X_train_path = getInformation(train_images)
    X_val_path = getInformation(val_images)
    # input('Stop!')

    model = loadModel()
    # 训练模型
    print('Training start...')
    train(model, X_train_path, X_val_path)
    # 保留最好的几个模型，删除其它的。
    # hm.clearModel(ModelPath)
    print('Congratulation! It finished.')


if __name__ == '__main__':
    import argparse

    # 命令行参数传入
    parser = argparse.ArgumentParser(description="Train 2D-ChipNet to detect scene text.")
    parser.add_argument('--weights', required=False, metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--tryinfo', required=False, metavar="01",
                        help="Description of the training model")
    parser.add_argument('--backbone', required=False,
                        metavar="vgg16", help="Backbone")
    parser.add_argument('--trainbatchsize', required=False,
                        metavar="32", help="Train Batch Size")
    parser.add_argument('--valbatchsize', required=False,
                        metavar="32", help="Val Batch Size")
    parser.add_argument('--epochs', required=False,
                        metavar="300", help="Epochs")
    args = parser.parse_args()

    DSR_x = 4
    DSR_y = 4

    # 训练描述信息
    TRY = '01'
    if args.tryinfo:
        TRY = args.tryinfo

    # BackBone
    backbone = "vgg16"
    if args.backbone:
        backbone = args.backbone
        # if backbone == 'resnet50' or backbone == 'resnet101':
        #     DSR_x = 32
        #     DSR_y = 32

    # Batch Size (1024集群内存不足)
    TrainBatchSize = 4
    if args.trainbatchsize:
        TrainBatchSize = int(args.trainbatchsize)

    ValBatchSize = 4
    if args.valbatchsize:
        ValBatchSize = int(args.valbatchsize)

    # Epochs 总数
    Epochs = 300
    if args.epochs:
        Epochs = int(args.epochs)

    # 训练初始化模型导入配置
    LoadModel = False
    if args.weights:
        LoadModel = True
    WeightFile = args.weights

    # 模型保存相关配置
    CNNModel = backbone + '_' + str(DSR_y) + '_' + str(DSR_x) + '_' + TRY  # example: vgg16_4_2_01
    ModelPath = './model/' + CNNModel + '/'
    # 保存的模型位置和名称，名称根据epoch和精度变化
    ModelFile = ModelPath + 'icdar2013-{epoch:03d}-{val_cls_out_binary_accuracy:.5f}-{val_loc_out_categorical_accuracy:.5f}.hdf5'

    FC = False
    # NoBgLoss = False
    DilatedConvolution = False
    # NoScale = False
    FCN = True
    Patience = 300
    Verbose = 1  # 显示方式
    ResizeH = 608
    ResizeW = 800
    ShortSide = 608
    LongSide = 800
    CHANNEL = 3
    CLASS = 1
    CrossRatio = 1  # Todo: 请确认
    train_pickle = './sample/icdar2013_train_sample.pkl'
    train_syn = './sample/icdar2013_train_syn_sample.pkl'
    val_pickle = './sample/icdar2013_test_sample_608_800.pkl'
    # scale_path = './sample/train_scale_bbox_608_800_' + \
    #              str(DSR_y) + '_' + str(DSR_x) + '.pkl'  # todo:改变了下采样需要改这里
    train_label = './sample/label_train_icdar2013_location_' + \
                  str(DSR_y) + '_' + str(DSR_x) + '_' + str(CrossRatio) + '/'
    # train_FORUEng2k_label = './sample/label_train_FORUEng2k_' + str(DSR_x) + '_' + str(CrossRatio) + '/'
    val_label = './sample/label_test_icdar2013_location_' + \
                str(DSR_y) + '_' + str(DSR_x) + '_' + str(CrossRatio) + '/'

    # 配置信息输出
    print("\n\n\n")
    print("Initialize Weights: ", WeightFile)
    # print("Dataset: ", args.dataset)
    print("Try Infomations: ", TRY)
    print("Model Backbone: ", backbone)
    print("Train Batch Size: ", str(TrainBatchSize))
    print("Val Batch Size: ", str(ValBatchSize))
    print("Model Epochs: ", str(Epochs))

    main()
