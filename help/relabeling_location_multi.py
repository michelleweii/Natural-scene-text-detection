import sys

sys.path.append('../')

from help import set_train_scale as sts
from help import sample_resize as sr
from help import min_max_scaler as mms
from help import handle_sample as hs
from help import show_picture as sp  # Todo:测试框放缩的时候使用
import pickle
from multiprocessing import Pool
import os, time, random
import math

DSR_x = 4  # Todo: 请确认
DSR_y = 4  # Todo: 请确认
ResizeH = 608
ResizeW = 800
ShortSide = 608
LongSide = 800
CHANNEL = 3
CLASS = 1
CrossRatio = 1  # Todo: 请确认
train_pickle = '../sample/icdar2013_train_syn_sample.pkl'  # Todo: 请确认 train or test
# scale_path = '../sample/train_scale_bbox_608_800_4_4.pkl'

Check_Resize = False  # Todo: 检查放缩之后的框是否正确
Check_Resize2 = False
# FirstTime = False  # Todo: 请确认

# Todo: 请确认 train or test
Relabeled_label = '../sample/label_train_icdar2013_location_' \
                  + str(DSR_y) + '_' + str(DSR_x) + '_' + str(CrossRatio) + '/'
if not os.path.exists(Relabeled_label):
    os.makedirs(Relabeled_label)


# 导入事先存好的样本
def loadSample(samples):
    with open(samples, 'rb') as f:
        all_images = pickle.load(f)
        class_mapping = pickle.load(f)
        classes_count = pickle.load(f)
    return all_images, class_mapping, classes_count


def getInformation(data):
    X_path = []
    y = []
    size = []
    for i in range(len(data)):
        X_path.append(data[i]['filepath'])
        y.append(data[i]['bboxes'])
        size.append([data[i]['height'], data[i]['width'], data[i]['channel']])
    return X_path, y, size


def checkResize(x_path, y, im_size):
    for i in range(len(y)):
        x_path[i] = '.' + x_path[i]
        sx, sy = sr.sampleResize(x_path=x_path[i:i + 1], y=y[i:i + 1], im_size=im_size[i:i + 1], short_side=ShortSide,
                                 long_side=LongSide, resizeh=ResizeH, resizew=ResizeW,
                                 resizeX=True, resizeY=True)
        # print(len(sx[0]),len(sy[0]))
        print(x_path[i])
        sp.show_bbox_in_one_image(sx, sy)


def generateGridLabel(x_path, y, img_size):
    # scale_file = open(scale_path, 'rb')
    # label_min = pickle.load(scale_file)
    # label_max = pickle.load(scale_file)
    # scale_file.close()

    if len(img_size) != len(y):
        print('The length of size_info and y is not same.')
        os._exit(0)

    if Check_Resize2:
        for i in range(len(y)):
            x_path[i] = '.' + x_path[i]
        x, y = sr.sampleResize(x_path=x_path, y=y, im_size=img_size, short_side=ShortSide,
                               long_side=LongSide, resizeh=ResizeH, resizew=ResizeW,
                               resizeX=True, resizeY=True)
        sp.show_bbox_in_one_image(x, y)
        input('Stop')

    y = sr.sampleResize(y=y, im_size=img_size, short_side=ShortSide,
                        long_side=LongSide, resizeh=ResizeH, resizew=ResizeW, resizeY=True)

    label_c, label_location = hs.relabel_2D_location(y, ResizeH, ResizeW, DSR_x, DSR_y, CLASS, CrossRatio)
    # label_c = np_utils.to_categorical(label_c, num_classes=CLASS + 1)
    # label_bbox = mms.vtMinMaxScaler(label_bbox, label_min, label_max)
    return label_c, label_location


def saveLabeltoTxt(images, label_c, label_location, save_path):
    if len(images) != label_c.shape[0]:
        print('The lengths of X and Y are not same.')
        os._exit(0)
    for i in range(len(images)):
        txt = save_path + str(os.path.basename(images[i]['filepath']).split('.')[0]) + ".txt"
        with open(txt, 'w') as f:
            for j in range(label_c.shape[1]):  #
                for k in range(label_c.shape[2]):  #
                    f.write('{} {} \n'.format(label_c[i][j][k][0], label_location[i][j][k][0]))


def multi_pro(X_path, all_images, Y_train, size):
    print('Generating grid samples...')
    label_c, label_location = generateGridLabel(X_path, Y_train, size)
    saveLabeltoTxt(all_images, label_c, label_location, Relabeled_label)
    print('The relabeled labels have been generated.')


def main():
    print('All samples are from pickle files...')
    all_images, class_mapping, classes_count = loadSample(train_pickle)

    X_path, Y_train, size = getInformation(all_images)

    if Check_Resize:  # Todo: 检查放缩之后的框是否正确
        checkResize(X_path, Y_train, size)
    # input('stop')

    # if FirstTime:
    #     sts.setTrainScale(ResizeW, ResizeH, DSR_x, DSR_y, scale_path)
    # input('stop')

    MultiNum = 1  # 并发线程数
    print("MultiProcess:" + MultiNum.__str__())
    # 新建指定数量的进程池用于对多进程进行管理
    pool = Pool(processes=MultiNum)

    # 划分任务, tmp 代表每一个进程完成的任务数
    tmp = int(math.ceil(len(Y_train) / MultiNum))
    for i in range(MultiNum):
        pool.apply_async(multi_pro, args=(X_path[i * tmp:i * tmp + tmp],
                                          all_images[i * tmp:i * tmp + tmp], Y_train[i * tmp:i * tmp + tmp],
                                          size[i * tmp:i * tmp + tmp]))

    pool.close()
    pool.join()


if __name__ == '__main__':
    # print('Parent process %s.' % os.getpid())
    # p = Pool(28)
    # for i in range(28):
    #     p.apply_async(main)
    # print('Waiting for all subprocesses done...')
    # p.close()
    # p.join()
    # print('All subprocesses done.')
    main()
