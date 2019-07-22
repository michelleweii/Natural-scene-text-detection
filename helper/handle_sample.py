import numpy as np
import random
from tqdm import tqdm
from collections import Counter
import os
from help import polygon_calculate as pc
import math

# todo: 获取样本，并对样本进行重新标注
'''
这里会把输入的图片竖地切成很多条，具体数目由CNN输出的特征图的宽度决定。
假设实验中输入图片160*32，特征图下采样为1/4，那么特征图为40*8，所以切成40条。
sample_x: 用于存放图片矩阵的像素信息。（样本数量，宽度，高度，通道数）
          注意高度和宽度需要对换轴，目的是送入LSTM的时候不用再变换，
          这样就不用添加Lambda层了。
label_c: 每张图片按照40条进行类别标注，如果和框的横边有50%的横向重叠，
         就标注为该类别，如果没有，标注为背景。因此一共有c+1类。
         注意：这里的50%是根据重叠除以更短边设置的。
label_bbox: 切出来的每条图片，进行bbox标注，如果对应为背景，全部置0，
            如果和框的横边有50%的横向重叠，就标注为该类别的GT。
'''


# X 绝对位置
def getSample_and_relabel_1D_abs(dict_img, dsr):
    (height, width, channel) = dict_img[0]['pixel'].shape
    sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), width // dsr, 1)) * 10
    label_bbox = np.zeros((len(dict_img), width // dsr, 4))
    for i in range(len(dict_img)):
        sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        bb_list = dict_img[i]['bboxes']
        for k in range(0, width, dsr):  # 160, 4
            w_spilt = set(range(k, k + dsr))
            cross_tmp = 0
            list_tmp = []
            for j in range(len(bb_list)):
                w = set(range(bb_list[j]['x1'], bb_list[j]['x2'] + 1))
                cross_ratio = len(w_spilt & w) / min(len(w), len(w_spilt))
                if cross_ratio >= 0.5:
                    if cross_ratio > cross_tmp:
                        label_c[i, k // dsr, 0] = int(bb_list[j]['class'])
                        label_bbox[i, k // dsr, 0] = bb_list[j]['x1']  # - k
                        label_bbox[i, k // dsr, 1] = bb_list[j]['y1']
                        label_bbox[i, k // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                        label_bbox[i, k // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                        list_tmp.clear()
                        list_tmp.append(j)
                    elif cross_ratio == cross_tmp:
                        list_tmp.append(j)
                        rand_digit = random.sample(list_tmp, 1)[0]
                        label_c[i, k // dsr, 0] = int(bb_list[rand_digit]['class'])
                        label_bbox[i, k // dsr, 0] = bb_list[rand_digit]['x1']  # - k
                        label_bbox[i, k // dsr, 1] = bb_list[rand_digit]['y1']
                        label_bbox[i, k // dsr, 2] = bb_list[rand_digit]['x2'] - bb_list[rand_digit]['x1'] + 1
                        label_bbox[i, k // dsr, 3] = bb_list[rand_digit]['y2'] - bb_list[rand_digit]['y1'] + 1
                cross_tmp = cross_ratio

    # 高度和宽度需要对换轴
    # sample_x = sample_x.transpose((0, 2, 1, 3))
    # print(sample_x[0])
    return sample_x, label_c, label_bbox


# X 相对位置
def getSample_and_relabel_1D_offset(dict_img, dsr):
    (height, width, channel) = dict_img[0]['pixel'].shape
    sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), width // dsr, 1)) * 10
    label_bbox = np.zeros((len(dict_img), width // dsr, 4))
    for i in tqdm(range(len(dict_img))):
        sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        bb_list = dict_img[i]['bboxes']
        for k in range(0, width, dsr):  # 160, 4
            w_spilt = set(range(k, k + dsr))
            cross_tmp = 0
            list_tmp = []
            for j in range(len(bb_list)):
                w = set(range(bb_list[j]['x1'], bb_list[j]['x2'] + 1))
                cross_ratio = len(w_spilt & w) / min(len(w), len(w_spilt))
                if cross_ratio >= 0.5:
                    if cross_ratio > cross_tmp:
                        label_c[i, k // dsr, 0] = int(bb_list[j]['class'])
                        label_bbox[i, k // dsr, 0] = bb_list[j]['x1'] - k
                        label_bbox[i, k // dsr, 1] = bb_list[j]['y1']
                        label_bbox[i, k // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                        label_bbox[i, k // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                        list_tmp.clear()
                        list_tmp.append(j)
                    elif cross_ratio == cross_tmp:
                        list_tmp.append(j)
                        # rand_digit = random.sample(list_tmp, 1)[0]
                        # 当交叉率相同的时候，选择宽度更窄的bbox作为标注框
                        w_min = len(range(bb_list[list_tmp[0]]['x1'], bb_list[list_tmp[0]]['x2'] + 1))
                        belong_to_which_bbox = 0
                        for choice in range(1, len(list_tmp)):
                            w_now = len(range(bb_list[list_tmp[choice]]['x1'], bb_list[list_tmp[choice]]['x2'] + 1))
                            if w_now < w_min:
                                w_min = w_now
                                belong_to_which_bbox = list_tmp[choice]
                            elif w_now == w_min:
                                rand_box = [belong_to_which_bbox, list_tmp[choice]]
                                belong_to_which_bbox = random.sample(rand_box, 1)[0]

                        label_c[i, k // dsr, 0] = int(bb_list[belong_to_which_bbox]['class'])
                        label_bbox[i, k // dsr, 0] = bb_list[belong_to_which_bbox]['x1'] - k
                        label_bbox[i, k // dsr, 1] = bb_list[belong_to_which_bbox]['y1']
                        label_bbox[i, k // dsr, 2] = bb_list[belong_to_which_bbox]['x2'] - \
                                                     bb_list[belong_to_which_bbox]['x1'] + 1
                        label_bbox[i, k // dsr, 3] = bb_list[belong_to_which_bbox]['y2'] - \
                                                     bb_list[belong_to_which_bbox]['y1'] + 1
                cross_tmp = cross_ratio
    return sample_x, label_c, label_bbox


# X 相对位置  当交叉率相同的时候，丢弃
def getSample_and_relabel_1D_offset_no_conflict(dict_img, dsr):
    (height, width, channel) = dict_img[0]['pixel'].shape
    sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), width // dsr, 1)) * 10
    label_bbox = np.zeros((len(dict_img), width // dsr, 4))
    for i in tqdm(range(len(dict_img))):
        sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        bb_list = dict_img[i]['bboxes']
        for k in range(0, width, dsr):  # 160, 4
            w_spilt = set(range(k, k + dsr))
            cross_tmp = 0
            list_tmp = []
            for j in range(len(bb_list)):
                w = set(range(bb_list[j]['x1'], bb_list[j]['x2'] + 1))
                cross_ratio = len(w_spilt & w) / min(len(w), len(w_spilt))
                if cross_ratio >= 0.5:
                    if cross_ratio > cross_tmp:
                        label_c[i, k // dsr, 0] = int(bb_list[j]['class'])
                        label_bbox[i, k // dsr, 0] = bb_list[j]['x1'] - k
                        label_bbox[i, k // dsr, 1] = bb_list[j]['y1']
                        label_bbox[i, k // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                        label_bbox[i, k // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                        list_tmp.clear()
                        list_tmp.append(j)
                    elif cross_ratio == cross_tmp:
                        list_tmp.append(j)
                        # rand_digit = random.sample(list_tmp, 1)[0]
                        # 当交叉率相同的时候，选择宽度更窄的bbox作为标注框
                        w_min = len(range(bb_list[list_tmp[0]]['x1'], bb_list[list_tmp[0]]['x2'] + 1))
                        belong_to_which_bbox = 0
                        for choice in range(1, len(list_tmp)):
                            w_now = len(range(bb_list[list_tmp[choice]]['x1'], bb_list[list_tmp[choice]]['x2'] + 1))
                            if w_now < w_min:
                                w_min = w_now
                                belong_to_which_bbox = list_tmp[choice]
                                label_c[i, k // dsr, 0] = int(bb_list[belong_to_which_bbox]['class'])
                                label_bbox[i, k // dsr, 0] = bb_list[belong_to_which_bbox]['x1'] - k
                                label_bbox[i, k // dsr, 1] = bb_list[belong_to_which_bbox]['y1']
                                label_bbox[i, k // dsr, 2] = bb_list[belong_to_which_bbox]['x2'] - \
                                                             bb_list[belong_to_which_bbox]['x1'] + 1
                                label_bbox[i, k // dsr, 3] = bb_list[belong_to_which_bbox]['y2'] - \
                                                             bb_list[belong_to_which_bbox]['y1'] + 1
                            elif w_now == w_min:
                                label_c[i, k // dsr, 0] = 10
                                label_bbox[i, k // dsr, 0] = 0
                                label_bbox[i, k // dsr, 1] = 0
                                label_bbox[i, k // dsr, 2] = 0
                                label_bbox[i, k // dsr, 3] = 0
                cross_tmp = cross_ratio
    return sample_x, label_c, label_bbox


def calculateCrossRatio(rec1, rec2):
    """
    computing cross ratio
    :param rec1: (x0, y0, w0, h0), which reflects
            (top, left, width, height)
    :param rec2: (x1, y1, w1, h1)
    :return: scala value of cross ratio
    """
    # computing area of each rectangles
    S_rec1 = rec1[2] * rec1[3]
    S_rec2 = rec2[2] * rec2[3]

    # computing the sum_area
    min_area = min(S_rec1, S_rec2)

    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2] + rec1[0] - 1, rec2[2] + rec2[0] - 1)
    top_line = max(rec1[1], rec2[1])
    bottom_line = min(rec1[3] + rec1[1] - 1, rec2[3] + rec2[1] - 1)

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line + 1) * (bottom_line - top_line + 1)
        return intersect / min_area


# X Y 相对位置
def getSample_and_relabel_2D(dict_img, dsr):
    (height, width, channel) = dict_img[0]['pixel'].shape
    print(dict_img[0]['pixel'].shape)
    # input('Stop!')

    # tmp_num = 0

    sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), height // dsr, width // dsr, 1)) * 10
    label_bbox = np.zeros((len(dict_img), height // dsr, width // dsr, 4))

    print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        # print(sample_x[i].shape)
        # input('Stop!')
        bb_list = dict_img[i]['bboxes']
        for y in range(0, height, dsr):  # 32, 4, y代表纵向
            for x in range(0, width, dsr):  # 160, 4, x代表横向
                grid = [x, y, dsr, dsr]
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                list_tmp = []
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    if cross_ratio >= 0.5:
                        if cross_ratio > cross_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1'] - x
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1'] - y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            list_tmp.clear()
                            list_tmp.append(j)
                        elif cross_ratio == cross_tmp:
                            list_tmp.append(j)
                            rand_digit = random.sample(list_tmp, 1)[0]
                            # tmp_num += 1
                            # print('{} random bounding boxes happened!'.format(tmp_num))
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[rand_digit]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[rand_digit]['x1'] - x
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[rand_digit]['y1'] - y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[rand_digit]['x2'] - \
                                                                   bb_list[rand_digit]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[rand_digit]['y2'] - \
                                                                   bb_list[rand_digit]['y1'] + 1
                    cross_tmp = cross_ratio

    # 高度和宽度需要对换轴
    # sample_x = sample_x.transpose((0, 2, 1, 3))
    # print(sample_x[0])
    return sample_x, label_c, label_bbox


# X Y 相对位置
def getSample_and_relabel_2D_no_conflict(dict_img, dsr):
    (height, width, channel) = dict_img[0]['pixel'].shape
    print(dict_img[0]['pixel'].shape)
    # input('Stop!')

    # tmp_num = 0

    sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), height // dsr, width // dsr, 1)) * 10
    label_bbox = np.zeros((len(dict_img), height // dsr, width // dsr, 4))

    print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        # input('continue?')
        sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        # print(sample_x[i].shape)
        # input('Stop!')
        bb_list = dict_img[i]['bboxes']
        for y in range(0, height, dsr):  # 32, 4, y代表纵向
            for x in range(0, width, dsr):  # 160, 4, x代表横向
                grid = [x, y, dsr, dsr]
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                list_tmp = []
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    # print(cross_ratio)
                    if cross_ratio >= 0.5:
                        if cross_ratio > cross_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1'] - x
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1'] - y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            list_tmp.clear()
                            list_tmp.append(j)
                        elif cross_ratio == cross_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = 10
                            label_bbox[i, y // dsr, x // dsr, 0] = 0
                            label_bbox[i, y // dsr, x // dsr, 1] = 0
                            label_bbox[i, y // dsr, x // dsr, 2] = 0
                            label_bbox[i, y // dsr, x // dsr, 3] = 0
                            # list_tmp.append(j)
                            # rand_digit = random.sample(list_tmp, 1)[0]
                            # tmp_num += 1
                            # print('{} random bounding boxes happened!'.format(tmp_num))
                            # label_c[i, y // dsr, x // dsr, 0] = int(bb_list[rand_digit]['class'])
                            # label_bbox[i, y // dsr, x // dsr, 0] = bb_list[rand_digit]['x1'] - x
                            # label_bbox[i, y // dsr, x // dsr, 1] = bb_list[rand_digit]['y1'] - y
                            # label_bbox[i, y // dsr, x // dsr, 2] = bb_list[rand_digit]['x2'] - \
                            #                                        bb_list[rand_digit]['x1'] + 1
                            # label_bbox[i, y // dsr, x // dsr, 3] = bb_list[rand_digit]['y2'] - \
                            #                                        bb_list[rand_digit]['y1'] + 1
                    cross_tmp = cross_ratio

    # 高度和宽度需要对换轴
    # sample_x = sample_x.transpose((0, 2, 1, 3))
    # print(sample_x[0])
    return sample_x, label_c, label_bbox


# X Y 相对位置
def getSample_and_relabel_2D_concentrate(dict_img, dsr):
    (height, width, channel) = dict_img[0]['pixel'].shape
    print(dict_img[0]['pixel'].shape)
    # input('Stop!')

    # tmp_num = 0

    sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), height // dsr, width // dsr, 1)) * 10
    label_bbox = np.zeros((len(dict_img), height // dsr, width // dsr, 4))

    print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        # input('continue?')
        sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        # print(sample_x[i].shape)
        # input('Stop!')
        bb_list = dict_img[i]['bboxes']
        for y in range(0, height, dsr):  # 32, 4, y代表纵向
            for x in range(0, width, dsr):  # 160, 4, x代表横向
                grid = [x, y, dsr, dsr]
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                flag = 0
                list_tmp = []
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    # print(cross_ratio)
                    if cross_ratio > 0.5:
                        flag = +1

                        if cross_ratio > cross_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1'] - x
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1'] - y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            list_tmp.clear()
                            list_tmp.append(j)
                        elif cross_ratio == cross_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = 10
                            label_bbox[i, y // dsr, x // dsr, 0] = 0
                            label_bbox[i, y // dsr, x // dsr, 1] = 0
                            label_bbox[i, y // dsr, x // dsr, 2] = 0
                            label_bbox[i, y // dsr, x // dsr, 3] = 0
                            # list_tmp.append(j)
                            # rand_digit = random.sample(list_tmp, 1)[0]
                            # tmp_num += 1
                            # print('{} random bounding boxes happened!'.format(tmp_num))
                            # label_c[i, y // dsr, x // dsr, 0] = int(bb_list[rand_digit]['class'])
                            # label_bbox[i, y // dsr, x // dsr, 0] = bb_list[rand_digit]['x1'] - x
                            # label_bbox[i, y // dsr, x // dsr, 1] = bb_list[rand_digit]['y1'] - y
                            # label_bbox[i, y // dsr, x // dsr, 2] = bb_list[rand_digit]['x2'] - \
                            #                                        bb_list[rand_digit]['x1'] + 1
                            # label_bbox[i, y // dsr, x // dsr, 3] = bb_list[rand_digit]['y2'] - \
                            #                                        bb_list[rand_digit]['y1'] + 1
                    cross_tmp = cross_ratio

    # 高度和宽度需要对换轴
    # sample_x = sample_x.transpose((0, 2, 1, 3))
    # print(sample_x[0])
    return sample_x, label_c, label_bbox


# Todo:最新方法在此
# X Y 相对位置
def relabel_2D_try_1(dict_img, height, width, dsr, cls):
    # (height, width, channel) = dict_img[0]['pixel'].shape
    # print(dict_img[0]['pixel'].shape)
    # input('Stop!')

    # tmp_num = 0

    # sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), height // dsr, width // dsr, 1)) * cls
    label_bbox = np.zeros((len(dict_img), height // dsr, width // dsr, 4))

    # print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        # input('continue?')
        # sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        # print(sample_x[i].shape)
        # input('Stop!')
        bb_list = dict_img[i]
        for y in range(0, height, dsr):  # 32, 4, y代表纵向
            for x in range(0, width, dsr):  # 160, 4, x代表横向
                grid = [x, y, dsr, dsr]
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                box_area_tmp = 0
                # list_tmp = []
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    box_area = bounding_box[2] * bounding_box[3]
                    # print(cross_ratio)
                    if cross_ratio > 0.5:
                        if cross_ratio > cross_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1'] - x
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1'] - y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            box_area_tmp = box_area
                            cross_tmp = cross_ratio
                            # list_tmp.clear()
                            # list_tmp.append(j)
                        elif cross_ratio == cross_tmp and box_area < box_area_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1'] - x
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1'] - y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            # list_tmp.append(j)
                            # rand_digit = random.sample(list_tmp, 1)[0]
                            # tmp_num += 1
                            # print('{} random bounding boxes happened!'.format(tmp_num))
                            # label_c[i, y // dsr, x // dsr, 0] = int(bb_list[rand_digit]['class'])
                            # label_bbox[i, y // dsr, x // dsr, 0] = bb_list[rand_digit]['x1'] - x
                            # label_bbox[i, y // dsr, x // dsr, 1] = bb_list[rand_digit]['y1'] - y
                            # label_bbox[i, y // dsr, x // dsr, 2] = bb_list[rand_digit]['x2'] - \
                            #                                        bb_list[rand_digit]['x1'] + 1
                            # label_bbox[i, y // dsr, x // dsr, 3] = bb_list[rand_digit]['y2'] - \
                            #                                        bb_list[rand_digit]['y1'] + 1

    # 检查label
    # check_label = label_c.flatten()
    # count_label = Counter(check_label)
    # if len(count_label) != 21:
    # print('Label is less than 21.')
    # print(count_label)
    # os._exit(0)
    return label_c, label_bbox


# X Y 绝对位置
def relabel_2D_try_2(dict_img, height, width, dsr, cls):
    # (height, width, channel) = dict_img[0]['pixel'].shape
    # print(dict_img[0]['pixel'].shape)
    # input('Stop!')

    # tmp_num = 0

    # sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), height // dsr, width // dsr, 1)) * cls
    label_bbox = np.zeros((len(dict_img), height // dsr, width // dsr, 4))

    # print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        # input('continue?')
        # sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        # print(sample_x[i].shape)
        # input('Stop!')
        bb_list = dict_img[i]
        for y in range(0, height, dsr):  # 32, 4, y代表纵向
            for x in range(0, width, dsr):  # 160, 4, x代表横向
                grid = [x, y, dsr, dsr]
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                box_area_tmp = 0
                # list_tmp = []
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    box_area = bounding_box[2] * bounding_box[3]
                    # print(cross_ratio)
                    if cross_ratio > 0.5:
                        if cross_ratio > cross_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1']
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1']
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            box_area_tmp = box_area
                            cross_tmp = cross_ratio
                            # list_tmp.clear()
                            # list_tmp.append(j)
                        elif cross_ratio == cross_tmp and box_area < box_area_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1']
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1']
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            # list_tmp.append(j)
                            # rand_digit = random.sample(list_tmp, 1)[0]
                            # tmp_num += 1
                            # print('{} random bounding boxes happened!'.format(tmp_num))
                            # label_c[i, y // dsr, x // dsr, 0] = int(bb_list[rand_digit]['class'])
                            # label_bbox[i, y // dsr, x // dsr, 0] = bb_list[rand_digit]['x1'] - x
                            # label_bbox[i, y // dsr, x // dsr, 1] = bb_list[rand_digit]['y1'] - y
                            # label_bbox[i, y // dsr, x // dsr, 2] = bb_list[rand_digit]['x2'] - \
                            #                                        bb_list[rand_digit]['x1'] + 1
                            # label_bbox[i, y // dsr, x // dsr, 3] = bb_list[rand_digit]['y2'] - \
                            #                                        bb_list[rand_digit]['y1'] + 1

    # 检查label
    # check_label = label_c.flatten()
    # count_label = Counter(check_label)
    # if len(count_label) != 21:
    # print('Label is less than 21.')
    # print(count_label)
    # os._exit(0)
    return label_c, label_bbox


# X Y 相对绝对可选
def relabel_2D_try_3(dict_img, height, width, dsr, cls, abs):
    # (height, width, channel) = dict_img[0]['pixel'].shape
    # print(dict_img[0]['pixel'].shape)
    # input('Stop!')

    # tmp_num = 0

    # sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), height // dsr, width // dsr, 1)) * cls
    label_bbox = np.zeros((len(dict_img), height // dsr, width // dsr, 4))

    # print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        # input('continue?')
        # sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        # print(sample_x[i].shape)
        # input('Stop!')
        bb_list = dict_img[i]
        for y in range(0, height, dsr):  # 32, 4, y代表纵向
            for x in range(0, width, dsr):  # 160, 4, x代表横向
                grid = [x, y, dsr, dsr]
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                box_area_tmp = 0
                # list_tmp = []
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    box_area = bounding_box[2] * bounding_box[3]
                    # print(cross_ratio)
                    if cross_ratio == 1.0:
                        if cross_ratio > cross_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1']
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1']
                            if abs is False:
                                label_bbox[i, y // dsr, x // dsr, 0] -= x
                                label_bbox[i, y // dsr, x // dsr, 1] -= y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            box_area_tmp = box_area
                            cross_tmp = cross_ratio
                            # list_tmp.clear()
                            # list_tmp.append(j)
                        elif cross_ratio == cross_tmp and box_area < box_area_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1']
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1']
                            if abs is False:
                                label_bbox[i, y // dsr, x // dsr, 0] -= x
                                label_bbox[i, y // dsr, x // dsr, 1] -= y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            # list_tmp.append(j)
                            # rand_digit = random.sample(list_tmp, 1)[0]
                            # tmp_num += 1
                            # print('{} random bounding boxes happened!'.format(tmp_num))
                            # label_c[i, y // dsr, x // dsr, 0] = int(bb_list[rand_digit]['class'])
                            # label_bbox[i, y // dsr, x // dsr, 0] = bb_list[rand_digit]['x1'] - x
                            # label_bbox[i, y // dsr, x // dsr, 1] = bb_list[rand_digit]['y1'] - y
                            # label_bbox[i, y // dsr, x // dsr, 2] = bb_list[rand_digit]['x2'] - \
                            #                                        bb_list[rand_digit]['x1'] + 1
                            # label_bbox[i, y // dsr, x // dsr, 3] = bb_list[rand_digit]['y2'] - \
                            #                                        bb_list[rand_digit]['y1'] + 1

    # 检查label
    # check_label = label_c.flatten()
    # count_label = Counter(check_label)
    # if len(count_label) != 21:
    # print('Label is less than 21.')
    # print(count_label)
    # os._exit(0)
    return label_c, label_bbox


# X Y 相对绝对可选
# cross_ratio可设定
def relabel_2D_try_4(dict_img, height, width, dsr, cls, abs, cr):
    # (height, width, channel) = dict_img[0]['pixel'].shape
    # print(dict_img[0]['pixel'].shape)
    # input('Stop!')
    # tmp_num = 0
    # sample_x = np.zeros((len(dict_img), height, width, channel))
    label_c = np.ones((len(dict_img), height // dsr, width // dsr, 1)) * cls
    label_bbox = np.zeros((len(dict_img), height // dsr, width // dsr, 4))

    # print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        # input('continue?')
        # sample_x[i] = (255 - dict_img[i]['pixel']) / 255
        # print(sample_x[i].shape)
        # input('Stop!')
        bb_list = dict_img[i]
        for y in range(0, height, dsr):  # 32, 4, y代表纵向
            for x in range(0, width, dsr):  # 160, 4, x代表横向
                grid = [x, y, dsr, dsr]
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                box_area_tmp = 0
                # list_tmp = []
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    box_area = bounding_box[2] * bounding_box[3]
                    # print(cross_ratio)
                    if cross_ratio >= cr:
                        if cross_ratio > cross_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1']
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1']
                            if abs is False:
                                label_bbox[i, y // dsr, x // dsr, 0] -= x
                                label_bbox[i, y // dsr, x // dsr, 1] -= y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            box_area_tmp = box_area
                            cross_tmp = cross_ratio
                            # list_tmp.clear()
                            # list_tmp.append(j)
                        elif cross_ratio == cross_tmp and box_area < box_area_tmp:
                            label_c[i, y // dsr, x // dsr, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr, x // dsr, 0] = bb_list[j]['x1']
                            label_bbox[i, y // dsr, x // dsr, 1] = bb_list[j]['y1']
                            if abs is False:
                                label_bbox[i, y // dsr, x // dsr, 0] -= x
                                label_bbox[i, y // dsr, x // dsr, 1] -= y
                            label_bbox[i, y // dsr, x // dsr, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr, x // dsr, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            # list_tmp.append(j)
                            # rand_digit = random.sample(list_tmp, 1)[0]
                            # tmp_num += 1
                            # print('{} random bounding boxes happened!'.format(tmp_num))
                            # label_c[i, y // dsr, x // dsr, 0] = int(bb_list[rand_digit]['class'])
                            # label_bbox[i, y // dsr, x // dsr, 0] = bb_list[rand_digit]['x1'] - x
                            # label_bbox[i, y // dsr, x // dsr, 1] = bb_list[rand_digit]['y1'] - y
                            # label_bbox[i, y // dsr, x // dsr, 2] = bb_list[rand_digit]['x2'] - \
                            #                                        bb_list[rand_digit]['x1'] + 1
                            # label_bbox[i, y // dsr, x // dsr, 3] = bb_list[rand_digit]['y2'] - \
                            #                                        bb_list[rand_digit]['y1'] + 1

    # 检查label
    # check_label = label_c.flatten()
    # count_label = Counter(check_label)
    # if len(count_label) != 21:
    # print('Label is less than 21.')
    # print(count_label)
    # os._exit(0)
    return label_c, label_bbox


def calculateCrossRatio_float(rec1, rec2):
    """
    computing cross ratio
    :param rec1: (x0, y0, w0, h0), which reflects
            (top, left, width, height)
    :param rec2: (x1, y1, w1, h1)
    :return: scala value of cross ratio
    It is special!!!
    rec1: grid
    rec2: bbox
    """
    if rec1[0] + rec1[2] < rec2[0] or rec2[0] + rec2[2] < rec1[0]:
        cross_ratio_x = 0
    else:
        cross_len_x = min(rec1[0] + rec1[2], rec2[0] + rec2[2]) - max(rec1[0], rec2[0])
        min_len_x = min(rec1[2], rec2[2])
        if min_len_x == 0:
            cross_ratio_x = 1
        else:
            cross_ratio_x = cross_len_x / min_len_x

    if rec1[1] + rec1[3] < rec2[1] or rec2[1] + rec2[3] < rec1[1]:
        cross_ratio_y = 0
    else:
        cross_len_y = min(rec1[1] + rec1[3], rec2[1] + rec2[3]) - max(rec1[1], rec2[1])
        min_len_y = min(rec1[3], rec2[3])
        if min_len_y == 0:
            cross_ratio_y = 1
        else:
            cross_ratio_y = cross_len_y / min_len_y

    cross_ratio = cross_ratio_x * cross_ratio_y
    return cross_ratio


def relabel_2D_polygon(dict_img, height, width, dsr_x, dsr_y, cls, cr, ignore=False):
    label_c = np.ones((len(dict_img), height // dsr_y, width // dsr_x, 1)) * cls
    label_bbox = np.zeros((len(dict_img), height // dsr_y, width // dsr_x, 8))

    print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        # for i in range(len(dict_img)):
        bb_list = dict_img[i]
        for y in range(0, height, dsr_y):  # y代表纵向
            for x in range(0, width, dsr_x):  # x代表横向
                grid = [x, y, x + dsr_x - 1, y, x + dsr_x - 1, y + dsr_y - 1, x, y + dsr_y - 1]
                # print(grid)
                cross_tmp = 0
                box_area_tmp = 0
                for j in range(len(bb_list)):
                    if ignore is False or (ignore and bb_list[j]['ignore'] == 0):
                        bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                        bb_list[j]['x2'], bb_list[j]['y2'],
                                        bb_list[j]['x3'], bb_list[j]['y3'],
                                        bb_list[j]['x4'], bb_list[j]['y4']]
                        cross_ratio = pc.calculate_polygon_IoM(grid, bounding_box)
                        # cross_ratio = pc.calculate_polygon_IoM_slow(grid, bounding_box, height, width)
                        box_area = pc.calculate_polygon_area(bounding_box)
                        if cross_ratio >= cr:
                            if cross_ratio > cross_tmp:
                                label_c[i, y // dsr_y, x // dsr_x, 0] = int(bb_list[j]['class'])
                                label_bbox[i, y // dsr_y, x // dsr_x, 0] = bb_list[j]['x1'] - x
                                label_bbox[i, y // dsr_y, x // dsr_x, 1] = bb_list[j]['y1'] - y
                                label_bbox[i, y // dsr_y, x // dsr_x, 2] = bb_list[j]['x2'] - x
                                label_bbox[i, y // dsr_y, x // dsr_x, 3] = bb_list[j]['y2'] - y
                                label_bbox[i, y // dsr_y, x // dsr_x, 4] = bb_list[j]['x3'] - x
                                label_bbox[i, y // dsr_y, x // dsr_x, 5] = bb_list[j]['y3'] - y
                                label_bbox[i, y // dsr_y, x // dsr_x, 6] = bb_list[j]['x4'] - x
                                label_bbox[i, y // dsr_y, x // dsr_x, 7] = bb_list[j]['y4'] - y
                                box_area_tmp = box_area
                                cross_tmp = cross_ratio
                            elif cross_ratio == cross_tmp and box_area < box_area_tmp:
                                label_c[i, y // dsr_y, x // dsr_x, 0] = int(bb_list[j]['class'])
                                label_bbox[i, y // dsr_y, x // dsr_x, 0] = bb_list[j]['x1'] - x
                                label_bbox[i, y // dsr_y, x // dsr_x, 1] = bb_list[j]['y1'] - y
                                label_bbox[i, y // dsr_y, x // dsr_x, 2] = bb_list[j]['x2'] - x
                                label_bbox[i, y // dsr_y, x // dsr_x, 3] = bb_list[j]['y2'] - y
                                label_bbox[i, y // dsr_y, x // dsr_x, 4] = bb_list[j]['x3'] - x
                                label_bbox[i, y // dsr_y, x // dsr_x, 5] = bb_list[j]['y3'] - y
                                label_bbox[i, y // dsr_y, x // dsr_x, 6] = bb_list[j]['x4'] - x
                                label_bbox[i, y // dsr_y, x // dsr_x, 7] = bb_list[j]['y4'] - y
                                box_area_tmp = box_area

    return label_c, label_bbox


def calculateCrossRatio_special(rec1, rec2):
    """
    computing cross ratio
    :param rec1: (x0, y0, w0, h0), which reflects
            (top, left, width, height)
    :param rec2: (x1, y1, w1, h1)
    :return: scala value of cross ratio
    It is special!!!
    rec1: grid
    rec2: bbox
    """
    set_grid_x = set(range(rec1[0], rec1[0] + rec1[2]))
    set_grid_y = set(range(rec1[1], rec1[1] + rec1[3]))
    set_bbox_x = set(range(rec2[0], rec2[0] + rec2[2]))
    set_bbox_y = set(range(rec2[1], rec2[1] + rec2[3]))

    special_cross_ratio_x = len(set_grid_x & set_bbox_x) / min(len(set_grid_x), len(set_bbox_x))
    special_cross_ratio_y = len(set_grid_y & set_bbox_y) / min(len(set_grid_y), len(set_bbox_y))
    special_cross_ratio = special_cross_ratio_x * special_cross_ratio_y
    if special_cross_ratio > 1:
        print('The cross ratio is wrong.')
        os._exit(0)

    return special_cross_ratio


def relabel_2D_bbox(dict_img, height, width, dsr_x, dsr_y, cls, cr):
    label_c = np.ones((len(dict_img), height // dsr_y, width // dsr_x, 1)) * cls
    label_bbox = np.zeros((len(dict_img), height // dsr_y, width // dsr_x, 4))

    print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        bb_list = dict_img[i]
        for y in range(0, height, dsr_y):  # 32, 4, y代表纵向
            for x in range(0, width, dsr_x):  # 160, 4, x代表横向
                grid = [x, y, dsr_x, dsr_y]
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                box_area_tmp = 0
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    box_area = bounding_box[2] * bounding_box[3]
                    # print(cross_ratio)
                    if cross_ratio >= cr:
                        if cross_ratio > cross_tmp:
                            label_c[i, y // dsr_y, x // dsr_x, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr_y, x // dsr_x, 0] = bb_list[j]['x1'] - x
                            label_bbox[i, y // dsr_y, x // dsr_x, 1] = bb_list[j]['y1'] - y
                            label_bbox[i, y // dsr_y, x // dsr_x, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr_y, x // dsr_x, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            box_area_tmp = box_area
                            cross_tmp = cross_ratio
                        elif cross_ratio == cross_tmp and box_area < box_area_tmp:
                            label_c[i, y // dsr_y, x // dsr_x, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr_y, x // dsr_x, 0] = bb_list[j]['x1'] - x
                            label_bbox[i, y // dsr_y, x // dsr_x, 1] = bb_list[j]['y1'] - y
                            label_bbox[i, y // dsr_y, x // dsr_x, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr_y, x // dsr_x, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            box_area_tmp = box_area

    return label_c, label_bbox


def relabel_2D_bbox_concentrate(dict_img, height, width, dsr_x, dsr_y, cls, cr):
    label_c = np.ones((len(dict_img), height // dsr_y, width // dsr_x, 1)) * cls
    label_bbox = np.zeros((len(dict_img), height // dsr_y, width // dsr_x, 4))

    print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        bb_list = dict_img[i]
        for y in range(0, height, dsr_y):  # 32, 4, y代表纵向
            for x in range(0, width, dsr_x):  # 160, 4, x代表横向
                grid = [x, y, dsr_x, dsr_y]
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                # box_area_tmp = 0
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    # box_area = bounding_box[2] * bounding_box[3]
                    # print(cross_ratio)
                    if cross_ratio >= cr:
                        if cross_ratio > cross_tmp:
                            label_c[i, y // dsr_y, x // dsr_x, 0] = int(bb_list[j]['class'])
                            label_bbox[i, y // dsr_y, x // dsr_x, 0] = bb_list[j]['x1'] - x
                            label_bbox[i, y // dsr_y, x // dsr_x, 1] = bb_list[j]['y1'] - y
                            label_bbox[i, y // dsr_y, x // dsr_x, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr_y, x // dsr_x, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            # box_area_tmp = box_area
                            cross_tmp = cross_ratio
                        elif cross_ratio == cross_tmp:  # and box_area < box_area_tmp:
                            label_c[i, y // dsr_y, x // dsr_x, 0] = cls
                            label_bbox[i, y // dsr_y, x // dsr_x, 0] = 0
                            label_bbox[i, y // dsr_y, x // dsr_x, 1] = 0
                            label_bbox[i, y // dsr_y, x // dsr_x, 2] = 0
                            label_bbox[i, y // dsr_y, x // dsr_x, 3] = 0
                            # box_area_tmp = box_area

    return label_c, label_bbox


def center_ness(l, r, t, b):
    return math.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))


def relabel_2D_center_ness(dict_img, height, width, dsr_x, dsr_y, cls, cr):
    label_cls = np.ones((len(dict_img), height // dsr_y, width // dsr_x, 1)) * cls
    label_cen = np.zeros((len(dict_img), height // dsr_y, width // dsr_x, 1))
    label_bbox = np.zeros((len(dict_img), height // dsr_y, width // dsr_x, 4))

    print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        bb_list = dict_img[i]
        for y in range(0, height, dsr_y):  # 32, 4, y代表纵向
            for x in range(0, width, dsr_x):  # 160, 4, x代表横向
                grid = [x, y, dsr_x, dsr_y]
                grid_cen_x = (x + x + dsr_x - 1) / 2
                grid_cen_y = (y + y + dsr_y - 1) / 2
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                # box_area_tmp = 0
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    # box_area = bounding_box[2] * bounding_box[3]
                    # print(cross_ratio)
                    if cross_ratio >= cr:
                        if cross_ratio > cross_tmp:
                            label_cls[i, y // dsr_y, x // dsr_x, 0] = int(bb_list[j]['class'])
                            left = max(grid_cen_x - bb_list[j]['x1'], 0)
                            right = max(bb_list[j]['x2'] - grid_cen_x, 0)
                            top = max(grid_cen_y - bb_list[j]['y1'], 0)
                            bottom = max(bb_list[j]['y2'] - grid_cen_y, 0)
                            label_cen[i, y // dsr_y, x // dsr_x, 0] = center_ness(left, right, top, bottom)
                            label_bbox[i, y // dsr_y, x // dsr_x, 0] = bb_list[j]['x1'] - x
                            label_bbox[i, y // dsr_y, x // dsr_x, 1] = bb_list[j]['y1'] - y
                            label_bbox[i, y // dsr_y, x // dsr_x, 2] = bb_list[j]['x2'] - bb_list[j]['x1'] + 1
                            label_bbox[i, y // dsr_y, x // dsr_x, 3] = bb_list[j]['y2'] - bb_list[j]['y1'] + 1
                            # box_area_tmp = box_area
                            cross_tmp = cross_ratio
                        elif cross_ratio == cross_tmp:  # and box_area < box_area_tmp:
                            label_cls[i, y // dsr_y, x // dsr_x, 0] = cls
                            label_cen[i, y // dsr_y, x // dsr_x, 0] = 0
                            label_bbox[i, y // dsr_y, x // dsr_x, 0] = 0
                            label_bbox[i, y // dsr_y, x // dsr_x, 1] = 0
                            label_bbox[i, y // dsr_y, x // dsr_x, 2] = 0
                            label_bbox[i, y // dsr_y, x // dsr_x, 3] = 0
                            # box_area_tmp = box_area

    return label_cls, label_cen, label_bbox


def relabel_2D_location(dict_img, height, width, dsr_x, dsr_y, cls, cr):
    label_cls = np.ones((len(dict_img), height // dsr_y, width // dsr_x, 1)) * cls
    label_location = np.zeros((len(dict_img), height // dsr_y, width // dsr_x, 1))

    print('Relabeling start...')
    for i in tqdm(range(len(dict_img))):
        bb_list = dict_img[i]
        for y in range(0, height, dsr_y):  # 32, 4, y代表纵向
            for x in range(0, width, dsr_x):  # 160, 4, x代表横向
                grid = [x, y, dsr_x, dsr_y]
                grid_cen_x = (x + x + dsr_x - 1) / 2
                grid_cen_y = (y + y + dsr_y - 1) / 2
                # w_spilt = set(range(w, w + dsr))
                cross_tmp = 0
                box_area_tmp = 0
                for j in range(len(bb_list)):
                    bounding_box = [bb_list[j]['x1'], bb_list[j]['y1'],
                                    bb_list[j]['x2'] - bb_list[j]['x1'] + 1,
                                    bb_list[j]['y2'] - bb_list[j]['y1'] + 1]
                    cross_ratio = calculateCrossRatio_special(grid, bounding_box)
                    box_area = bounding_box[2] * bounding_box[3]
                    # print(cross_ratio)
                    if cross_ratio >= cr:
                        if cross_ratio > cross_tmp:
                            label_cls[i, y // dsr_y, x // dsr_x, 0] = int(bb_list[j]['class'])
                            box_cen_x = (bb_list[j]['x1'] + bb_list[j]['x2']) / 2
                            box_cen_y = (bb_list[j]['y1'] + bb_list[j]['y2']) / 2
                            if grid_cen_x < box_cen_x and grid_cen_y <= box_cen_y:
                                label_location[i, y // dsr_y, x // dsr_x, 0] = 1
                            elif grid_cen_x >= box_cen_x and grid_cen_y < box_cen_y:
                                label_location[i, y // dsr_y, x // dsr_x, 0] = 2
                            elif grid_cen_x <= box_cen_x and grid_cen_y > box_cen_y:
                                label_location[i, y // dsr_y, x // dsr_x, 0] = 3
                            elif grid_cen_x > box_cen_x and grid_cen_y >= box_cen_y:
                                label_location[i, y // dsr_y, x // dsr_x, 0] = 4
                            box_area_tmp = box_area
                            cross_tmp = cross_ratio
                        elif cross_ratio == cross_tmp and box_area < box_area_tmp:
                            label_cls[i, y // dsr_y, x // dsr_x, 0] = int(bb_list[j]['class'])
                            box_cen_x = (bb_list[j]['x1'] + bb_list[j]['x2']) / 2
                            box_cen_y = (bb_list[j]['y1'] + bb_list[j]['y2']) / 2
                            if grid_cen_x < box_cen_x and grid_cen_y <= box_cen_y:
                                label_location[i, y // dsr_y, x // dsr_x, 0] = 1
                            elif grid_cen_x >= box_cen_x and grid_cen_y < box_cen_y:
                                label_location[i, y // dsr_y, x // dsr_x, 0] = 2
                            elif grid_cen_x <= box_cen_x and grid_cen_y > box_cen_y:
                                label_location[i, y // dsr_y, x // dsr_x, 0] = 3
                            elif grid_cen_x > box_cen_x and grid_cen_y >= box_cen_y:
                                label_location[i, y // dsr_y, x // dsr_x, 0] = 4
                            box_area_tmp = box_area

    return label_cls, label_location