import math
from tqdm import tqdm
import os
import cv2
import numpy as np
import copy


# bboxes是list
def bboxesTransform(bboxes, size, pad):
    (hr, wr, h, w, h_o, w_o) = size
    (top_pad, bottom_pad, left_pad, right_pad) = pad

    for i in range(len(bboxes)):
        bboxes[i]['x1'] = min(max(int(math.ceil((float(bboxes[i]['x1']) + 1) * wr - 1)), 0), w - 1) + left_pad
        bboxes[i]['x2'] = max(min(int(math.floor(float(bboxes[i]['x2']) * wr)), w - 1), 0) + left_pad
        # bboxes[i]['x3'] = max(min(int(math.floor(float(bboxes[i]['x3']) * wr)), w - 1), 0) + left_pad
        # bboxes[i]['x4'] = min(max(int(math.ceil((float(bboxes[i]['x4']) + 1) * wr - 1)), 0), w - 1) + left_pad
        bboxes[i]['y1'] = min(max(int(math.ceil((float(bboxes[i]['y1']) + 1) * hr - 1)), 0), h - 1) + top_pad
        bboxes[i]['y2'] = min(max(int(math.ceil((float(bboxes[i]['y2']) + 1) * hr - 1)), 0), h - 1) + top_pad
        # bboxes[i]['y3'] = max(min(int(math.floor(float(bboxes[i]['y3']) * hr)), h - 1), 0) + top_pad
        # bboxes[i]['y4'] = max(min(int(math.floor(float(bboxes[i]['y4']) * hr)), h - 1), 0) + top_pad
        # if bboxes[i]['x2'] < bboxes[i]['x1'] or bboxes[i]['y2'] < bboxes[i]['y1']:
        #     print('The box is wrong.')
        #     os._exit(0)
        # if bboxes[i]['x1'] < 0 or bboxes[i]['y1'] < 0 or bboxes[i]['x2'] > w or bboxes[i]['y2'] > h:
        #     print('The box is out of image.')
        #     os._exit(0)
    return bboxes


# bboxes是list
def bboxesTransform_inverse(bboxes, size, pad, isWH=False):
    (hr, wr, h, w, h_o, w_o) = size
    (top_pad, bottom_pad, left_pad, right_pad) = pad

    for i in range(len(bboxes)):
        if isWH:
            bboxes[i][2] += bboxes[i][0] - 1
            bboxes[i][3] += bboxes[i][1] - 1
        bboxes[i][0] = min(max(int(math.ceil((float(bboxes[i][0]) - left_pad + 1) / wr - 1)), 0), w_o - 1)
        # bboxes[i][1] = max(min(int(math.floor((float(bboxes[i][1]) - left_pad) / wr)), w - 1), 0)
        bboxes[i][2] = max(min(int(math.floor((float(bboxes[i][2]) - left_pad) / wr)), w_o - 1), 0)
        # bboxes[i][3] = min(max(int(math.ceil((float(bboxes[i][3]) - left_pad + 1) / wr - 1)), 0), w - 1)
        bboxes[i][1] = min(max(int(math.ceil((float(bboxes[i][1]) - top_pad + 1) / hr - 1)), 0), h_o - 1)
        # bboxes[i][5] = min(max(int(math.ceil((float(bboxes[i][5]) - top_pad + 1) / hr - 1)), 0), h - 1)
        bboxes[i][3] = max(min(int(math.floor((float(bboxes[i][3]) - top_pad) / hr)), h_o - 1), 0)
        # bboxes[i][7] = max(min(int(math.floor((float(bboxes[i][7]) - top_pad) / hr)), h - 1), 0)

    return bboxes


# x1,y1,x2,y2,x3,y3,x4,y4
def bboxesRotate90anti(bboxes, h):
    for i in range(len(bboxes)):
        x1 = bboxes[i]['x1']
        y1 = bboxes[i]['y1']
        bboxes[i]['x1'] = bboxes[i]['y2']
        bboxes[i]['y1'] = str(int(h - float(bboxes[i]['x2'])))
        bboxes[i]['x2'] = bboxes[i]['y3']
        bboxes[i]['y2'] = str(int(h - float(bboxes[i]['x3'])))
        bboxes[i]['x3'] = bboxes[i]['y4']
        bboxes[i]['y3'] = str(int(h - float(bboxes[i]['x4'])))
        bboxes[i]['x4'] = y1
        bboxes[i]['y4'] = str(int(h - float(x1)))
    return bboxes


# x1,y1,x2,y2,x3,y3,x4,y4
def bboxesRotate90anti_inverse(bboxes, h):
    for i in range(len(bboxes)):
        x4 = bboxes[i][6]
        y4 = bboxes[i][7]
        bboxes[i][6] = int(h - float(bboxes[i][5]))
        bboxes[i][7] = int(bboxes[i][4])
        bboxes[i][4] = int(h - float(bboxes[i][3]))
        bboxes[i][5] = int(bboxes[i][2])
        bboxes[i][2] = int(h - float(bboxes[i][1]))
        bboxes[i][3] = int(bboxes[i][0])
        bboxes[i][0] = int(y4)
        bboxes[i][1] = int(h - float(x4))
    return bboxes


def exchange(x, y):
    return y, x


def sampleResize_dif_wh(x_path=None, y=None, im_size=None, short_side=800, long_side=1024,
                        resizeh=1024, resizew=1024, channel=3, resizeX=False, resizeY=False,
                        sizeinfo=False, h_short=False):
    if y is None:
        length = len(x_path)
    else:
        length = len(y)

    if sizeinfo:
        resizeY = True

    if resizeX or resizeY:
        imgs = []
        info_s = []
        info_p = []
        # for m in tqdm(range(length)):
        for m in range(length):
            if resizeX:
                if channel == 1:
                    img = cv2.imread(x_path[m], 0)
                else:
                    img = cv2.imread(x_path[m])
                h_ori = img.shape[0]
                w_ori = img.shape[1]
                if h_short:
                    if h_ori > w_ori:
                        img = np.rot90(img)
                        h_ori = img.shape[0]
                        w_ori = img.shape[1]

            if resizeY:
                h_ori = im_size[m][0]
                w_ori = im_size[m][1]
                if h_short:
                    if h_ori > w_ori:
                        h_ori = im_size[m][1]
                        w_ori = im_size[m][0]
                        y[m] = bboxesRotate90anti(y[m], h_ori)

            if h_ori <= w_ori:
                height_ratio = short_side / h_ori
                w = round(height_ratio * w_ori, 0)
                h = short_side
                width_ratio = w / w_ori
                if w > long_side:
                    width_ratio = long_side / w_ori
                    h = round(width_ratio * h_ori, 0)
                    w = long_side
                    height_ratio = h / h_ori
                    if h > short_side:
                        h = short_side
                        height_ratio = h / h_ori
                        # print('Something wrong.')
                        # os._exit(0)
            else:
                width_ratio = short_side / w_ori
                h = round(width_ratio * h_ori, 0)
                w = short_side
                height_ratio = h / h_ori
                resizeh, resizew = exchange(resizeh, resizew)
                if h > long_side:
                    height_ratio = long_side / h_ori
                    w = round(height_ratio * w_ori, 0)
                    h = long_side
                    width_ratio = w / w_ori
                    if w > short_side:
                        w = short_side
                        width_ratio = w / w_ori
                        # print('Something wrong.')
                        # os._exit(0)

            w = int(w)
            h = int(h)
            top_pad = int((resizeh - h) // 2)
            bottom_pad = int(resizeh - h - top_pad)
            left_pad = int((resizew - w) // 2)
            right_pad = int(resizew - w - left_pad)
            size_info = (height_ratio, width_ratio, h, w)
            pad_info = (top_pad, bottom_pad, left_pad, right_pad)

            if sizeinfo:
                info_s.append(size_info)
                info_p.append(pad_info)

            if resizeX:  # for path in x_path:
                img = cv2.resize(img, (w, h))
                print(size_info, pad_info)
                padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
                img = np.pad(img, padding, mode='constant', constant_values=0)
                img = img.astype('float32')
                img = img / 255
                # img -= 1.
                imgs.append(img)

            if resizeY:
                y[m] = bboxesTransform(y[m], size_info, pad_info)

        if sizeinfo:
            return info_s, info_p
        if resizeX and resizeY is False:
            return np.array(imgs).reshape(len(x_path), resizeh, resizew, channel)
        if resizeY and resizeX is False:
            return y
        if resizeX and resizeY:
            return np.array(imgs).reshape(len(x_path), resizeh, resizew, channel), y
    else:
        print('Nothing changes, please check.')


def sampleResize(x_path=None, y=None, im_size=None, short_side=800, long_side=1024,
                 resizeh=1024, resizew=1024, channel=3, resizeX=False, resizeY=False,
                 sizeinfo=False, h_short=False):
    if y is None:
        length = len(x_path)
    else:
        length = len(y)

    if sizeinfo:
        resizeY = True

    if resizeX or resizeY:
        imgs = []
        info_s = []
        info_p = []
        # for m in tqdm(range(length)):
        for m in range(length):
            if resizeX:
                if channel == 1:
                    img = cv2.imread(x_path[m], 0)
                else:
                    img = cv2.imread(x_path[m])
                h_ori = img.shape[0]
                w_ori = img.shape[1]
                if h_short:
                    if h_ori > w_ori:
                        img = np.rot90(img)
                        h_ori = img.shape[0]
                        w_ori = img.shape[1]

            if resizeY:
                h_ori = im_size[m][0]
                w_ori = im_size[m][1]
                if h_short:
                    if h_ori > w_ori:
                        h_ori = im_size[m][1]
                        w_ori = im_size[m][0]
                        y[m] = bboxesRotate90anti(y[m], h_ori)

            height_ratio = short_side / h_ori
            w = round(height_ratio * w_ori, 0)
            h = short_side
            width_ratio = w / w_ori
            if w > long_side:
                width_ratio = long_side / w_ori
                h = round(width_ratio * h_ori, 0)
                w = long_side
                height_ratio = h / h_ori
                if h > short_side:
                    h = short_side
                    height_ratio = h / h_ori

            w = int(w)
            h = int(h)
            top_pad = int((resizeh - h) // 2)
            bottom_pad = int(resizeh - h - top_pad)
            left_pad = int((resizew - w) // 2)
            right_pad = int(resizew - w - left_pad)
            size_info = (height_ratio, width_ratio, h, w, h_ori, w_ori)
            pad_info = (top_pad, bottom_pad, left_pad, right_pad)

            if sizeinfo:
                info_s.append(size_info)
                info_p.append(pad_info)
            else:
                if resizeY:
                    y[m] = bboxesTransform(y[m], size_info, pad_info)

            if resizeX:  # for path in x_path:
                img = cv2.resize(img, (w, h))
                # print(size_info, pad_info)
                padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
                img = np.pad(img, padding, mode='constant', constant_values=0)
                img = img.astype('float32')
                img = img / 255
                # img -= 1.
                imgs.append(img)

        if sizeinfo:
            return info_s, info_p
        elif resizeX and resizeY is False:
            return np.array(imgs).reshape(len(x_path), resizeh, resizew, channel)
        elif resizeY and resizeX is False:
            return y
        elif resizeX and resizeY:
            return np.array(imgs).reshape(len(x_path), resizeh, resizew, channel), y
    else:
        print('Nothing changes, please check.')
