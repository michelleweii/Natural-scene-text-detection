import random
import cv2
import math
import os
import numpy as np
from skimage.util import random_noise
from skimage import exposure
from PIL import Image, ImageEnhance


# bboxes是list
def bboxesTransform(bboxes, hr, wr, h, w):
    for i in range(len(bboxes)):
        bboxes[i]['x1'] = min(max(int(math.ceil((float(bboxes[i]['x1']) + 1) * wr - 1)), 0), w - 1)
        bboxes[i]['x2'] = max(min(int(math.floor(float(bboxes[i]['x2']) * wr)), w - 1), bboxes[i]['x1'])
        bboxes[i]['y1'] = min(max(int(math.ceil((float(bboxes[i]['y1']) + 1) * hr - 1)), 0), h - 1)
        bboxes[i]['y2'] = max(min(int(math.floor(float(bboxes[i]['y2']) * hr)), h - 1), bboxes[i]['y1'])
        if bboxes[i]['x2'] < bboxes[i]['x1'] or bboxes[i]['y2'] < bboxes[i]['y1']:
            print('The box is wrong.')
            os._exit(0)
        if bboxes[i]['x1'] < 0 or bboxes[i]['y1'] < 0 or bboxes[i]['x2'] > w or bboxes[i]['y2'] > h:
            print('The box is out of image.')
            os._exit(0)
    return bboxes


def getEvenNumber(a):
    if math.floor(a) % 2 == 0:
        a = math.floor(a)
    else:
        a = math.floor(a) + 1
    return a


class TextAugment():
    def __init__(self, max_width=None):
        self.max_width = max_width

    def resizeTextWidth(self, img, bboxes, width_scale=0.35):
        w = img.shape[1]
        rand_a = random.uniform((-1) * width_scale, (-1) * width_scale + 0.1)
        rand_b = random.uniform(width_scale - 0.1, width_scale)
        rand_num = random.sample([rand_a, rand_b], 1)[0]
        resize_w = int(w * (1 + rand_num))
        # resize_w = getEvenNumber(resize_w)
        # resize_w = min(resize_w, self.max_width)
        img = cv2.resize(img, (resize_w, img.shape[0]))
        ratio_w = resize_w / w
        bboxes = bboxesTransform(bboxes, 1, ratio_w, img.shape[0], resize_w)
        return img, bboxes

    def resizeTextHeight(self, img, bboxes, height_scale=0.35):
        h = img.shape[0]
        rand_a = random.uniform((-1) * height_scale, (-1) * height_scale + 0.1)
        rand_b = random.uniform(height_scale - 0.1, height_scale)
        rand_num = random.sample([rand_a, rand_b], 1)[0]
        resize_h = int(h * (1 + rand_num))
        # resize_h = getEvenNumber(resize_h)
        img = cv2.resize(img, (img.shape[1], resize_h))
        ratio_h = resize_h / h
        bboxes = bboxesTransform(bboxes, ratio_h, 1, resize_h, img.shape[1])
        return img, bboxes

    def resizeText(self, img, bboxes, scale=0.3):
        w = img.shape[1]
        h = img.shape[0]
        rand_a = random.uniform((-1) * scale, (-1) * scale + 0.1)
        rand_b = random.uniform(scale - 0.1, scale)
        rand_num = random.sample([rand_a, rand_b], 1)[0]
        resize_w = int(w * (1 + rand_num))
        # resize_w = getEvenNumber(resize_w)
        resize_h = int(h * (1 + rand_num))
        # resize_h = getEvenNumber(resize_h)
        img = cv2.resize(img, (resize_w, resize_h))
        ratio_w = resize_w / w
        ratio_h = resize_h / h
        bboxes = bboxesTransform(bboxes, ratio_h, ratio_w, resize_h, resize_w)
        return img, bboxes

    def resizeImage(self, img, bboxes, resize_h, resize_w):
        w = img.shape[1]
        h = img.shape[0]
        img = cv2.resize(img, (resize_w, resize_h))
        ratio_w = resize_w / w
        ratio_h = resize_h / h
        bboxes = bboxesTransform(bboxes, ratio_h, ratio_w, resize_h, resize_w)
        return img, bboxes

    # 旋转
    def rotateText(self, img, bboxes, angle=3, scale=1):
        '''
        参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
        输入:
            img:图像array,(h,w,c)
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
            angle:旋转角度
            scale:默认1
        输出:
            rot_img:旋转后的图像array
            rot_bboxes:旋转后的boundingbox坐标list
        '''
        # ---------------------- 旋转图像 ----------------------
        # scale = random.uniform(0.8, 1.2)

        rand_a = random.uniform((-1) * angle, (-1) * angle + 2)
        rand_b = random.uniform(angle - 2, angle)
        angle = random.sample([rand_a, rand_b], 1)[0]

        w = img.shape[1]
        h = img.shape[0]
        # 角度变弧度
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # 仿射变换
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4,
                                 borderValue=color)

        # ---------------------- 矫正bbox坐标 ----------------------
        # rot_mat是最终的旋转矩阵
        # 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
        rot_bboxes = list()
        for bbox in bboxes:
            xmin = bbox['x1']
            ymin = bbox['y1']
            xmax = bbox['x2']
            ymax = bbox['y2']
            point1 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymin, 1]))
            point2 = np.dot(rot_mat, np.array([xmax, (ymin + ymax) / 2, 1]))
            point3 = np.dot(rot_mat, np.array([(xmin + xmax) / 2, ymax, 1]))
            point4 = np.dot(rot_mat, np.array([xmin, (ymin + ymax) / 2, 1]))
            # 合并np.array
            concat = np.vstack((point1, point2, point3, point4))
            # 改变array类型
            concat = concat.astype(np.float32)

            # parameter = h * 0.001 * angle * (xmax - xmin + 1) / (ymax - ymin + 1)
            # 得到旋转后的坐标
            rx, ry, rw, rh = cv2.boundingRect(concat)
            rx_min = round(rx, 0)
            ry_min = round(ry, 0)
            rx_max = round(rx + rw, 0)
            ry_max = round(ry + rh, 0)
            # 加入list中
            rot_bboxes.append({'class': bbox['class'],
                               'x1': rx_min, 'y1': ry_min,
                               'x2': rx_max, 'y2': ry_max})

        return rot_img, rot_bboxes

    # def affineText(self, img, bboxes, base=10):
    #     w = img.shape[1]
    #     h = img.shape[0]
    #     rand_num = random.randint(-3, 3)
    #     base += rand_num
    #     pts1 = np.float32([[base, base], [base, base + 10], [base + 10, base]])
    #     pts2 =

    def addNoise(self, img):
        '''
        输入:
            img:图像array
        输出:
            加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
        '''
        # random.seed(int(time.time()))
        # return random_noise(img, mode='gaussian', seed=int(time.time()), clip=True)*255
        return random_noise(img, mode='gaussian', clip=True) * 255

    def changeLight(self, img):
        # random.seed(int(time.time()))
        flag = random.uniform(0.5, 1.5)  # flag>1为调暗,小于1为调亮
        return exposure.adjust_gamma(img, flag)

    # 裁剪
    def cropText(self, img, bboxes):
        '''
        裁剪后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_img:裁剪后的图像array
            crop_bboxes:裁剪后的bounding box的坐标list
        '''
        # ---------------------- 裁剪图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w  # 裁剪后的包含所有目标框的最小的框
        x_max = 1
        y_min = h
        y_max = 1
        for bbox in bboxes:
            x_min = min(x_min, bbox['x1'])
            y_min = min(y_min, bbox['y1'])
            x_max = max(x_max, bbox['x2'])
            y_max = max(y_max, bbox['y2'])

        d_to_left = x_min - 1  # 包含所有目标框的最小框到左边的距离
        d_to_right = w - x_max  # 包含所有目标框的最小框到右边的距离
        d_to_top = y_min - 1  # 包含所有目标框的最小框到顶端的距离
        d_to_bottom = h - y_max  # 包含所有目标框的最小框到底部的距离

        # 随机扩展这个最小框
        crop_x_min = int(x_min - random.uniform(0, d_to_left))
        crop_y_min = int(y_min - random.uniform(0, d_to_top))
        crop_x_max = int(x_max + random.uniform(0, d_to_right))
        crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

        # 随机扩展这个最小框 , 防止别裁的太小
        # crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
        # crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
        # crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
        # crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))

        # 确保不要越界
        crop_x_min = max(1, crop_x_min)
        crop_y_min = max(1, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        # print('x_max:', x_max)
        # print('x_min:', x_min)
        # print('y_max:', y_max)
        # print('y_min:', y_min)
        # print('crop_x_max:', crop_x_max)
        # print('crop_x_min:', crop_x_min)
        # print('crop_y_max:', crop_y_max)
        # print('crop_y_min:', crop_y_min)

        crop_img = img[crop_y_min - 1:crop_y_max, crop_x_min - 1:crop_x_max]

        # ---------------------- 裁剪boundingbox ----------------------
        # 裁剪后的boundingbox坐标计算
        crop_bboxes = list()
        for bbox in bboxes:
            # crop_bboxes.append([bbox[0] - crop_x_min + 1, bbox[1] - crop_y_min + 1,
            #                     bbox[2] - crop_x_min + 1, bbox[3] - crop_y_min + 1,
            #                     bbox[4], bbox[5]])
            # 加入list中
            crop_bboxes.append({'class': bbox['class'],
                                'x1': bbox['x1'] - crop_x_min + 1,
                                'y1': bbox['y1'] - crop_y_min + 1,
                                'x2': bbox['x2'] - crop_x_min + 1,
                                'y2': bbox['y2'] - crop_y_min + 1})

        return crop_img, crop_bboxes


    def randomColor(self, img, bboxes):

        img = Image.fromarray(img)

        random_factor = np.random.randint(0, 31) / 10.  # 随机因子

        color_image = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度

        random_factor = np.random.randint(10, 21) / 10.  # 随机因子

        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度

        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子

        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度

        random_factor = np.random.randint(0, 31) / 10.  # 随机因子

        imgss = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

        img = np.array(imgss)

        return img, bboxes  # 调整图像锐度


    def randomSpot(self, img, bboxes):

        foreground = Image.open("../dataset/light/" + str(random.randint(0, 1)) + ".png")
        background = Image.fromarray(img)
        bb_cnt = len(bboxes)
        bb_index = random.randint(0, bb_cnt - 1)
        x = random.randint(bboxes[bb_index]['x1'], bboxes[bb_index]['x2']) - int(img.shape[1] / 2)
        # print(x)
        # print(bboxes[bb_index]['x1'])
        # print(bboxes[bb_index]['x2'])
        y = random.randint(bboxes[bb_index]['y1'], bboxes[bb_index]['y2']) - int(img.shape[0] / 2)
        # foreground = foreground.resize((int((bboxes[bb_index]['x2'] - bboxes[bb_index]['x1']) * 0.3 + 1), int((bboxes[bb_index]['y2'] - bboxes[bb_index]['y1']) * 0.7 + 1)))
        background.paste(foreground, (x, y), foreground)
        img = np.array(background)

        return img, bboxes

    # 平移
    def shiftText(self, img, bboxes):
        '''
        参考:https://blog.csdn.net/sty945/article/details/79387054
        平移后的图片要包含所有的框
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            shift_img:平移后的图像array
            shift_bboxes:平移后的bounding box的坐标list
        '''
        # ---------------------- 平移图像 ----------------------
        w = img.shape[1]
        h = img.shape[0]
        x_min = w - 1  # 裁剪后的包含所有目标框的最小的框
        x_max = 0
        y_min = h - 1
        y_max = 0
        for bbox in bboxes:
            # x_min = min(x_min, bbox[0])
            # y_min = min(y_min, bbox[1])
            # x_max = max(x_max, bbox[2])
            # y_max = max(y_max, bbox[3])
            x_min = min(x_min, bbox['x1'])
            y_min = min(y_min, bbox['y1'])
            x_max = max(x_max, bbox['x2'])
            y_max = max(y_max, bbox['y2'])

        d_to_left = x_min - 1  # 包含所有目标框的最大左移动距离
        d_to_right = w - x_max  # 包含所有目标框的最大右移动距离
        d_to_top = y_min - 1  # 包含所有目标框的最大上移动距离
        d_to_bottom = h - y_max  # 包含所有目标框的最大下移动距离

        x = random.randint(-d_to_left, d_to_right)
        y = random.randint(-d_to_top, d_to_bottom)
        x = int(x / 3)
        y = int(y / 3)
        M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # ---------------------- 平移boundingbox ----------------------
        shift_bboxes = list()
        for bbox in bboxes:
            # shift_bboxes.append([bbox[0] + x, bbox[1] + y, bbox[2] + x,
            #                      bbox[3] + y, bbox[4], bbox[5]])
            shift_bboxes.append({'class': bbox['class'],
                                 'x1': bbox['x1'] + x,
                                 'y1': bbox['y1'] + y,
                                 'x2': bbox['x2'] + x,
                                 'y2': bbox['y2'] + y})

        return shift_img, shift_bboxes
