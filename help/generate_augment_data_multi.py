import sys
sys.path.append('../')
import cv2
import os.path
from help import show_picture as sp
from help import text_augment as ta
from tqdm import tqdm
import copy
import random
from multiprocessing import Pool
import math


ShowPicture = False
Channels = 3
TrainTextAug = False
TextAugRandom = False
TrainTextAug_manual = True
test = False
ResizeH = 608
ResizeW = 800
AugLoop = 100  # 229 per Loop
Aug_path = '../dataset/icdar2013_train_augment/'
Aug_GT_path = '../dataset/icdar2013_train_augment_gt/'
if not os.path.exists(Aug_path):
    os.mkdir(Aug_path)
if not os.path.exists(Aug_GT_path):
    os.mkdir(Aug_GT_path)


def classMapping(cls):
    class_list = ['foreground']

    if cls not in class_list:
        print('The current class is wrong.')
        os._exit(0)
    else:
        return str(class_list.index(cls))


# 获取样本信息
def get_data(input_path, PATH_NOTICE=True):
    all_imgs = {}
    classes_count = {}
    class_mapping = {}
    image_num = 0
    image_width_sum = 0
    image_width_max = 0
    image_width_min = 5000
    image_height_sum = 0
    image_height_max = 0
    image_height_min = 5000

    # 把样本以字典形式存储
    with open(input_path, 'r', encoding='utf-8') as f:
        print('Parsing annotation files')
        for line in tqdm(f):
            line_split = line.strip().split('\t')
            # if len(line_split) > 11:

            # print(line_split)
            (filename, x1, y1, x2, y2, class_name) = line_split
            # print(class_name)

            if x1 == x2 == y1 == y2 == '-1':
                continue
            else:
                if int(x1) < 0 or int(x2) < 0 or int(y1) < 0 or int(y2) < 0:
                    print('The coordinates are out of the image.')
                    os._exit(0)

                if PATH_NOTICE:
                    filename = filename[1:]

                class_name = 'foreground'

                if class_name not in classes_count:
                    classes_count[class_name] = 1
                else:
                    classes_count[class_name] += 1

                if classMapping(class_name) not in class_mapping:
                    class_mapping[classMapping(class_name)] = class_name  # 注意，mapping设置

                if filename not in all_imgs:

                    all_imgs[filename] = {}
                    # print (os.path.isfile(filename))
                    img = cv2.imread(filename)
                    # print(img.shape)
                    (rows, cols, channels) = img.shape

                    # print(img.shape)
                    # resize_img = cv2.resize(img, (ResizeShape_width, ResizeShape_height))
                    # height_ratio = ResizeShape_height / rows
                    # width_ratio = ResizeShape_width / cols

                    # (rows, cols, channels) = resize_img.shape
                    # print(resize_img.shape)
                    # input('Stop!')
                    if channels != 3:
                        print('There is a mistake.')
                        os._exit(0)

                    image_num += 1
                    image_width_sum += cols
                    image_height_sum += rows

                    if cols > image_width_max:
                        image_width_max = cols
                    if cols < image_width_min:
                        image_width_min = cols
                    if rows > image_height_max:
                        image_height_max = rows
                    if rows < image_height_min:
                        image_height_min = rows

                    all_imgs[filename]['filepath'] = filename
                    all_imgs[filename]['height'] = rows
                    all_imgs[filename]['width'] = cols
                    all_imgs[filename]['channel'] = channels
                    # all_imgs[filename]['pixel'] = img

                    # all_imgs[filename]['ignore'] = int(ign)

                    # all_imgs[filename]['width_ratio'] = width_ratio
                    # all_imgs[filename]['height_ratio'] = height_ratio

                    all_imgs[filename]['bboxes'] = []
                    # FLAG = np.random.randint(0, 39)
                    # if FLAG > 0:
                    all_imgs[filename]['imageset'] = 'train'
                    # else:
                    #     all_imgs[filename]['imageset'] = 'validation'

                # 每个图有多个框
                # x1 = max(int(math.ceil((float(x1) + 1) * width_ratio - 1)), 0)
                # x2 = min(int(math.floor(float(x2) * width_ratio)), cols - 1)
                # y1 = max(int(math.ceil((float(y1) + 1) * height_ratio - 1)), 0)
                # y2 = min(int(math.floor(float(y2) * height_ratio)), rows - 1)

                all_imgs[filename]['bboxes'].append({'class': classMapping(class_name),
                                                     'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        print('The max width is {}.'.format(image_width_max))
        print('The min width is {}.'.format(image_width_min))
        print('The average width is {}.'.format(image_width_sum / image_num))

        print('The max height is {}.'.format(image_height_max))
        print('The min height is {}.'.format(image_height_min))
        print('The average height is {}.'.format(image_height_sum / image_num))

        return all_data, class_mapping, classes_count


def showPicture(one_image):
    coords = []
    for bn in range(len(one_image['bboxes'])):
        coords.append([one_image['bboxes'][bn]['x1'], one_image['bboxes'][bn]['y1'],
                       one_image['bboxes'][bn]['x2'], one_image['bboxes'][bn]['y2']])
    sp.show_pic(one_image['pixel'], coords)


def generateAugData(img, bboxes, path, iteration, mark):
    img_path = os.path.join(Aug_path, mark + str(iteration) + '_' + os.path.basename(path))
    cv2.imwrite(img_path, img)
    gt_path = os.path.join(Aug_GT_path, mark + str(iteration) + '_' + os.path.basename(path).split('.')[0] + '.txt')
    with open(gt_path, 'w') as f:
        for i in range(len(bboxes)):
            x1 = bboxes[i]['x1']
            y1 = bboxes[i]['y1']
            x2 = bboxes[i]['x2']
            y2 = bboxes[i]['y2']
            class_name = bboxes[i]['class']
            f.write('{} {} {} {} {}\n'.format(x1, y1, x2, y2, class_name))

    tmp_dict = {}
    tmp_dict['filepath'] = 'Augmented file, No path.'
    tmp_dict['width'] = img.shape[1]
    tmp_dict['height'] = img.shape[0]
    tmp_dict['channel'] = Channels
    tmp_dict['pixel'] = img
    tmp_dict['bboxes'] = bboxes
    # showPicture(tmp_dict)
    return tmp_dict


def readImage(path):
    return cv2.imread(path)


def textAugment(all_data, Aug_width=False, Aug_height=False, Aug_resize=False,
                Aug_rotate=False, Aug_rotate_height=False, Aug_rotate_width=False,
                Aug_add_noise=False, Aug_change_light=False, Aug_crop=False,
                Aug_shift=False, iteration=None):
    text_aug = ta.TextAugment()


    if Aug_width:
        for i in range(len(all_data)):
            img, bboxes = text_aug.resizeTextWidth(readImage(all_data[i]['filepath']), all_data[i]['bboxes'])
            tmp_dict = generateAugData(img, bboxes, all_data[i]['filepath'], iteration, mark='wid')
            if ShowPicture:
                showPicture(tmp_dict)
            # images.append(tmp_dict)
        # image_02 =copy.deepcopy(image_)
    # input('stop1')

    if Aug_height:
        for i in range(len(all_data)):
            img, bboxes = text_aug.resizeTextHeight(readImage(all_data[i]['filepath']), all_data[i]['bboxes'])
            tmp_dict = generateAugData(img, bboxes, all_data[i]['filepath'], iteration, mark='hei')
            if ShowPicture:
                showPicture(tmp_dict)
            # images.append(tmp_dict)
        # image_03 =copy.deepcopy(image_02)

    if Aug_resize:
        for i in range(len(all_data)):
            img, bboxes = text_aug.resizeText(readImage(all_data[i]['filepath']), all_data[i]['bboxes'])
            tmp_dict = generateAugData(img, bboxes, all_data[i]['filepath'], iteration, mark='res')
            if ShowPicture:
                showPicture(tmp_dict)
            # images.append(tmp_dict)

    if Aug_rotate:
        for i in range(len(all_data)):
            img, bboxes = text_aug.rotateText(readImage(all_data[i]['filepath']), all_data[i]['bboxes'])
            tmp_dict = generateAugData(img, bboxes, all_data[i]['filepath'], iteration, mark='rot')
            if ShowPicture:
                showPicture(tmp_dict)
            # images.append(tmp_dict)

    if Aug_rotate_height:
        for i in range(len(all_data)):
            img, bboxes = text_aug.resizeTextHeight(readImage(all_data[i]['filepath']), all_data[i]['bboxes'])
            img, bboxes = text_aug.rotateText(img, bboxes)
            tmp_dict = generateAugData(img, bboxes, all_data[i]['filepath'], iteration, mark='rhe')
            if ShowPicture:
                showPicture(tmp_dict)
            # images.append(tmp_dict)

    if Aug_rotate_width:
        for i in range(len(all_data)):
            img, bboxes = text_aug.resizeTextWidth(readImage(all_data[i]['filepath']), all_data[i]['bboxes'])
            img, bboxes = text_aug.rotateText(img, bboxes)
            tmp_dict = generateAugData(img, bboxes, all_data[i]['filepath'], iteration, mark='rwi')
            if ShowPicture:
                showPicture(tmp_dict)
            # images.append(tmp_dict)

    if Aug_add_noise:
        for i in range(len(all_data)):
            img = text_aug.addNoise(readImage(all_data[i]['filepath']))
            tmp_dict = generateAugData(img, all_data[i]['bboxes'], all_data[i]['filepath'], iteration, mark='noi')
            if ShowPicture:
                showPicture(tmp_dict)
            # images.append(tmp_dict)

    if Aug_change_light:
        for i in range(len(all_data)):
            img = text_aug.changeLight(readImage(all_data[i]['filepath']))
            tmp_dict = generateAugData(img, all_data[i]['bboxes'], all_data[i]['filepath'], iteration, mark='lig')
            if ShowPicture:
                showPicture(tmp_dict)
            # images.append(tmp_dict)

    if Aug_crop:
        for i in range(len(all_data)):
            img, bboxes = text_aug.cropText(readImage(all_data[i]['filepath']), all_data[i]['bboxes'])
            tmp_dict = generateAugData(img, bboxes, all_data[i]['filepath'], iteration, mark='crp')
            if ShowPicture:
                showPicture(tmp_dict)
            # images.append(tmp_dict)

    if Aug_shift:
        for i in range(len(all_data)):
            img, bboxes = text_aug.shiftText(readImage(all_data[i]['filepath']), all_data[i]['bboxes'])
            tmp_dict = generateAugData(img, bboxes, all_data[i]['filepath'], iteration, mark='shf')
            if ShowPicture:
                showPicture(tmp_dict)



def textAugment_random(all_data, Aug_width=False, Aug_height=False, Aug_resize=False,
                       Aug_rotate=False, Aug_rotate_height=False, Aug_rotate_width=False,
                       Aug_add_noise=False, Aug_change_light=False, Aug_crop=False,
                       Aug_shift=False, iteration=None):
    text_aug = ta.TextAugment()
    img = readImage(all_data['filepath'])
    bboxes = all_data['bboxes']

    if Aug_width:
        img, bboxes = text_aug.resizeTextWidth(img, bboxes)

    if Aug_height:
        img, bboxes = text_aug.resizeTextHeight(img, bboxes)

    if Aug_resize:
        img, bboxes = text_aug.resizeText(img, bboxes)

    if Aug_rotate:
        img, bboxes = text_aug.rotateText(img, bboxes)

    if Aug_rotate_height:
        img, bboxes = text_aug.resizeTextHeight(img, bboxes)

    if Aug_rotate_width:
        img, bboxes = text_aug.resizeTextWidth(img, bboxes)

    if Aug_add_noise:
        img = text_aug.addNoise(img)

    if Aug_change_light:
        img = text_aug.changeLight(img)

    if Aug_crop:
        img, bboxes = text_aug.cropText(img, bboxes)

    if Aug_shift:
        img, bboxes = text_aug.shiftText(img, bboxes)

    tmp_dict = generateAugData(img, bboxes, all_data['filepath'], iteration, mark='mix')
    if ShowPicture:
        showPicture(tmp_dict)


def textAugment_manual(all_data, Aug_width=False, Aug_height=False, Aug_resize=False,
                       Aug_rotate=False, Aug_rotate_height=False, Aug_rotate_width=False,
                       Aug_add_noise=False, Aug_change_light=False, Aug_crop=False,
                       Aug_shift=False, Aug_color=False, Aug_spot=False, iteration=None):

    text_aug = ta.TextAugment()
    img = readImage(all_data['filepath'])
    bboxes = all_data['bboxes']

    if Aug_width:
        img, bboxes = text_aug.resizeTextWidth(img, bboxes)

    if Aug_height:
        img, bboxes = text_aug.resizeTextHeight(img, bboxes)

    if Aug_resize:
        img, bboxes = text_aug.resizeText(img, bboxes)

    if Aug_rotate:
        img, bboxes = text_aug.rotateText(img, bboxes)

    if Aug_rotate_height:
        img, bboxes = text_aug.resizeTextHeight(img, bboxes)

    if Aug_rotate_width:
        img, bboxes = text_aug.resizeTextWidth(img, bboxes)

    if Aug_add_noise:
        img = text_aug.addNoise(img)

    if Aug_change_light:
        img = text_aug.changeLight(img)

    if Aug_crop:
        img, bboxes = text_aug.cropText(img, bboxes)

    if Aug_shift:
        img, bboxes = text_aug.shiftText(img, bboxes)

    if Aug_color:
        img, bboxes = text_aug.randomColor(img, bboxes)

    if Aug_spot:
        img, bboxes = text_aug.randomSpot(img, bboxes)

    tmp_dict = generateAugData(img, bboxes, all_data['filepath'], iteration, mark='mix')
    if ShowPicture:
        showPicture(tmp_dict)


def multi_pro(all_images, end_index):

    print("end image is:" + end_index.__str__())
    flag = []
    cnt = 1
    flag.append([False] * 5)

    for ii in range(5):
        tmp = [False] * 5
        tmp[ii] = True
        flag.append(tmp)
        cnt += 1

    # rs = [0, 1]
    # rt = [2, 3, 4]
    # nl = [5, 6]
    # cs = [7, 8]
    # cl = [9]
    # sp = [10]
    # sw = [rs, rt, nl, cs, cl, sp]

    # rs = [0, 1]
    rt = [0]
    nl = [1]
    cs = [2]
    cl = [3]
    sp = [4]
    sw = [rt, nl, cs, cl]

    for ii in range(3):
        for jj in range(ii + 1, 4):
            a = sw[ii]
            b = sw[jj]
            for aa in a:
                for bb in b:
                    tmp = [False] * 5
                    tmp[bb] = True
                    tmp[aa] = True
                    flag.append(tmp)
                    cnt += 1

    if test:
        flag = [[False, False, False, False, False]]
    print(flag)
    print(cnt)

    for ii in range(cnt):

        print('The loop is ', ii)
        Aug_width = False
        Aug_height = False
        Aug_resize = False

        Aug_rotate = flag[ii][0]
        Aug_rotate_height = False
        Aug_rotate_width = False

        Aug_add_noise = False
        Aug_change_light = flag[ii][1]

        Aug_crop = flag[ii][2]
        Aug_shift = False

        Aug_color = flag[ii][3]

        Aug_spot = flag[ii][4]

        for j in tqdm(range(len(all_images))):
            print(all_images[j]['filepath'])
            textAugment_manual(copy.deepcopy(all_images[j]), Aug_width=Aug_width, Aug_height=Aug_height,
                               Aug_resize=Aug_resize, Aug_rotate=Aug_rotate,
                               Aug_rotate_height=Aug_rotate_height,
                               Aug_rotate_width=Aug_rotate_width, Aug_add_noise=Aug_add_noise,
                               Aug_change_light=Aug_change_light, Aug_crop=Aug_crop,
                               Aug_shift=Aug_shift, Aug_color=Aug_color, Aug_spot=Aug_spot, iteration=ii)


def main():
    # 注释文件
    ANNOTATIONS_FILE = '../sample/icdar2013_annotations_train_set.txt'
    # 返回样本的所有信息，各类别数目，以及类别映射
    all_images, classes_count, class_mapping = get_data(ANNOTATIONS_FILE, PATH_NOTICE=False)
    print(len(all_images))
    print(classes_count)
    print(class_mapping)

    if TextAugRandom:
        MultiNum = 28  # 并发线程数
        print("MultiProcess:" + MultiNum.__str__())
        # 新建指定数量的进程池用于对多进程进行管理
        pool = Pool(processes=MultiNum)

        for ii in range(AugLoop):
            print('The loop is ', ii)
            pool.apply_async(multi_pro, args=(all_images, ii))

        pool.close()
        pool.join()

    # input('STOP!!')

    if TrainTextAug:
        for ii in range(AugLoop):
            aug_img = copy.deepcopy(all_images)

            textAugment(copy.deepcopy(aug_img), Aug_width=True, iteration=ii)
            textAugment(copy.deepcopy(aug_img), Aug_height=True, iteration=ii)
            # textAugment(copy.deepcopy(aug_img), Aug_resize=True, iteration=ii)
            textAugment(copy.deepcopy(aug_img), Aug_rotate=True, iteration=ii)
            textAugment(copy.deepcopy(aug_img), Aug_rotate_height=True, iteration=ii)
            textAugment(copy.deepcopy(aug_img), Aug_rotate_width=True, iteration=ii)
            textAugment(copy.deepcopy(aug_img), Aug_change_light=True, iteration=ii)
            textAugment(copy.deepcopy(aug_img), Aug_crop=True, iteration=ii)
            textAugment(copy.deepcopy(aug_img), Aug_shift=True, iteration=ii)
            textAugment(copy.deepcopy(aug_img), Aug_add_noise=True, iteration=ii)

    if TrainTextAug_manual:

        MultiNum = 28  # 并发线程数
        print("MultiProcess:" + MultiNum.__str__())
        # 新建指定数量的进程池用于对多进程进行管理
        pool = Pool(processes=MultiNum)

        # 划分任务, tmp 代表每一个进程完成的任务数
        tmp = int(math.ceil(len(all_images) / MultiNum))
        # print(tmp)
        for i in range(MultiNum):
            print("MultiProcess:" + i.__str__())
            pool.apply_async(multi_pro, args=(all_images[i * tmp:i * tmp + tmp], i * tmp + tmp))

        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
