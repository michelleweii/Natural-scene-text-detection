import numpy as np
import cv2
import colorsys
# import copy
from help import NMS


def _create_unique_color_float(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def _create_unique_color_uchar(tag, hue_step=0.41):
    r, g, b = _create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def recoverBBox(x, y, w, h, dsr_x, dsr_y):
    x = x * dsr_x
    y = y * dsr_y
    w = w * dsr_x
    h = h * dsr_y
    return x, y, w, h


def drawBBoxes(img, bboxes, dsr_x, dsr_y, color_index=0, threshold=1):
    for i in range(len(bboxes)):
        if bboxes[i][2] > threshold and bboxes[i][3] > threshold:
            x, y, w, h = recoverBBox(bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3], dsr_x, dsr_y)
            color = _create_unique_color_uchar(color_index)
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, 2)
    return img


def getBoundingRect(contours, threshold=1):
    bboxes = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if h > threshold and w > threshold:
            bboxes.append([x, y, w, h])
    return bboxes


def drawBoundingRect(img, contours, dsr_x, dsr_y, color_index=0, threshold=1):
    for i in range(len(contours)):
        # print(cv2.minAreaRect(contours[i]))
        x, y, w, h = cv2.boundingRect(contours[i])
        if h > threshold and w > threshold:
            x, y, w, h = recoverBBox(x, y, w, h, dsr_x, dsr_y)
            color = _create_unique_color_uchar(color_index)
            cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), color, 2)
    return img


def findContours(img):
    _, contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def includeBoxes(big_box, small_box):
    big_x1 = big_box[0]
    big_x2 = big_box[0] + big_box[2] - 1
    big_y1 = big_box[1]
    big_y2 = big_box[1] + big_box[3] - 1
    inc_box = []
    # rem_box = copy.deepcopy(small_box)
    for i in range(len(small_box)):
        center_x = small_box[i][0] + (small_box[i][2] - 1) / 2
        center_y = small_box[i][1] + (small_box[i][3] - 1) / 2
        if center_x > big_x1 and center_x < big_x2 and center_y > big_y1 and center_y < big_y2:
            inc_box.append(small_box[i])
            # rem_box.remove(small_box[i])
    return inc_box  # , rem_box


def filterBBox(bboxes):
    for i in range(len(bboxes)):
        bboxes[i].append(bboxes[i][2] * bboxes[i][3])
    return NMS.IoM_suppression_fast(bboxes)


def isLeftRightTouched(bbox1, bbox2, offset=1):
    condition1 = bbox1[0] + bbox1[2] + offset > bbox2[0]
    condition2 = bbox1[0] < bbox2[0]
    condition3 = bbox1[0] + bbox1[2] - 1 < bbox2[0] + bbox2[2] - 1
    condition4 = bbox1[1] <= bbox2[1] + bbox2[3] - 1
    condition5 = bbox1[1] + bbox1[3] - 1 >= bbox2[1]
    if condition1 and condition2 and condition3 and condition4 and condition5:
        return True
    else:
        return False


def isUpDownTouched(bbox1, bbox2, offset=1):
    condition1 = bbox1[1] + bbox1[3] + offset > bbox2[1]
    condition2 = bbox1[1] < bbox2[1]
    condition3 = bbox1[1] + bbox1[3] - 1 < bbox2[1] + bbox2[3] - 1
    condition4 = bbox1[0] <= bbox2[0] + bbox2[2] - 1
    condition5 = bbox1[0] + bbox1[2] - 1 >= bbox2[0]
    if condition1 and condition2 and condition3 and condition4 and condition5:
        return True
    else:
        return False


def isMatchPairLR(j, bboxes1, bboxes2):
    index1 = []
    area = 0
    for k in range(len(bboxes2)):
        if isLeftRightTouched(bboxes1[j], bboxes2[k]):
            if bboxes2[k][2] * bboxes2[k][3] > area:
                area = bboxes2[k][2] * bboxes2[k][3]
                index1 = [j, k]
    if index1:
        index2 = []
        area = 0
        for l in range(len(bboxes1)):
            if isLeftRightTouched(bboxes1[l], bboxes2[index1[1]]):
                if bboxes1[l][2] * bboxes1[l][3] > area:
                    area = bboxes1[l][2] * bboxes1[l][3]
                    index2 = [l, index1[1]]
    else:
        return False, index1

    if index1 == index2:
        return True, index1
    else:
        return False, index1


def isMatchPairRL(j, bboxes1, bboxes2):
    index1 = []
    area = 0
    for k in range(len(bboxes2)):
        if isLeftRightTouched(bboxes2[k], bboxes1[j]):
            if bboxes2[k][2] * bboxes2[k][3] > area:
                area = bboxes2[k][2] * bboxes2[k][3]
                index1 = [j, k]
    if index1:
        index2 = []
        area = 0
        for l in range(len(bboxes1)):
            if isLeftRightTouched(bboxes2[index1[1]], bboxes1[l]):
                if bboxes1[l][2] * bboxes1[l][3] > area:
                    area = bboxes1[l][2] * bboxes1[l][3]
                    index2 = [l, index1[1]]
    else:
        return False, index1

    if index1 == index2:
        return True, index1
    else:
        return False, index1


def isMatchPairUD(j, bboxes1, bboxes2):
    index1 = []
    area = 0
    for k in range(len(bboxes2)):
        if isUpDownTouched(bboxes1[j], bboxes2[k]):
            if bboxes2[k][2] * bboxes2[k][3] > area:
                area = bboxes2[k][2] * bboxes2[k][3]
                index1 = [j, k]
    if index1:
        index2 = []
        area = 0
        for l in range(len(bboxes1)):
            if isUpDownTouched(bboxes1[l], bboxes2[index1[1]]):
                if bboxes1[l][2] * bboxes1[l][3] > area:
                    area = bboxes1[l][2] * bboxes1[l][3]
                    index2 = [l, index1[1]]
    else:
        return False, index1

    if index1 == index2:
        return True, index1
    else:
        return False, index1


def isMatchPairDU(j, bboxes1, bboxes2):
    index1 = []
    area = 0
    for k in range(len(bboxes2)):
        if isUpDownTouched(bboxes2[k], bboxes1[j]):
            if bboxes2[k][2] * bboxes2[k][3] > area:
                area = bboxes2[k][2] * bboxes2[k][3]
                index1 = [j, k]
    if index1:
        index2 = []
        area = 0
        for l in range(len(bboxes1)):
            if isUpDownTouched(bboxes2[index1[1]], bboxes1[l]):
                if bboxes1[l][2] * bboxes1[l][3] > area:
                    area = bboxes1[l][2] * bboxes1[l][3]
                    index2 = [l, index1[1]]
    else:
        return False, index1

    if index1 == index2:
        return True, index1
    else:
        return False, index1


def segmentConnectedText(bbox_all, bbox_up_left, bbox_up_right, bbox_down_left, bbox_down_right):
    seg_checked_bbox = []
    for i in range(len(bbox_all)):
        up_left = includeBoxes(bbox_all[i], bbox_up_left)
        if len(up_left) <= 1:
            seg_checked_bbox.append(bbox_all[i])
            continue
        else:
            up_right = includeBoxes(bbox_all[i], bbox_up_right)
            if len(up_right) <= 1:
                seg_checked_bbox.append(bbox_all[i])
                continue
            else:
                down_right = includeBoxes(bbox_all[i], bbox_down_right)
                if len(down_right) <= 1:
                    seg_checked_bbox.append(bbox_all[i])
                    continue
                else:
                    down_left = includeBoxes(bbox_all[i], bbox_down_left)
                    if len(down_left) <= 1:
                        seg_checked_bbox.append(bbox_all[i])
                        continue
                    else:
                        final_index = []
                        for j in range(len(up_left)):
                            match_index = []
                            flag1, index1 = isMatchPairLR(j, up_left, up_right)
                            if flag1:
                                match_index.extend(index1)
                                flag2, index2 = isMatchPairUD(index1[1], up_right, down_right)
                                if flag2:
                                    match_index.append(index2[1])
                                    flag3, index3 = isMatchPairRL(index2[1], down_right, down_left)
                                    if flag3:
                                        match_index.append(index3[1])
                                        flag4, index4 = isMatchPairDU(index3[1], down_left, up_left)
                                        if flag4:
                                            # Got a group
                                            final_index.append(match_index)

                        if len(final_index) > 1:
                            for l in range(len(final_index)):
                                x1 = min(up_left[final_index[l][0]][0], up_right[final_index[l][1]][0],
                                         down_right[final_index[l][2]][0], down_left[final_index[l][3]][0])
                                y1 = min(up_left[final_index[l][0]][1], up_right[final_index[l][1]][1],
                                         down_right[final_index[l][2]][1], down_left[final_index[l][3]][1])
                                x2 = max(up_left[final_index[l][0]][0] + up_left[final_index[l][0]][2] - 1,
                                         up_right[final_index[l][1]][0] + up_right[final_index[l][1]][2] - 1,
                                         down_right[final_index[l][2]][0] + down_right[final_index[l][2]][2] - 1,
                                         down_left[final_index[l][3]][0] + down_left[final_index[l][3]][2] - 1)
                                y2 = max(up_left[final_index[l][0]][1] + up_left[final_index[l][0]][3] - 1,
                                         up_right[final_index[l][1]][1] + up_right[final_index[l][1]][3] - 1,
                                         down_right[final_index[l][2]][1] + down_right[final_index[l][2]][3] - 1,
                                         down_left[final_index[l][3]][1] + down_left[final_index[l][3]][3] - 1)
                                seg_checked_bbox.append([x1, y1, x2 - x1 + 1, y2 - y1 + 1])
                                bbox_up_left.remove(up_left[final_index[l][0]])
                                bbox_up_right.remove(up_right[final_index[l][1]])
                                bbox_down_right.remove(down_right[final_index[l][2]])
                                bbox_down_left.remove(down_left[final_index[l][3]])
                        else:
                            seg_checked_bbox.append(bbox_all[i])
    return seg_checked_bbox


def processLocation(img, cls_image=None, locations=None, dsr_x=4, dsr_y=4, drawBBox=False, filterBox=True):
    for i in range(cls_image.shape[0]):
        for j in range(cls_image.shape[1]):
            if cls_image[i, j, 0] < 0.5:
                cls_image[i, j, 0] = 255
            else:
                cls_image[i, j, 0] = 0
            loc = locations[i, j, :].tolist()
            loc = loc.index(max(loc))
            locations[i, j, :] = 0
            locations[i, j, loc] = 255

    cls_image = np.array(cls_image, np.uint8)
    up_left = np.array(locations[:, :, 1], np.uint8)
    up_right = np.array(locations[:, :, 2], np.uint8)
    down_left = np.array(locations[:, :, 3], np.uint8)
    down_right = np.array(locations[:, :, 4], np.uint8)
    # img_color = cv2.cvtColor(cls_image, cv2.COLOR_GRAY2RGB)
    # img = cv2.imread('test.jpg')
    # imgray = cv2.cvtColor(classes, cv2.COLOR_BGR2GRAY)  # 彩色转灰度
    # ret, thresh = cv2.threshold(img, 127, 255, 0)  # 进行二值化
    # print(thresh)
    contours = findContours(cls_image)
    bbox_all = getBoundingRect(contours)
    contours_up_left = findContours(up_left)
    bbox_up_left = getBoundingRect(contours_up_left)
    contours_up_right = findContours(up_right)
    bbox_up_right = getBoundingRect(contours_up_right)
    contours_down_left = findContours(down_left)
    bbox_down_left = getBoundingRect(contours_down_left)
    contours_down_right = findContours(down_right)
    bbox_down_right = getBoundingRect(contours_down_right)

    bbox_all = segmentConnectedText(bbox_all, bbox_up_left, bbox_up_right, bbox_down_left, bbox_down_right)

    # 检索模式为树形cv2.RETR_TREE，
    # 轮廓存储模式为简单模式cv2.CHAIN_APPROX_SIMPLE，如果设置为 cv2.CHAIN_APPROX_NONE，所有的边界点都会被存储。
    # img = cv2.drawContour(img, contours, -1, (0, 255, 0), 3)  # 此时是将轮廓绘制到了原始图像上
    # 第三个参数是轮廓的索引（在绘制独立轮廓是很有用，当设置为 -1 时绘制所有轮廓）。接下来的参数是轮廓的颜色和厚度等

    # img = drawBoundingRect(img, contours, dsr_x, dsr_y, color_index=0)
    # img = drawBoundingRect(img, contours_up_left, dsr_x, dsr_y, color_index=1)
    # img = drawBoundingRect(img, contours_up_right, dsr_x, dsr_y, color_index=2)
    # img = drawBoundingRect(img, contours_down_left, dsr_x, dsr_y, color_index=3)
    # img = drawBoundingRect(img, contours_down_right, dsr_x, dsr_y, color_index=4)
    if filterBox:
        bbox_all = filterBBox(bbox_all)
    if drawBBox:
        img = drawBBoxes(img, bbox_all, dsr_x, dsr_y, color_index=0)
        return img
    else:
        for i in range(len(bbox_all)):
            bbox_all[i][0] *= dsr_x
            bbox_all[i][1] *= dsr_y
            bbox_all[i][2] *= dsr_x
            bbox_all[i][3] *= dsr_y
        return bbox_all

    # img2 = cv2.drawContours(img_color, contours, -1, (0, 255, 0), 1)
    # cv2.imshow('img', img)  # 显示原始图像
    # cv2.waitKey()  # 窗口等待按键，无此代码，窗口闪一下就消失

    # ret, binary = cv2.threshold(classes, 127, 255, cv2.THRESH_BINARY)
    # binary = classes.reshape((classes.shape[0], classes.shape[1], 1))

    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # gray = classes.reshape((classes.shape[0], classes.shape[1], 1))
    # # gray = cv2.cvtColor(classes, cv2.COLOR_BGR2GRAY)
    # mser = cv2.MSER_create()
    # contours, regions = mser.detectRegions(gray)
    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # cv2.polylines(classes, hulls, 1, (0, 255, 0))
    # cv2.imshow('img', cls_image)

    # cv2.imshow('pic', classes)
    # cv2.waitKey(0)
#
#
# def main():
#     classes = np.ones((ResizeH // 4, ResizeW // 4, 1))
#     for i in range(50, 80):
#         for j in range(60, 100):
#             classes[i, j, 0] = 0
#     for i in range(60, 90):
#         for j in range(65, 120):
#             classes[i, j, 0] = 0
#     processLocation(cls_image=classes)
#
#
# if __name__ == '__main__':
#     ResizeH = 608
#     ResizeW = 800
#     main()
