import os
import cv2
import numpy as np
import colorsys


def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(0, img.shape[0], 16):
        cv2.line(img, (0, i), (img.shape[1] - 1, i), 127, 1)
    cv2.line(img, (0, img.shape[0] - 1), (img.shape[1] - 1, img.shape[0] - 1), 27, 1)

    for i in range(0, img.shape[1], 16):
        cv2.line(img, (i, 0), (i, img.shape[0] - 1), 127, 1)
    cv2.line(img, (img.shape[1] - 1, 0), (img.shape[1] - 1, img.shape[0] - 1), 27, 1)

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
    cv2.namedWindow('pic', 1)  # 1表示原图
    # cv2.moveWindow('pic', 0, 0)
    # cv2.resizeWindow('pic')  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    os.remove('./1.jpg')


def show_polygon(file_path, coordinates):
    x1 = coordinates[0]
    y1 = coordinates[1]
    x2 = coordinates[2]
    y2 = coordinates[3]
    x3 = coordinates[4]
    y3 = coordinates[5]
    x4 = coordinates[6]
    y4 = coordinates[7]
    img = cv2.imread(file_path)
    a = np.array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], dtype=np.int32)
    cv2.polylines(img, a, 1, 255, thickness=2)
    cv2.putText(img, '1', (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(img, '2', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(img, '3', (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv2.putText(img, '4', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    cv2.circle(img, (x1, y1), 5, color=(255, 255, 0))
    cv2.circle(img, (x2, y2), 5, color=(0, 0, 0))
    cv2.circle(img, (x3, y3), 5, color=(255, 0, 255))
    cv2.circle(img, (x4, y4), 5, color=(0, 255, 255))
    cv2.imshow('pic', img)
    cv2.waitKey(0)


def show_bbox_in_one_image(x, y):
    # y=y0[0]
    for i in range(len(y)):
        img = x[i]
        img = np.array(img)
        # img = 255 - img * 255
        # img += 1
        # img *= 127.5
        for j in range(len(y[i])):
            # if y[i][j]['ignore'] == 1:
            #     # print("Ignore")
            line_color = (0, 255, 0)
            # else:
            #     line_color = (255, 0, 0)
            x1 = int(y[i][j]['x1'])
            y1 = int(y[i][j]['y1'])
            x2 = int(y[i][j]['x2'])
            y2 = int(y[i][j]['y2'])
            # x3 = int(y[i][j]['x3'])
            # y3 = int(y[i][j]['y3'])
            # x4 = int(y[i][j]['x4'])
            # y4 = int(y[i][j]['y4'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # a = np.array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], dtype=np.int32)
            # cv2.polylines(img, a, 1, line_color, thickness=2)
            # cv2.putText(img, '1', (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # cv2.putText(img, '2', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # cv2.putText(img, '3', (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # cv2.putText(img, '4', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # cv2.circle(img, (x1, y1), 5, color=(255, 255, 0))
            # cv2.circle(img, (x2, y2), 5, color=(0, 0, 0))
            # cv2.circle(img, (x3, y3), 5, color=(255, 0, 255))
            # cv2.circle(img, (x4, y4), 5, color=(0, 255, 255))
        cv2.imshow('pic', img)
        cv2.waitKey(0)


def show_polygon_in_one_image(x, y):
    # y=y0[0]
    for i in range(len(y)):
        img = x[i]
        img = np.array(img)
        # img = 255 - img * 255
        # img += 1
        # img *= 127.5
        for j in range(len(y[i])):
            if y[i][j]['ignore'] == 1:
                # print("Ignore")
                line_color = (0, 255, 0)
            else:
                line_color = (255, 0, 0)
            x1 = int(y[i][j]['x1'])
            y1 = int(y[i][j]['y1'])
            x2 = int(y[i][j]['x2'])
            y2 = int(y[i][j]['y2'])
            x3 = int(y[i][j]['x3'])
            y3 = int(y[i][j]['y3'])
            x4 = int(y[i][j]['x4'])
            y4 = int(y[i][j]['y4'])
            a = np.array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], dtype=np.int32)
            cv2.polylines(img, a, 1, line_color, thickness=2)
            # cv2.putText(img, '1', (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # cv2.putText(img, '2', (x2, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # cv2.putText(img, '3', (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # cv2.putText(img, '4', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            # cv2.circle(img, (x1, y1), 5, color=(255, 255, 0))
            # cv2.circle(img, (x2, y2), 5, color=(0, 0, 0))
            # cv2.circle(img, (x3, y3), 5, color=(255, 0, 255))
            # cv2.circle(img, (x4, y4), 5, color=(0, 255, 255))
        cv2.imshow('pic', img)
        cv2.waitKey(0)


def _create_unique_color_float(tag, hue_step=0.41):
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def _create_unique_color_uchar(tag, hue_step=0.41):
    r, g, b = _create_unique_color_float(tag, hue_step)
    return int(255 * r), int(255 * g), int(255 * b)


def draw_polygon_in_one_image(img, bboxes):
    for c, boxes in bboxes.items():
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            x3 = int(box[4])
            y3 = int(box[5])
            x4 = int(box[6])
            y4 = int(box[7])
            index = int(box[-1])
            unique_color = _create_unique_color_uchar(index)
            a = np.array([[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]], dtype=np.int32)
            cv2.polylines(img, a, 1, unique_color, thickness=1)
    return img


def draw_bbox_in_one_image(img, bboxes):
    for c, boxes in bboxes.items():
        for box in boxes:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            index = int(box[-1])
            unique_color = _create_unique_color_uchar(index)
            cv2.rectangle(img, (x1, y1), (x2, y2), unique_color, 2)
    return img


def draw_boxes_and_label_cv2(img, classes, CLA):
    w = classes.shape[1]
    h = classes.shape[0]
    img_width = img.shape[1]
    img_height = img.shape[0]
    grid_x = img_width // w
    grid_y = img_height // h

    for i in range(h):
        for j in range(w):
            tmp_cls = classes[i, j, :].tolist()
            cls = tmp_cls.index(max(tmp_cls))
            score = np.max(classes[i][j])
            if cls != CLA:
                # print(cls)
                if score > 0.7:
                    unique_color = _create_unique_color_uchar(cls)
                else:
                    unique_color = _create_unique_color_uchar(cls + 2)
                grid_left = j * grid_x
                grid_top = i * grid_y
                grid_right = grid_left + grid_x - 1
                grid_bottom = grid_top + grid_y - 1
                cv2.rectangle(img, (grid_left, grid_top), (grid_right, grid_bottom), unique_color, 1)
    return img


def draw_grid_binary(img, classes, CLA):
    w = classes.shape[1]
    h = classes.shape[0]
    img_width = img.shape[1]
    img_height = img.shape[0]
    grid_x = img_width // w
    grid_y = img_height // h

    for i in range(h):
        for j in range(w):
            score = 1 - classes[i][j][0]
            if score <= 0.5:
                cls = CLA
            else:
                cls = 0
            if cls != CLA:
                # print(cls)
                if score > 0.7:
                    unique_color = _create_unique_color_uchar(cls)
                else:
                    unique_color = _create_unique_color_uchar(cls + 2)
                grid_left = j * grid_x
                grid_top = i * grid_y
                grid_right = grid_left + grid_x - 1
                grid_bottom = grid_top + grid_y - 1
                cv2.rectangle(img, (grid_left, grid_top), (grid_right, grid_bottom), unique_color, 1)
    return img


def draw_grid_location(img, classes, location, CLA):
    w = classes.shape[1]
    h = classes.shape[0]
    img_width = img.shape[1]
    img_height = img.shape[0]
    grid_x = img_width // w
    grid_y = img_height // h

    for i in range(h):
        for j in range(w):
            loc = location[i, j, :].tolist()
            loc = loc.index(max(loc))
            score = 1 - classes[i][j][0]
            if score <= 0.5:
                cls = CLA
            else:
                cls = 0
            if cls != CLA:
                # print(cls)
                if score > 0.7:
                    thickness = 1
                else:
                    thickness = -1
                if loc == 1:
                    unique_color = _create_unique_color_uchar(cls + 3)
                elif loc == 2:
                    unique_color = _create_unique_color_uchar(cls + 4)
                elif loc == 3:
                    unique_color = _create_unique_color_uchar(cls + 5)
                elif loc == 4:
                    unique_color = _create_unique_color_uchar(cls + 6)
                else:
                    unique_color = _create_unique_color_uchar(cls + 7)
                grid_left = j * grid_x
                grid_top = i * grid_y
                grid_right = grid_left + grid_x - 1
                grid_bottom = grid_top + grid_y - 1
                cv2.rectangle(img, (grid_left, grid_top), (grid_right, grid_bottom), unique_color, thickness=thickness)
    return img
