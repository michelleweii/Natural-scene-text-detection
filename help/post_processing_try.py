import numpy as np
import cv2


def processLocation(h, w, classes=None, locations=None, dsr_x=4, dsr_y=4):
    for i in range(classes.shape[0]):
        for j in range(classes.shape[1]):
            if classes[i, j, 0] < 0.5:
                classes[i, j, 0] = 255
            else:
                classes[i, j, 0] = 0
    img = np.array(classes, np.uint8)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # img = cv2.imread('test.jpg')
    # imgray = cv2.cvtColor(classes, cv2.COLOR_BGR2GRAY)  # 彩色转灰度
    # ret, thresh = cv2.threshold(img, 127, 255, 0)  # 进行二值化
    # print(thresh)
    img, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE,
                                                  cv2.CHAIN_APPROX_SIMPLE)  # 检索模式为树形cv2.RETR_TREE，
    # 轮廓存储模式为简单模式cv2.CHAIN_APPROX_SIMPLE，如果设置为 cv2.CHAIN_APPROX_NONE，所有的边界点都会被存储。
    # img = cv2.drawContour(img, contours, -1, (0, 255, 0), 3)  # 此时是将轮廓绘制到了原始图像上
    # 第三个参数是轮廓的索引（在绘制独立轮廓是很有用，当设置为 -1 时绘制所有轮廓）。接下来的参数是轮廓的颜色和厚度等
    for i in range(0, len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (153, 153, 0), 3)

    img2 = cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)
    cv2.imshow('img', img2)  # 显示原始图像
    cv2.waitKey()  # 窗口等待按键，无此代码，窗口闪一下就消失

    # ret, binary = cv2.threshold(classes, 127, 255, cv2.THRESH_BINARY)
    # binary = classes.reshape((classes.shape[0], classes.shape[1], 1))

    # _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # gray = classes.reshape((classes.shape[0], classes.shape[1], 1))
    # # gray = cv2.cvtColor(classes, cv2.COLOR_BGR2GRAY)
    # mser = cv2.MSER_create()
    # contours, regions = mser.detectRegions(gray)
    # hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    # cv2.polylines(classes, hulls, 1, (0, 255, 0))
    cv2.imshow('img', classes)

    # cv2.imshow('pic', classes)
    cv2.waitKey(0)


def main():
    classes = np.ones((ResizeH // 4, ResizeW // 4, 1))
    for i in range(50, 80):
        for j in range(60, 100):
            classes[i, j, 0] = 0
    for i in range(60, 90):
        for j in range(65, 120):
            classes[i, j, 0] = 0
    processLocation(ResizeH, ResizeW, classes=classes, dsr_x=4, dsr_y=4)


if __name__ == '__main__':
    ResizeH = 608
    ResizeW = 800
    main()
