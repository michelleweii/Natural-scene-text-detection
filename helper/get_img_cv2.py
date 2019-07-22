import numpy as np
import cv2


# 读取图片函数
def get_im_cv2(paths, img_cols, img_rows, color_type=3, normalize=True):
    '''
    参数：
        paths：要读取的图片路径列表
        img_rows:图片行
        img_cols:图片列
        color_type:图片颜色通道
    返回:
        imgs: 图片数组
    '''
    # Load as grayscale if color_type=1
    imgs = []
    for path in paths:
        if color_type == 1:
            img = cv2.imread(path, 0)
        elif color_type == 3:
            img = cv2.imread(path)
        # Reduce size
        resize_img = cv2.resize(img, (img_cols, img_rows))
        if normalize:
            resize_img = resize_img.astype('float32')
            resize_img /= 127.5
            resize_img -= 1.

        imgs.append(resize_img)

    return np.array(imgs).reshape(len(paths), img_rows, img_cols, color_type)
