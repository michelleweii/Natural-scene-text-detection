import numpy as np
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import cv2


def calculate_polygon_IoM(grid, polygon_bbox):
    # 四边形四个点坐标的一维数组表示，[x,y,x,y....]
    a = np.array(grid).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    b = np.array(polygon_bbox).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    try:
        inter_area = poly1.intersection(poly2).area  # 相交面积
        # print(inter_area)
        # union_area = poly1.area + poly2.area - inter_area
        # union_area = MultiPoint(union_poly).convex_hull.area
        min_area = min(poly1.area, poly2.area)
        # print(union_area)
        if min_area == 0:
            iom = 0
        else:
            iom = float(inter_area) / min_area
    except shapely.geos.TopologicalError:
        print('shapely.geos.TopologicalError occured, iou set to 0')
        iom = 0
    return iom


def calculate_polygon_IoU(grid, polygon_bbox):
    # 四边形四个点坐标的一维数组表示，[x,y,x,y....]
    a = np.array(grid).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    b = np.array(polygon_bbox).reshape(4, 2)
    poly2 = Polygon(b).convex_hull
    # union_poly = np.concatenate((a, b))  # 合并两个box坐标，变为8*2
    try:
        inter_area = poly1.intersection(poly2).area  # 相交面积
        # print(inter_area)
        union_area = poly1.area + poly2.area - inter_area
        # union_area = MultiPoint(union_poly).convex_hull.area
        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / union_area
    except shapely.geos.TopologicalError:
        print('shapely.geos.TopologicalError occured, iou set to 0')
        iou = 0
    return iou


def calculate_polygon_area(polygon_bbox):
    a = np.array(polygon_bbox).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(a).convex_hull
    return poly1.area


def calculate_polygon_IoM_slow(grid, polygon_bbox, h, w):
    a = np.array(grid).reshape(4, 2)  # 四边形二维坐标表示
    b = np.array(polygon_bbox).reshape(4, 2)  # 四边形二维坐标表示
    img1 = np.zeros((h, w), np.uint8)
    img2 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(img1, [a], 1)
    cv2.fillPoly(img2, [b], 1)
    return (img1 * img2).sum() / min(img1.sum(), img2.sum())