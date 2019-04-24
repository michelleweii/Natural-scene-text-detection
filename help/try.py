#!/usr/bin/evn python
# coding:utf-8
import json
from tqdm import tqdm
import os
import cv2
import numpy as np
from help import show_picture as sp

train_json = '../dataset/ReCTS/gt_unicode/train_ReCTS_019668.json'
# val_json = '../dataset/ReCTS/annotations/instances_val2017.json'
train_Jpeg = '../dataset/ReCTS/img/train_ReCTS_019668.jpg'

json_data = json.load(open(train_json, 'r'))

for i in range(len(json_data['lines'])):
    x1 = json_data['lines'][i]['points'][0]
    y1 = json_data['lines'][i]['points'][1]
    x2 = json_data['lines'][i]['points'][2]
    y2 = json_data['lines'][i]['points'][3]
    x3 = json_data['lines'][i]['points'][4]
    y3 = json_data['lines'][i]['points'][5]
    x4 = json_data['lines'][i]['points'][6]
    y4 = json_data['lines'][i]['points'][7]
    print(json_data['lines'][i]['points'])

    sp.show_polygon(train_Jpeg, json_data['lines'][i]['points'])
