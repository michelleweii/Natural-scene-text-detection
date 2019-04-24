import pickle
import numpy as np
from help import handle_model as hm
from help import sample_resize as sr
from help import min_max_scaler as mms
from help import show_picture as sp
from help import metrics as me
from keras.models import load_model
import time
from help import NMS
import cv2
import os, tarfile
import shutil
from keras import backend as K
from tqdm import tqdm
from network import resnet
from help import post_processing as pp
import copy


# input the saved samples
def loadSample(samples):
    with open(samples, 'rb') as f:
        all_images = pickle.load(f)
        class_mapping = pickle.load(f)
        classes_count = pickle.load(f)
    return all_images, class_mapping, classes_count


# (x,y,w,h)->(x1,y1,x2,y2,score,cls)
# handle the whole bounding boxes
def handleBBOX(bbox, score):
    bbox_score_cls = []
    for num in range(bbox.shape[0]):  # number
        tmp_list = []
        for i in range(bbox.shape[1]):  # 18
            for j in range(bbox.shape[2]):  # 36
                # tmp_cls = score[num, i, j, :].tolist()
                # tmp_cls = tmp_cls.index(max(tmp_cls))
                text_score = 1 - score[num][i][j][0]
                if text_score <= 0.5:
                    tmp_cls = CLASS
                else:
                    tmp_cls = 0
                if tmp_cls != CLASS:  # and text_score > 0.99:
                    x1 = bbox[num][i][j][0] + j * DSR_x
                    y1 = bbox[num][i][j][1] + i * DSR_y
                    x2 = x1 + bbox[num][i][j][2] - 1
                    y2 = y1 + bbox[num][i][j][3] - 1
                    if x1 <= x2 and y1 <= y2:
                        tmp_list.append([x1, y1, x2, y2,
                                         text_score, tmp_cls])
        bbox_score_cls.append(tmp_list)
    # print(bbox_score_cls[0])
    return bbox_score_cls


# fast test, draw predicted bounding box, and save results to files
def testFast(test_img, md, name_model):
    # (height, width, channel) = test_img[0]['pixel'].shape
    # test_x = np.zeros((len(test_img), height, widt h, channel))
    predict_cls = np.zeros((len(test_img), ResizeH // DSR_y, ResizeW // DSR_x, 1))
    predict_loc = np.zeros((len(test_img), ResizeH // DSR_y, ResizeW // DSR_x, 5))

    # scale_file = open(scale_path, 'rb')
    # label_min = pickle.load(scale_file)
    # label_max = pickle.load(scale_file)
    # scale_file.close()

    X_path, Y, size = getInformation(test_img)
    x = sr.sampleResize(x_path=X_path, short_side=ShortSide, long_side=LongSide,
                        resizeh=ResizeH, resizew=ResizeW, resizeX=True)

    bbox = []
    start = time.time()
    for i in range(len(test_img)):
        # predicting...
        img = x[i]
        test_x = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        predict_cls[i], predict_loc[i] = md.predict(test_x)
        box = pp.processLocation(img, copy.deepcopy(predict_cls[i]), copy.deepcopy(predict_loc[i]),
                                 DSR_x, DSR_y, filterBox=FilterBBox)
        bbox.append(box)

    # inverse data standardizing
    # predict_bbox = mms.vtMinMaxScaler_inverse(predict_bbox, label_min, label_max)  # 有点疑问，是否要reshape
    # handle bounding box
    # 明天从这里写下去
    # bbox = handleBBOX(predict_loc, predict_cls)
    # current_model = ModelPath + name_model + '.hdf5'
    # NMS
    # for i in range(len(bbox)):
    #     bbox[i] = NMS.non_max_suppression_fast(bbox[i], overlap_thresh=0.5,
    #                                            max_boxes=20, current_model=name_model)
    cost_time = time.time() - start
    print('The cost time for an image is', str(cost_time / len(test_img)) + '.')

    # if len(bbox) != len(test_img):
    #     print('The lengths of bboxes and images are inconsistent.')
    #     os._exit(0)

    # 画小切块的类别
    if GridImage:
        ResultImage2_md = ResultImage + name_model + '_grid/'
        if not os.path.exists(ResultImage2_md):
            os.makedirs(ResultImage2_md)
        for i in tqdm(range(x.shape[0])):
            img_path = test_img[i]['filepath']
            base_name = str(os.path.basename(img_path).split('.')[0])
            img2 = x[i] * 255
            img2 = sp.draw_grid_location(img2, predict_cls[i], predict_loc[i], CLASS)
            result_image2_path = ResultImage2_md + base_name + '.png'
            cv2.imwrite(result_image2_path, img2)

    if LocationImage:
        ResultImage3_md = ResultImage + name_model + '_location/'
        if not os.path.exists(ResultImage3_md):
            os.makedirs(ResultImage3_md)
        for i in tqdm(range(x.shape[0])):
            img_path = test_img[i]['filepath']
            base_name = str(os.path.basename(img_path).split('.')[0])
            img3 = x[i] * 255
            img3 = pp.processLocation(img3, predict_cls[i], predict_loc[i], DSR_x, DSR_y, drawBBox=True,
                                      filterBox=FilterBBox)
            result_image3_path = ResultImage3_md + base_name + '.png'
            cv2.imwrite(result_image3_path, img3)

    # if ResizedImage_bbox:
    #     ResultImage3_md = ResultImage + name_model + '_bbox/'
    #     if not os.path.exists(ResultImage3_md):
    #         os.makedirs(ResultImage3_md)
    #     for i in tqdm(range(x.shape[0])):
    #         img_path = test_img[i]['filepath']
    #         base_name = str(os.path.basename(img_path).split('.')[0])
    #         img3 = x[i] * 255
    #         bboxes = {}
    #         for b in range(len(bbox[i])):
    #             class_name = int(bbox[i][b][5])
    #             class_prob = round(bbox[i][b][4], 2)
    #             x1 = round(bbox[i][b][0], 2)
    #             y1 = round(bbox[i][b][1], 2)
    #             x2 = round(bbox[i][b][2], 2)
    #             y2 = round(bbox[i][b][3], 2)
    #             if class_name not in bboxes:
    #                 bboxes[class_name] = []
    #             bboxes[class_name].append([x1, y1, x2, y2, class_prob, class_name])
    #
    #         img3 = sp.draw_bbox_in_one_image(img3, bboxes)
    #         # result_image_path = ResultImage_md + mark + base_name + '.png'
    #         result_image3_path = ResultImage3_md + base_name + '.png'
    #         cv2.imwrite(result_image3_path, img3)
    #
    for i in range(len(bbox)):
        bbox[i] = sr.bboxesTransform_inverse(bbox[i], test_img[i]['resizeinfo'], test_img[i]['padinfo'], isWH=True)
    return bbox


def writeResults(test_img, bbox, name_model):
    ResultPath_md = Result + name_model + '/'
    ResultImage_md = ResultImage + name_model + '/'

    if not os.path.exists(ResultPath_md):
        os.makedirs(ResultPath_md)
    if not os.path.exists(ResultImage_md):
        os.mkdir(ResultImage_md)

    # progress = ProgressBar()
    for i in tqdm(range(len(test_img))):
        img_path = test_img[i]['filepath']
        base_name = str(os.path.basename(img_path).split('.')[0])
        # result_txt = ResultPath_md + mark + base_name + '.txt'
        result_txt = ResultPath_md + base_name + '.txt'
        bboxes = {}
        # write the predicted results to files
        with open(result_txt, 'w') as f:
            for b in range(len(bbox[i])):
                class_name = 0
                class_prob = 1.0
                x1 = round(bbox[i][b][0], 2)
                y1 = round(bbox[i][b][1], 2)
                x2 = round(bbox[i][b][2], 2)
                y2 = round(bbox[i][b][3], 2)
                # x3 = round(bbox[i][b][4], 2)
                # y3 = round(bbox[i][b][5], 2)
                # x4 = round(bbox[i][b][6], 2)
                # y4 = round(bbox[i][b][7], 2)
                f.write(
                    '{} {} {} {} {} {} \n'.format(class_name, class_prob, x1, y1, x2, y2))

                if class_name not in bboxes:
                    bboxes[class_name] = []
                bboxes[class_name].append([x1, y1, x2, y2, class_prob, class_name])

        img1 = cv2.imread(img_path)
        img1 = sp.draw_bbox_in_one_image(img1, bboxes)
        # result_image_path = ResultImage_md + mark + base_name + '.png'
        result_image_path = ResultImage_md + base_name + '.png'
        cv2.imwrite(result_image_path, img1)
        # print(predict_cls[i].shape)
        # for t1 in range(predict_cls[i].shape[0]):
        #     for t2 in range(predict_cls[i].shape[1]):
        #         print(predict_cls[i][t1][t2])


def calculateResults(name_model):
    print('Calculating the final metrics...')
    eval = me.calculateF1score(ResultPath, name_model, iou=0.5)

    if FilterModel:
        good = 0.6
        normal = 0.5

        good_model = ModelPath + 'good_model/'
        normal_model = ModelPath + 'normal_model'
        src_model = ModelPath + name_model + '.hdf5'
        source_dir = ResultPath + '/predicted_' + name_model
        if eval >= good:
            if not os.path.exists(good_model):
                os.mkdir(good_model)
            shutil.move(src_model, good_model)
            # pack the results into tar file
            # os.chdir('./result/' + CNNModel + '/')
            if os.path.exists(ResultPath + '/predicted.tar'):
                os.remove(ResultPath + '/predicted.tar')
            # os.rename('./predicted', source_dir)
            os.chdir(source_dir)
            tar_file = './predicted.tar'
            with tarfile.open(tar_file, "w:") as tar:
                for file in tqdm(os.listdir('./')):
                    tar.add(file)
            shutil.move('./predicted.tar', '../')
        elif eval >= normal:
            if not os.path.exists(normal_model):
                os.mkdir(normal_model)
            shutil.move(src_model, normal_model)
        else:
            os.remove(src_model)


def getInformation(data):
    X_path = []
    y = []
    size = []
    for i in range(len(data)):
        X_path.append(data[i]['filepath'])
        y.append(data[i]['bboxes'])
        size.append([data[i]['height'], data[i]['width'], data[i]['channel']])
    return X_path, y, size


def bgIgnoredLoss(y_true, y_pred):
    c1 = K.cast(K.equal(y_true[:, :, 0], 0), 'float32')
    c2 = K.cast(K.equal(y_true[:, :, 1], 0), 'float32')
    c3 = K.cast(K.equal(y_true[:, :, 2], 0), 'float32')
    c4 = K.cast(K.equal(y_true[:, :, 3], 0), 'float32')
    check = 1 - c1 * c2 * c3 * c4
    normalizer = K.sum(check) + K.pow(10.0, -9)
    loss_x = K.sum(K.pow((y_true[:, :, 0] - y_pred[:, :, 0]) * check, 2))
    loss_y = K.sum(K.pow((y_true[:, :, 1] - y_pred[:, :, 1]) * check, 2))
    loss_w = K.sum(K.pow((K.sqrt(y_true[:, :, 2]) - K.sqrt(y_pred[:, :, 2])) * check, 2))
    loss_h = K.sum(K.pow((K.sqrt(y_true[:, :, 3]) - K.sqrt(y_pred[:, :, 3])) * check, 2))
    return 1 * (loss_x + loss_y + loss_w + loss_h) / normalizer


def main():
    print('All samples are from pickle files...')
    test_images, class_mapping, classes_count = loadSample(test_pickle)

    # if yes, input
    # if no, end
    FLAG, MN = hm.checkModel(ModelPath)
    if FLAG:
        print('Load model...')
        if backbone == 'resnet50' or backbone == 'resnet101':
            model = load_model(MN, custom_objects={'bgIgnoredLoss': bgIgnoredLoss,
                                                   'BatchNorm': resnet.BatchNorm})
        else:
            model = load_model(MN, custom_objects={'bgIgnoredLoss': bgIgnoredLoss})
        model_name = os.path.basename(MN).split('/')[-1]
        print('The model named by', model_name, 'is loaded.')
        model_name = model_name[:-5]

        print('Testing for saving files starts...')
        bboxes = testFast(test_images, model, model_name)

        print('Writing the results to files...')
        writeResults(test_images, bboxes, model_name)

        print('Calculate the results...')
        calculateResults(model_name)

        print('Congratulation! It finished.')
    else:
        print('There is no trained model. Please check.')


if __name__ == '__main__':
    import argparse

    # 命令行参数传入
    parser = argparse.ArgumentParser(description="Test 2D-ChipNet to detect scene text.")
    parser.add_argument('--tryinfo', required=False, metavar="01",
                        help="Description of the training model")
    parser.add_argument('--backbone', required=False,
                        metavar="vgg16", help="Backbone")
    args = parser.parse_args()

    TRY = '01'
    if args.tryinfo:
        TRY = args.tryinfo

    # BackBone
    backbone = "vgg16"
    if args.backbone:
        backbone = args.backbone

    GridImage = True
    ResizedImage_bbox = True
    FilterModel = True
    LocationImage = True
    FilterBBox = False
    DSR_x = 4
    DSR_y = 4
    CNNModel = backbone + '_' + str(DSR_y) + '_' + str(DSR_x) + '_' + TRY  # example: vgg16_4_2_01
    # CNNModel = 'vgg16_' + str(DSR) + '_' + TRY
    ModelPath = './model/' + CNNModel + '/'
    # ModelPath = './tmp/'
    ResultPath = './result/' + CNNModel
    Result = ResultPath + '/predicted_'
    ResultImage = ResultPath + '/predicted_images_'
    CLASS = 1
    CHANNEL = 3
    ResizeH = 608
    ResizeW = 800
    ShortSide = 608
    LongSide = 800
    # NMS_SAME_CLASS = True
    test_pickle = './sample/icdar2013_test_sample_608_800.pkl'
    # scale_path = './sample/train_scale_bbox_608_800_' + \
    #              str(DSR_y) + '_' + str(DSR_x) + '.pkl'  # todo:改变了下采样需要改这里

    print("Try Infomations: ", TRY)

    main()
