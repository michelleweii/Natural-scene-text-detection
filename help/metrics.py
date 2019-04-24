import os
from tqdm import tqdm
from help import polygon_calculate as pc
import datetime
import pandas as pd
from help import NMS


def calculateF1score(ResultPath, name_model, iou=0.5):
    gt_dir = ResultPath + '/ground-truth/'
    predict_dir = ResultPath + '/predicted_' + name_model + '/'

    gt_list = os.listdir(gt_dir)
    predict_list = os.listdir(predict_dir)

    if 'desktop.ini' in gt_list:
        gt_list.remove('desktop.ini')
    if gt_list != predict_list:
        print("the number of file is not equal!!")
        os._exit(0)

    TP = 0
    FN = 0
    FP = 0
    for gt_file in tqdm(gt_list):
        gt = []
        with open(gt_dir + gt_file, 'r') as f_gt:
            for gt_line in f_gt:
                line_split = gt_line.strip().split(' ')
                # if int(float(line_split[-1])) == 0:
                gt.append(list(map(int, line_split[1:5])))

        predict = []
        with open(predict_dir + gt_file, 'r') as f_pred:
            for pred_line in f_pred:
                line_split = pred_line.strip().split(' ')
                predict.append(list(map(float, line_split[2:])))

        for i in range(len(gt)):
            if len(predict) > 0:
                hit_iou = 0
                hit_j = None
                for j in range(len(predict)):
                    IoU = NMS.calculate_iou(gt[i], predict[j])
                    if IoU > iou and IoU > hit_iou:
                        hit_j = j
                        hit_iou = IoU
                if hit_iou == 0:
                    FN += 1
                else:
                    TP += 1
                    del predict[hit_j]
            else:
                FN += (len(gt) - i)
                break
        if len(predict) > 0:
            FP += len(predict)

    ACC = TP / (TP + FP + FN)
    Precision = TP / (TP + FP)
    Recall = TP / (TP + FN)
    if Precision + Recall == 0:
        F1 = 0
    else:
        F1 = 2 * Precision * Recall / (Precision + Recall)

    print("1. Accuracy: {}".format(ACC))
    print("2. Precision: {}".format(Precision))
    print("3. Recall: {}".format(Recall))
    print("4. F1-score: {}".format(F1))
    print("5. TP, True positives: {}".format(TP))
    print("6. FP, False positives: {}".format(FP))
    print("7. FN, False negatives: {}".format(FN))
    print("8. the total test number: {}".format(len(gt_list)))
    current_time = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d %H:%M:%S'),  # 获取当前时间
    print(current_time)

    Result = {'Accuracy': ACC,
              'Precision': Precision,
              'Recall': Recall,
              'F1-score': F1,
              'TP': TP,
              'FP': FP,
              'FN': FN,
              'test_time': current_time,
              'model': name_model}

    columns = ['test_time', 'model', 'Accuracy', 'Precision',
               'Recall', 'F1-score', 'TP', 'FP', 'FN']

    DataFrame = pd.DataFrame([Result])

    ResPath = 'icdar2013_Results.csv'
    if os.path.exists(ResPath):
        DataFrame.to_csv(ResPath, sep=',', columns=columns,
                         header=None, mode='a', index=0)
    else:
        DataFrame.to_csv(ResPath, sep=',', columns=columns, index=0)

    return F1


if __name__ == '__main__':
    path_res = '../result/vgg16'
    mn = 'try'
    print(calculateF1score(path_res, mn))
