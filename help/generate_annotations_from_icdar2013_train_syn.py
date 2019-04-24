# -*- coding:utf-8 -*-
from __future__ import print_function, division
import os
import random

# 产生的文件路径和名字
generate_txt = '../sample/icdar2013_annotations_train_syn_set.txt'
# 样本存在的位置，分别存在不同的地方，最后取到一个文件
icdar2013_dir = ['../dataset']


# 产生样本注释文件
def generate(dirs_):
    for i in range(len(dirs_)):
        # 检查路径是否正确
        if not os.path.exists(dirs_[i]):
            print('label dir: {} doest not exist'.format(dirs_[i]))
            exit(0)

    # 打开要写入的文件，如果存在会覆盖掉
    target_file = open(generate_txt, 'w')

    # 逐个位置取样本
    for dir_ in dirs_:

        label_dir_full = dir_ + '/icdar2013_synth_image_gt/'  # 标注文件夹
        img_dir_full = dir_ + '/icdar2013_synth_image/'  # 图片文件夹
        all_label_files = [i for i in os.listdir(label_dir_full) if i.endswith('.txt')]
        all_image_files = [i for i in os.listdir(img_dir_full) if i.endswith('.jpg')]

        # 对比图片和标注是否匹配
        for j in range(len(all_label_files)):
            if all_label_files[j].split('.')[0] != all_image_files[j].split('.')[0]:
                print('The image does not match its label.')
                exit(0)

        # all_label_files = random.sample(all_label_files, each_group_samples)
        print('got {} label files.'.format(len(all_label_files)))

        for label_file_name in all_label_files:
            label_file = os.path.join(label_dir_full, label_file_name)

            # 打开文件
            with open(label_file, 'r') as f:
                for l in f.readlines():
                    each_line = l.strip().split(' ')
                    # 检查标注文件每行数据的数目是否正确
                    if len(each_line) < 5:
                        print(label_file)
                        print('There is a mistake, please check.')
                        os._exit(0)
                    elif len(each_line) > 5:
                        for i in range(len(each_line) - 5):
                            # print(each_line)
                            each_line[4] += ' ' + each_line[5 + i]
                        each_line = each_line[:5]
                    x1, y1, x2, y2, class_name = each_line
                    # x2 = int(x1) + int(w) - 1
                    # y2 = int(y1) + int(h) - 1

                    # 检查标注文件每行的标注坐标是否正确
                    # print(x1, x2, y1, y2)

                    if int(x1) > int(x2) or int(y1) > int(y2):
                        print('The axis label is wrong, please check.')
                        print(x1, x2, y1, y2)
                        print(label_file)
                        exit(0)

                    # 检查标注文件每行的类别名称是否正确
                    # if class_name not in [str(x) for x in range(0, 10)]:
                    #     print('The class label is wrong, please check.')
                    #     print(class_name)
                    #     print(label_file)
                    #     exit(0)

                    # 写入信息到文件
                    target_file.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(
                        os.path.join(img_dir_full, label_file_name.replace('txt', 'jpg')),
                        x1, y1, x2, y2, class_name))

    # 结束关闭文件
    target_file.close()
    print('convert finished.')


if __name__ == '__main__':
    generate(icdar2013_dir)
