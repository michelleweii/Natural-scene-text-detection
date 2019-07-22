import os

os.system('python generate_trainval_annotations_from_coco2017.py')
os.system('python get_trainval_data.py')
os.system('python get_test_data.py')
os.system('python cal_train_scale.py')
