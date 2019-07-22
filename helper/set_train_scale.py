from help import handle_sample as hs
import pickle
import os
from help import sample_resize as sr
import numpy as np


def setTrainScale(w, h, dsr_x, dsr_y, SaveFile):
    l_min = np.array([w * (-1) + dsr_x, h * (-1) + dsr_y, 0, 0], dtype=float)
    l_max = np.array([dsr_x, dsr_y, w, h], dtype=float)
    print(l_min)
    print(l_max)
    # input('stop')
    #
    save_train_scale = open(SaveFile, 'wb')
    pickle.dump(l_min, save_train_scale)
    pickle.dump(l_max, save_train_scale)
    save_train_scale.close()


if __name__ == '__main__':
    print('Please do something')
    # setTrainScale(576,1152,'ssss')
