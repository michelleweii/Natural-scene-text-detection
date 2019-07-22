from tqdm import tqdm


def vtMinMaxScaler(label_r, l_min, l_max, feature_range=(0.1, 1)):
    length = label_r.shape[0] * label_r.shape[1] * label_r.shape[2]
    label = label_r.reshape((length, label_r.shape[3]))
    for i in tqdm(range(length)):
        if label[i][0] == label[i][1] == label[i][2] == label[i][3] == 0:
            continue
        else:
            label[i] = (1 - feature_range[0]) * (label[i] - l_min) / (l_max - l_min) + feature_range[0]
    label = label.reshape((label_r.shape[0], label_r.shape[1], label_r.shape[2], label_r.shape[3]))
    return label


def vtMinMaxScaler_inverse(label_r, l_min, l_max, feature_range=(0.1, 1)):
    length = label_r.shape[0] * label_r.shape[1] * label_r.shape[2]
    label = label_r.reshape((length, label_r.shape[3]))
    for i in range(length):
        if label[i][0] == label[i][1] == label[i][2] == label[i][3] == 0:
            continue
        else:
            label[i] = (label[i] - feature_range[0]) * (l_max - l_min) / (1 - feature_range[0]) + l_min
    label = label.reshape((label_r.shape[0], label_r.shape[1], label_r.shape[2], label_r.shape[3]))
    return label
