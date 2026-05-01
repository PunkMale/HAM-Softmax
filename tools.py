'''
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
'''
import os
import pandas as pd
import logging
import uuid
from operator import itemgetter
from sklearn import metrics
import numpy as np


def check_csv_finished(csv_file_path, max_epoch):
    df = pd.read_csv(csv_file_path)
    epochs = df['Epoch'].astype(str).str.strip().tolist()
    if str(max_epoch) in epochs and epochs[-1] == 'final':
        return True
    else:
        print(f"epochs: {epochs}")
        return False


class Logger:
    def __init__(self, log_path=None):
        self.logger = logging.getLogger(str(uuid.uuid4()))
        self.logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        print(f"Logger initialized with log file {log_path}.")

    def print(self, message):
        self.logger.info(message)


def count_speaker_number(train_list):
    with open(train_list, 'r') as file:
        lines = file.readlines()
    id_set = set()
    for line in lines:
        id_value, _ = line.split(' ', 1)
        id_set.add(id_value.strip())

    print("speaker num:{}".format(len(id_set)))
    return len(id_set)


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
    idxE = np.nanargmin(np.absolute((fnr - fpr)))
    eer = max(fpr[idxE], fnr[idxE]) * 100

    return tunedThreshold, eer, fpr, fnr


def ComputeErrorRates(scores, labels):
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    fnrs = [x / float(fnrs_norm) for x in fnrs]

    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold
