# -*- coding: utf-8 -*-


# This project is for Deberta model.

import time
import os
import torch
import logging
from datetime import datetime
from parameter import parse_args
from sklearn.metrics import f1_score, precision_score, recall_score
from utils import isSort,correct_data, collect_mult_event, replace_mult_event,convert,assert_handler
from tools import get_batch, calculate,calculate_direction

torch.cuda.empty_cache()
args = parse_args()  # load parameters


args.log = './predict_out/'
if not os.path.exists(args.log):
    os.mkdir(args.log)

# args.fold = 2
mode = 'soft'
l_tr='5e-06'

t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + mode + '_' + 'fold-' + str(args.fold) + '__' + t + '.txt'

# refine
for name in logging.root.manager.loggerDict:
    if 'transformers' in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)

logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=args.log,
                    filemode='w')

logger = logging.getLogger(__name__)


def printlog(message: object, printout: object = True) -> object:
    message = '{}: {}'.format(datetime.now(), message)
    if printout:
        print(message)
    logger.info(message)


def readFile(file_name):
    prediction1,prediction2,label,clabel=[],[],[],[]
    file_out = open(file_name, "r")
    data = file_out.read()
    data = data.split('\n')[:-1]
    for i in range(len(data)):
        data[i] = data[i].split('\t')
    for i in range(len(data)):
        prediction1.append([float(d) for d in data[i][:3]])
        prediction2.append([float(d) for d in data[i][3:6]])
        label.append(int(data[i][6]))
        clabel.append(int(data[i][7]))
    return prediction1,prediction2,label,clabel


def hardJudge(prediction1,prediction2):
    predt1 = [sub_list.index(max(sub_list)) for sub_list in prediction1]
    predt2 = [sub_list.index(max(sub_list)) for sub_list in prediction2]
    predt_direction = []
    for i in range(len(predt1)):
        if predt1[i] == 0 and predt2[i] == 1:
            predt_direction.append(1)
        elif predt1[i] == 1 and predt2[i] == 0:
            predt_direction.append(2)
        else:
            predt_direction.append(0)
    return predt_direction

def softJudge(prediction1,prediction2,rate_sort,direction_sort):
    predt_direction = []
    for iii in range(len(prediction1)):  # 当[MASK1]和[MASK2]在NA上的概率和小于rate_sort时，认为有因果
        if prediction1[iii][2] + prediction2[iii][2] < rate_sort:
            if prediction1[iii][0] + prediction2[iii][1] > prediction1[iii][1] + prediction2[iii][0]+direction_sort:
                predt_direction.append(1)  # e1 cause e2
            else:
            # elif prediction1[iii][1] + prediction2[iii][0] +direction_sort> prediction1[iii][0] + prediction2[iii][1]:
                predt_direction.append(2)  # e2 cause e1
        else:
            predt_direction.append(0)
    return predt_direction


best_intra_rate = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_cross_rate = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_intra_cross_rate = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_intra_dir_rate = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_cross_dir_rate = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_intra_cross_dir_rate = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
best_epoch_rate, best_epoch_dir_rate = 0,0
best_rate_sort, best_rate_sort_dir = 0,0
best_direction_rate_sort, best_direction_rate_sort_dir = 0,0

rate_sort_list = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.00,
                  1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50]
direction_sort_list = [-0.35,-0.3,-0.25,-0.2,-0.15,-0.1,-0.05,0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35]

for direction_sort in direction_sort_list:
    for rate_sort in rate_sort_list:
        # save model and result
        best_intra = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
        best_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
        best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
        dev_best_intra_cross = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
        best_intra_dir = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
        best_cross_dir ={'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
        best_intra_cross_dir = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
        dev_best_intra_cross_dir = {'epoch': 0, 'p': 0, 'r': 0, 'f1': 0}
        best_epoch, best_epoch_dir = 0,0
        breakout, breakout_dir = 0,0

        for epoch in range(args.num_epoch):
            dev_file = 'predict/'+l_tr+'_0_dev_file' + str(args.fold) + '_' + str(epoch) + '.txt'
            test_file = 'predict/'+l_tr+'_0_test_file' + str(args.fold) + '_' + str(epoch) + '.txt'
            dev_prediction1,  dev_prediction2,  dev_label_direction,  dev_clabel =  readFile(dev_file)
            test_prediction1, test_prediction2, test_label_direction, test_clabel = readFile(test_file)

            if mode == 'hard':
                dev_predt_direction = hardJudge(dev_prediction1,dev_prediction2)
            else:
                dev_predt_direction = softJudge(dev_prediction1,dev_prediction2,rate_sort,direction_sort)
            dev_label = [1 if item == 1 or item == 2 else item for item in dev_label_direction]
            dev_predt = [1 if item == 1 or item == 2 else item for item in dev_predt_direction]
            dev_intra_p, dev_intra_r, dev_intra_f, dev_cross_p, dev_cross_r, dev_cross_f, dev_all_p, dev_all_r, dev_all_f, dev_matrix = \
                calculate(dev_label, dev_predt, dev_clabel)
            dev_intra_dir_p, dev_intra_dir_r, dev_intra_dir_f, dev_cross_dir_p, dev_cross_dir_r, dev_cross_dir_f, dev_all_dir_p, dev_all_dir_r, dev_all_dir_f, dev_matrix_dir = \
                calculate_direction(dev_label_direction, dev_predt_direction, dev_clabel)

            if mode == 'hard':
                test_predt_direction = hardJudge(test_prediction1,test_prediction2)
            else:
                test_predt_direction = softJudge(test_prediction1,test_prediction2,rate_sort,direction_sort)
            test_label = [1 if item == 1 or item == 2 else item for item in test_label_direction]
            test_predt = [1 if item == 1 or item == 2 else item for item in test_predt_direction]
            test_intra_p, test_intra_r, test_intra_f, test_cross_p, test_cross_r, test_cross_f, test_all_p, test_all_r, test_all_f, test_matrix = \
                calculate(test_label, test_predt, test_clabel)
            test_intra_dir_p, test_intra_dir_r, test_intra_dir_f, test_cross_dir_p, test_cross_dir_r, test_cross_dir_f, test_all_dir_p, test_all_dir_r, test_all_dir_f, test_matrix_dir = \
                calculate_direction(test_label_direction, test_predt_direction, test_clabel)

            assert_handler(dev_all_p, dev_all_r, dev_all_f, dev_matrix, dev_matrix_dir)
            assert_handler(test_all_p, test_all_r, test_all_f, test_matrix, test_matrix_dir)

            breakout += 1
            breakout_dir += 1

            # record the best result
            if dev_all_f > dev_best_intra_cross['f1']:
                dev_best_intra_cross = {'epoch': epoch, 'p': dev_all_p, 'r': dev_all_r, 'f1': dev_all_f}
                best_intra_cross = {'epoch': epoch, 'p': test_all_p, 'r': test_all_r, 'f1': test_all_f}
                best_intra = {'epoch': epoch, 'p': test_intra_p, 'r': test_intra_r, 'f1': test_intra_f}
                best_cross = {'epoch': epoch, 'p': test_cross_p, 'r': test_cross_r, 'f1': test_cross_f}
                best_epoch = epoch
                breakout = 0
            if dev_all_dir_f > dev_best_intra_cross_dir['f1']:
                dev_best_intra_cross_dir = {'epoch': epoch, 'p': dev_all_dir_p, 'r': dev_all_dir_r, 'f1': dev_all_dir_f}
                best_intra_cross_dir = {'epoch': epoch, 'p': test_all_dir_p, 'r': test_all_dir_r, 'f1': test_all_dir_f}
                best_intra_dir = {'epoch': epoch, 'p': test_intra_dir_p, 'r': test_intra_dir_r, 'f1': test_intra_dir_f}
                best_cross_dir = {'epoch': epoch, 'p': test_cross_dir_p, 'r': test_cross_dir_r, 'f1': test_cross_dir_f}
                best_epoch_dir = epoch
                breakout_dir = 0

        printlog('*' * 10 + str(rate_sort) + '*' * 10)
        printlog('*' * 10 + str(direction_sort) + '*' * 10)
        printlog('=' * 20)
        printlog('Undirection:')
        printlog('Best result at epoch: {}'.format(best_epoch))
        printlog('Eval intra: {}'.format(best_intra))
        printlog('Eval cross: {}'.format(best_cross))
        printlog('Eval intra cross: {}'.format(best_intra_cross))
        printlog('Breakout: {}'.format(breakout))
        printlog('=' * 20)
        printlog('Direction:')
        printlog('Best result at epoch: {}'.format(best_epoch_dir))
        printlog('Eval intra: {}'.format(best_intra_dir))
        printlog('Eval cross: {}'.format(best_cross_dir))
        printlog('Eval intra cross: {}'.format(best_intra_cross_dir))
        printlog('Breakout: {}'.format(breakout_dir))

        if best_intra_cross['f1'] > best_intra_cross_rate['f1']:
            best_intra_cross_rate=best_intra_cross
            best_cross_rate=best_cross
            best_intra_rate=best_intra
            best_epoch_rate=best_epoch
            best_rate_sort = rate_sort
            best_direction_rate_sort = direction_sort
        if best_intra_cross_dir['f1'] > best_intra_cross_dir_rate['f1']:
            best_intra_cross_dir_rate=best_intra_cross_dir
            best_cross_dir_rate=best_cross_dir
            best_intra_dir_rate=best_intra_dir
            best_epoch_dir_rate=best_epoch_dir
            best_rate_sort_dir=rate_sort
            best_direction_rate_sort_dir=direction_sort

printlog('=' * 40)
printlog('*' * 40)
printlog('=' * 40)
printlog('Undirection:')
printlog('Best result at epoch: {}'.format(best_epoch_rate))
printlog('Eval intra: {}'.format(best_intra_rate))
printlog('Eval cross: {}'.format(best_cross_rate))
printlog('Eval intra cross: {}'.format(best_intra_cross_rate))
printlog('Rate_sort: {}'.format(best_rate_sort))
printlog('best_direction_rate_sort: {}'.format(best_direction_rate_sort))
printlog('=' * 20)
printlog('Direction:')
printlog('Best result at epoch: {}'.format(best_epoch_dir_rate))
printlog('Eval intra: {}'.format(best_intra_dir_rate))
printlog('Eval cross: {}'.format(best_cross_dir_rate))
printlog('Eval intra cross: {}'.format(best_intra_cross_dir_rate))
printlog('Rate_sort: {}'.format(best_rate_sort_dir))
printlog('best_direction_rate_sort_dir: {}'.format(best_direction_rate_sort_dir))
# print(data)