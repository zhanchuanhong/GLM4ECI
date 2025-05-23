# -*- coding: utf-8 -*-

# This project is for Deberta model.

import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import numpy as np
import tokenizers
import torch
import torch.nn as nn
import logging
import tqdm
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from load_data import load_data
from transformers import BartTokenizer, AdamW
from parameter import parse_args
from utils import isSort,correct_data, collect_mult_event, replace_mult_event,getPredt,assert_handler
from tools import get_batch, calculate, calculate_direction,cal_dir_handler

from model import MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
torch.cuda.empty_cache()
args = parse_args()  # load parameters

if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.model):
    os.mkdir(args.model)
t = time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())
args.log = args.log + 'fold-planB1.40_' + str(args.fold) + '__' + t + '.txt'

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

# set seed for random number
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

setup_seed(args.seed)

# load Roberta model
printlog('Passed args:')
printlog('log path: {}'.format(args.log))
printlog('transformer model: {}'.format(args.model_name))

tokenizer = BartTokenizer.from_pretrained(args.model_name)

# load data tsv file
printlog('Loading data')

train_data, dev_data, test_data = load_data(args)
train_size = len(train_data)
dev_size = len(dev_data)
test_size = len(test_data)
print('Data loaded')

train_data=correct_data(train_data)
dev_data=correct_data(dev_data)
test_data=correct_data(test_data)


multi_event,special_multi_event_token,event_dict,reverse_event_dict,to_add=collect_mult_event(train_data+dev_data+test_data,tokenizer)

additional_mask=['<mask2>','<c1>','<c2>','</c1>','</c2>','<na>']     #50265、50266、50267、50268、50269、50270
to_add[additional_mask[0]]=[50264]
to_add[additional_mask[5]]=[i for i in range(50265)]   # initialization
tokenizer.add_tokens(additional_mask)           #5
tokenizer.add_tokens(special_multi_event_token) #516
args.vocab_size = len(tokenizer)                #50265+5+516



train_data = replace_mult_event(train_data,reverse_event_dict)
dev_data = replace_mult_event(dev_data,reverse_event_dict)
test_data = replace_mult_event(test_data,reverse_event_dict)


# ---------- network ----------

net = MLP(args).to(device)
net.handler(to_add, tokenizer)





no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.wd},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(params=optimizer_grouped_parameters, lr=args.t_lr)


cross_entropy = nn.CrossEntropyLoss().to(device)

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
state = {}


printlog('fold: {}'.format(args.fold))
printlog('batch_size:{}'.format(args.batch_size))
printlog('epoch_num: {}'.format(args.num_epoch))
printlog('initial_t_lr: {}'.format(args.t_lr))
printlog('sample_rate: {}'.format(args.sample_rate))
printlog('sort rate: {}'.format(args.rate_sort))
printlog('seed: {}'.format(args.seed))
printlog('wd: {}'.format(args.wd))
printlog('len_enc_arg: {}'.format(args.len_enc_arg))
printlog('len_dec_arg: {}'.format(args.len_dec_arg))
printlog('few_shot_rate: {}'.format(args.few_shot_rate))

printlog('Start training ...')


##################################  epoch  #################################
for epoch in range(args.num_epoch):
    print('=' * 20)
    printlog('Epoch: {}'.format(epoch))
    torch.cuda.empty_cache()
    args.model = './outmodel/' + 'base__fold-' + str(args.fold) + 'epoch' + str(epoch) + '__' + t + '.pth'
    all_indices = torch.randperm(train_size).split(args.batch_size)
    loss_epoch = 0.0
    acc = 0.0
    train_label, train_label_1, train_label_2, train_predt, train_clabel = [],[],[],[],[]
    train_label_direction, train_predt_direction = [],[]

    f1_pred = torch.IntTensor([]).to(device)
    f1_truth = torch.IntTensor([]).to(device)

    f1_pred_direction, f1_truth_direction = [],[]

    start = time.time()

    printlog('lr:{}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
    printlog('t_lr:{}'.format(optimizer.state_dict()['param_groups'][1]['lr']))

    ############################################################################
    ##################################  train  #################################
    ############################################################################
    test_file = open('./predict/' + str(args.t_lr) + '_' + str(args.few_shot_rate) + '_test_file' + str(args.fold) + '_' + str(epoch) + '.txt', "w")
    dev_file = open('./predict/' + str(args.t_lr) + '_' + str(args.few_shot_rate) + '_dev_file' + str(args.fold) + '_' + str(epoch) + '.txt', "w")

    net.train()
    flag = 'train'
    progress = tqdm.tqdm(total=len(train_data) // args.batch_size + 1, ncols=75,
                         desc='Train {}'.format(epoch))
    total_step = len(train_data) // args.batch_size + 1
    step = 0
    for i, batch_indices in enumerate(all_indices, 1):
        progress.update(1)
        # get a batch of wordvecs
        batch_enc_idx, batch_enc_mask, batch_dec_idx, batch_dec_mask, label_direction, label1, label2, clabel = get_batch(train_data, args, batch_indices, tokenizer, flag)

        batch_enc_idx = batch_enc_idx.to(device)
        batch_enc_mask = batch_enc_mask.to(device)
        batch_dec_idx = batch_dec_idx.to(device)
        batch_dec_mask = batch_dec_mask.to(device)
        length = len(batch_indices)
        # fed data into network
        prediction1,prediction2 = net(batch_enc_idx, batch_enc_mask, batch_dec_idx, batch_dec_mask, flag)

        train_label_1 += label1
        train_label_2 += label2
        train_clabel += clabel

        predt_direction=getPredt(args,prediction1,prediction2)

        predt = [1 if item == 1 or item == 2 else item for item in predt_direction]
        label = [1 if item == 1 or item == 2 else item for item in label_direction]
        train_label += label
        train_predt += predt
        train_label_direction += label_direction
        train_predt_direction += predt_direction

        predt = torch.LongTensor(predt).to(device)
        label = torch.LongTensor(label).to(device)
        label1 = torch.LongTensor(label1).to(device)
        label2 = torch.LongTensor(label2).to(device)

        num_correct = (predt == label).sum()
        acc += num_correct.item()
        f1_pred = torch.cat((f1_pred, predt.type(f1_pred.type())), 0)
        f1_truth = torch.cat((f1_truth, label.type(f1_truth.type())), 0)
        f1_pred_direction += predt_direction
        f1_truth_direction += label_direction


        # loss
        loss = cross_entropy(prediction1, label1) + cross_entropy(prediction2, label2)  # 将两个[MASK]的损失相加

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step += 1


        loss_epoch += loss.item()
        if i % (3000 // args.batch_size) == 0:
            all_p, all_r, all_f, _ = cal_dir_handler(f1_truth_direction, f1_pred_direction)
            printlog('loss={:.4f}, acc2={:.4f}, Recall2={:.4f} F1_score2={:.4f}, acc3={:.4f}, Recall3={:.4f} F1_score3={:.4f},'.format(
                loss_epoch / (3000 // args.batch_size), acc / 3000,
                recall_score(f1_truth.cpu(), f1_pred.cpu()),
                f1_score(f1_truth.cpu(), f1_pred.cpu(),average='macro'),
                all_p, all_r, all_f))
            loss_epoch = 0.0
            acc = 0.0
            f1_pred = torch.IntTensor([]).to(device)
            f1_truth = torch.IntTensor([]).to(device)
            f1_pred_direction, f1_truth_direction = [], []
    end = time.time()
    print('Training Time: {:.2f}s'.format(end - start))

    progress.close()

    ############################################################################
    ##################################  dev  ###################################
    ############################################################################
    temp_batch=4
    all_indices = torch.randperm(dev_size).split(temp_batch)
    dev_label_direction, dev_predt_direction, dev_clabel_direction = [],[],[]

    progress = tqdm.tqdm(total=len(dev_data) // temp_batch + 1, ncols=75,
                         desc='Eval {}'.format(epoch))
    flag = 'dev'
    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_enc_idx, batch_enc_mask, batch_dec_idx, batch_dec_mask, label_direction, label1, label2, clabel = get_batch(dev_data, args, batch_indices, tokenizer, flag)

        batch_enc_idx = batch_enc_idx.to(device)
        batch_enc_mask = batch_enc_mask.to(device)
        batch_dec_idx = batch_dec_idx.to(device)
        batch_dec_mask = batch_dec_mask.to(device)
        length = len(batch_indices)
        # fed data into network
        prediction1,prediction2 = net(batch_enc_idx, batch_enc_mask, batch_dec_idx, batch_dec_mask, flag)

        predt_direction = []
        anser_predt1, anser_predt2 = torch.softmax(prediction1, dim=1), torch.softmax(prediction2, dim=1)
        for iii in range(len(prediction1)):  # 当[MASK1]和[MASK2]在NA上的概率和小于rate_sort时，认为有因果
            if anser_predt1[iii][2].item() + anser_predt2[iii][2].item() < args.rate_sort:
                if anser_predt1[iii][0] + anser_predt2[iii][1] > anser_predt1[iii][1] + anser_predt2[iii][0]:
                    predt_direction.append(1)  # e1 cause e2
                elif anser_predt1[iii][1] + anser_predt2[iii][0] > anser_predt1[iii][0] + anser_predt2[iii][1]:
                    predt_direction.append(2)  # e2 cause e1
            else:
                predt_direction.append(0)

        dev_label_direction += label_direction
        dev_predt_direction += predt_direction
        dev_clabel_direction += clabel

        topic_id = []
        document_id = []
        for temp_i in range(len(predt_direction)):
            topic_id.append(dev_data[batch_indices[temp_i]][1])
            document_id.append(dev_data[batch_indices[temp_i]][2])

        # Save the prediction
        for predt_i in range(len(prediction1)):
            for predt_j in range(len(anser_predt1[predt_i])):
                dev_file.write(str(anser_predt1[predt_i][predt_j].item()) + '\t')
            for predt_j in range(len(anser_predt2[predt_i])):
                dev_file.write(str(anser_predt2[predt_i][predt_j].item()) + '\t')
            dev_file.write(str(label_direction[predt_i]) + '\t')
            dev_file.write(str(clabel[predt_i]) + '\t')
            dev_file.write(str(topic_id[predt_i]) + '\t')
            dev_file.write(str(document_id[predt_i] + '\n'))

    progress.close()

    ############################################################################
    ##################################  test  ##################################
    ############################################################################
    all_indices = torch.randperm(test_size).split(temp_batch)
    test_label_direction, test_predt_direction, test_clabel_direction = [],[],[]
    acc = 0.0

    progress = tqdm.tqdm(total=len(test_data) // temp_batch + 1, ncols=75,
                         desc='Eval {}'.format(epoch))

    net.eval()
    for batch_indices in all_indices:
        progress.update(1)

        # get a batch of dev_data
        batch_enc_idx, batch_enc_mask, batch_dec_idx, batch_dec_mask, label_direction, label1, label2, clabel = get_batch(test_data, args, batch_indices, tokenizer, flag)

        batch_enc_idx = batch_enc_idx.to(device)
        batch_enc_mask = batch_enc_mask.to(device)
        batch_dec_idx = batch_dec_idx.to(device)
        batch_dec_mask = batch_dec_mask.to(device)
        length = len(batch_indices)

        # fed data into network
        prediction1,prediction2 = net(batch_enc_idx, batch_enc_mask, batch_dec_idx, batch_dec_mask, flag)

        predt_direction = []
        anser_predt1, anser_predt2 = torch.softmax(prediction1, dim=1), torch.softmax(prediction2, dim=1)
        for iii in range(len(prediction1)):  # 当[MASK1]和[MASK2]在NA上的概率和小于rate_sort时，认为有因果
            if anser_predt1[iii][2].item() + anser_predt2[iii][2].item() < args.rate_sort:
                if anser_predt1[iii][0] + anser_predt2[iii][1] > anser_predt1[iii][1] + anser_predt2[iii][0]:
                    predt_direction.append(1)  # e1 cause e2
                elif anser_predt1[iii][1] + anser_predt2[iii][0] > anser_predt1[iii][0] + anser_predt2[iii][1]:
                    predt_direction.append(2)  # e2 cause e1
            else:
                predt_direction.append(0)

        test_label_direction += label_direction
        test_predt_direction += predt_direction
        test_clabel_direction += clabel

        topic_id = []
        document_id = []
        for temp_i in range(len(predt_direction)):
            topic_id.append(test_data[batch_indices[temp_i]][1])
            document_id.append(test_data[batch_indices[temp_i]][2])

        for predt_i in range(len(prediction1)):
            for predt_j in range(len(anser_predt1[predt_i])):
                test_file.write(str(anser_predt1[predt_i][predt_j].item()) + '\t')
            for predt_j in range(len(anser_predt2[predt_i])):
                test_file.write(str(anser_predt2[predt_i][predt_j].item()) + '\t')
            test_file.write(str(label_direction[predt_i]) + '\t')
            test_file.write(str(clabel[predt_i]) + '\t')
            test_file.write(str(topic_id[predt_i]) + '\t')
            test_file.write(str(document_id[predt_i] + '\n'))

    progress.close()

    ############################################################################
    ##################################  result  ##################################
    ############################################################################
    ######### Train Results Print #########
    printlog('-------------------')
    all_p, all_r, all_f, _ = cal_dir_handler(train_label_direction, train_predt_direction)
    printlog("TIME: {}".format(time.time() - start))
    printlog('EPOCH : {}'.format(epoch))
    printlog("TRAIN:")
    printlog('acc2={:.4f}, Recall2={:.4f} F1_score2={:.4f}, acc3={:.4f}, Recall3={:.4f} F1_score3={:.4f},'.format(
            precision_score(train_label, train_predt), recall_score(train_label, train_predt),
            f1_score(train_label, train_predt, average='macro'),
            all_p, all_r, all_f))

    ######### Dev Results Print #########
    printlog("DEV:")
    dev_label = [1 if item == 1 or item == 2 else item for item in dev_label_direction]
    dev_predt = [1 if item == 1 or item == 2 else item for item in dev_predt_direction]
    dev_intra_p,dev_intra_r,dev_intra_f, dev_cross_p,dev_cross_r,dev_cross_f, dev_all_p,dev_all_r,dev_all_f,dev_matrix = \
        calculate(dev_label, dev_predt, dev_clabel_direction)
    dev_intra_dir_p,dev_intra_dir_r,dev_intra_dir_f, dev_cross_dir_p,dev_cross_dir_r,dev_cross_dir_f, dev_all_dir_p,dev_all_dir_r,dev_all_dir_f, dev_matrix_dir = \
        calculate_direction(dev_label_direction, dev_predt_direction, dev_clabel_direction)

    printlog('\tINTRA-SENTENCE:')
    printlog("\t\tprecision_2: {:.2f}\trecall_2: {:.2f}\tf1_2: {:.2f}\tprecision_3: {:.2f}\trecall_3: {:.2f}\tf1_3: {:.2f}"
             .format(dev_intra_p, dev_intra_r, dev_intra_f,dev_intra_dir_p,dev_intra_dir_r,dev_intra_dir_f))
    printlog('\tCROSS-SENTENCE:')
    printlog("\t\tprecision_2: {:.2f}\trecall_2: {:.2f}\tf1_2: {:.2f}\tprecision_3: {:.2f}\trecall_3: {:.2f}\tf1_3: {:.2f}"
             .format(dev_cross_p,dev_cross_r,dev_cross_f, dev_cross_dir_p,dev_cross_dir_r,dev_cross_dir_f))
    printlog('\tINTRA + CROSS:')
    printlog("\t\tprecision_2: {:.2f}\trecall_2: {:.2f}\tf1_2: {:.2f}\tprecision_3: {:.2f}\trecall_3: {:.2f}\tf1_3: {:.2f}"
             .format(dev_all_p,dev_all_r,dev_all_f, dev_all_dir_p,dev_all_dir_r,dev_all_dir_f))

    ######### Dev Results Print #########
    printlog("TEST:")
    test_label = [1 if item == 1 or item == 2 else item for item in test_label_direction]
    test_predt = [1 if item == 1 or item == 2 else item for item in test_predt_direction]
    test_intra_p,test_intra_r,test_intra_f, test_cross_p,test_cross_r,test_cross_f, test_all_p,test_all_r,test_all_f,test_matrix = \
        calculate(test_label, test_predt, test_clabel_direction)
    test_intra_dir_p,test_intra_dir_r,test_intra_dir_f, test_cross_dir_p,test_cross_dir_r,test_cross_dir_f, test_all_dir_p,test_all_dir_r,test_all_dir_f, test_matrix_dir = \
        calculate_direction(test_label_direction, test_predt_direction, test_clabel_direction)

    printlog('\tINTRA-SENTENCE:')
    printlog("\t\tprecision_2: {:.2f}\trecall_2: {:.2f}\tf1_2: {:.2f}\tprecision_3: {:.2f}\trecall_3: {:.2f}\tf1_3: {:.2f}"
             .format(test_intra_p, test_intra_r, test_intra_f,test_intra_dir_p,test_intra_dir_r,test_intra_dir_f))
    printlog('\tCROSS-SENTENCE:')
    printlog("\t\tprecision_2: {:.2f}\trecall_2: {:.2f}\tf1_2: {:.2f}\tprecision_3: {:.2f}\trecall_3: {:.2f}\tf1_3: {:.2f}"
             .format(test_cross_p,test_cross_r,test_cross_f, test_cross_dir_p,test_cross_dir_r,test_cross_dir_f))
    printlog('\tINTRA + CROSS:')
    printlog("\t\tprecision_2: {:.2f}\trecall_2: {:.2f}\tf1_2: {:.2f}\tprecision_3: {:.2f}\trecall_3: {:.2f}\tf1_3: {:.2f}"
             .format(test_all_p,test_all_r,test_all_f, test_all_dir_p,test_all_dir_r,test_all_dir_f))

    assert_handler(dev_all_p,dev_all_r,dev_all_f,dev_matrix,dev_matrix_dir)
    assert_handler(test_all_p,test_all_r,test_all_f,test_matrix,test_matrix_dir)

    breakout += 1
    breakout_dir += 1

    # record the best result
    if dev_all_f > dev_best_intra_cross['f1']:
        printlog('New best epoch...')
        dev_best_intra_cross = {'epoch': epoch,'p': dev_all_p,'r': dev_all_r,'f1': dev_all_f}
        best_intra_cross = {'epoch': epoch,'p': test_all_p,'r': test_all_r,'f1': test_all_f}
        best_intra = {'epoch': epoch,'p': test_intra_p,'r': test_intra_r,'f1': test_intra_f}
        best_cross = {'epoch': epoch,'p': test_cross_p,'r': test_cross_r,'f1': test_cross_f}
        best_epoch = epoch
        breakout = 0
    if dev_all_dir_f > dev_best_intra_cross_dir['f1']:
        printlog('New best epoch...')
        dev_best_intra_cross_dir = {'epoch': epoch, 'p': dev_all_dir_p, 'r': dev_all_dir_r, 'f1': dev_all_dir_f}
        best_intra_cross_dir = {'epoch': epoch, 'p': test_all_dir_p, 'r': test_all_dir_r, 'f1': test_all_dir_f}
        best_intra_dir = {'epoch': epoch, 'p': test_intra_dir_p, 'r': test_intra_dir_r, 'f1': test_intra_dir_f}
        best_cross_dir = {'epoch': epoch, 'p': test_cross_dir_p, 'r': test_cross_dir_r, 'f1': test_cross_dir_f}
        best_epoch_dir = epoch
        breakout_dir = 0
    # state = {'roberta_model': net.roberta_model.state_dict(),
    #          'roberta_model2': net.roberta_model2.state_dict()}
    # torch.save(state, args.model)
    # time.sleep(15)

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

    test_file.close()
    dev_file.close()


# torch.save(state, args.model)
