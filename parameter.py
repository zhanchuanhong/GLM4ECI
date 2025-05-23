# -*- coding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='ECI')

    # Dataset
    parser.add_argument('--fold', default=1, type=int, help='Fold number used to be test set')
    parser.add_argument('--sample_rate', default=0, type=float, help='Negetive sample rate')  # 抛弃掉这么多比例的负例
    parser.add_argument('--few_shot_rate', default=0, type=float, help='Few shot rate')  # 抛弃掉这么多比例的数据
    parser.add_argument('--rate_sort', default=1.0, type=float, help='Judge rate')
    parser.add_argument('--len_enc_arg', default=200, type=int, help='Sentence length')
    parser.add_argument('--len_dec_arg', default=15, type=int, help='Template length')

    # Model
    parser.add_argument('--model_name', default="BartForGeneration/bart-base", type=str, help='Model used to be encoder')
    parser.add_argument('--vocab_size', default=50265, type=int, help='Size of RoBERTa vocab')

    # Prompt and Contrastive Training
    parser.add_argument('--num_epoch', default=15, type=int, help='Number of total epochs to run prompt learning')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size for prompt learning')
    parser.add_argument('--t_lr', default=1e-5, type=float, help='Initial lr')
    parser.add_argument('--wd', default=1e-2, type=float, help='weight decay')

    # Others
    parser.add_argument('--seed', default=209, type=int, help='Seed for reproducibility')
    parser.add_argument('--log', default='./out/', type=str, help='Log result file name')
    parser.add_argument('--model', default='./outmodel/', type=str, help='Model parameters result file name')
    parser.add_argument('--model_time_day', default='111', type=str, help='The saved model time (day)')
    parser.add_argument('--model_time_min', default='111', type=str, help='The saved model time (minutes)')

    args = parser.parse_args()
    return args
