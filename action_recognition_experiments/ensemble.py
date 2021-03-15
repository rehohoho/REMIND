import pickle
import argparse

import numpy as np
import torch
from tqdm import tqdm


def main(args):

    r1 = torch.load(args.score1)
    r2 = torch.load(args.score2)

    with open(args.label_file, 'rb') as f:
        _, labels = pickle.load(f)

    print(len(labels), len(r1), len(r2))

    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(r1))):

        l = labels[i]
        r = r1[i] + r2[i] * arg.alpha

        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1

    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(acc, acc5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--score1', default='streaming_experiments/ntu60xsubbone_shuffled_all60/preds_min_trained_0_max_trained_60.pth',
        help='first pt file containing scores per sample.')
    parser.add_argument('--score2', default='streaming_experiments/ntu60xsubjoint_shuffled_all60/preds_min_trained_0_max_trained_60.pth',
        help='second pt file containing scores per sample.')
    parser.add_argument('--label_file', default='/home/ltj/codes/MS-G3D/data/ntu_60/xsub/val_label.pkl',
        help='file containing labels')
    parser.add_argument('--alpha', default=1, help='weighted summation')
    arg = parser.parse_args()

    main(arg)
