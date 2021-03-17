import os
import pickle
import json
import argparse
import pprint

import numpy as np
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt


def main(args):

    accuracy_file = os.path.join(args.result_dir, 'accuracies_min_trained_0_max_trained_%s.json' %args.max_class)
    with open(accuracy_file, 'r') as f:
        results = json.load(f)
    
    seen_classes_accuracy = results['seen_classes_top1'][1:]
    xlabs = np.arange(args.min_class, args.max_class)

    assert len(xlabs) == len(seen_classes_accuracy), 'Number of classes %s does not match number of result outputs %s.' %(
        len(xlabs), len(seen_classes_accuracy)
    )

    pprint.pprint(seen_classes_accuracy)

    plt.ylim([0, 100])
    plt.plot(xlabs, seen_classes_accuracy)
    plt.savefig('test.png')

    with open(args.label_path, 'rb') as f:
        _, labels = pickle.load(f)
    labels = np.array(labels)

    acc = []
    la = [] # accuracy on task immediately after training
    taskwise_accs = []
    curr_min_class = 1
    curr_max_class = args.class_increment + curr_min_class

    for class_idx in range(args.min_class, args.max_class + 1):
        preds_file = os.path.join(args.result_dir, 'preds_min_trained_0_max_trained_%s.pth' %class_idx)
        print('reading %s...' %preds_file)
        preds = torch.load(preds_file).argmax(1)

        # get indices, generate match map
        ixs = np.load(args.indices_path)
        seen_ixs = list(np.where(np.logical_and(ixs >= 0, ixs < curr_max_class))[0])
        match = np.zeros((2, len(labels)))
        match[0, seen_ixs] = preds
        match[1, seen_ixs] = labels[seen_ixs]
        
        # extract and compare relevant indices
        seen_acc = np.equal(match[0, seen_ixs], match[1, seen_ixs]).sum() / len(seen_ixs)
        print('acc %s, size %s class %s to %s' %(seen_acc, len(seen_ixs), 0, curr_max_class))
        
        this_ixs = list(np.where(np.logical_and(ixs >= curr_max_class-args.class_increment, ixs < curr_max_class))[0])
        this_acc = np.equal(match[0, this_ixs], match[1, this_ixs]).sum() / len(this_ixs)
        print('la %s, size %s class %s to %s' %(this_acc, len(this_ixs), curr_max_class-args.class_increment, curr_max_class))
        
        acc.append(seen_acc)
        la.append(this_acc)
        
        task_accs = []
        for i in range(curr_min_class, args.max_class, args.class_increment):
            if i < curr_max_class:
                task_ixs = list(np.where(np.logical_and(ixs >= i, ixs < i+args.class_increment))[0])
                task_acc = np.equal(match[0, task_ixs], match[1, task_ixs]).sum() / len(task_ixs)
                task_accs.append(task_acc)
            else:
                task_accs.append(0)
        taskwise_accs.append(task_accs)
        
        curr_max_class += args.class_increment

    acc = np.array(acc)[1:]
    la = np.array(la)[1:]
    taskwise_accs = np.array(taskwise_accs)

    print('acc %s %s' %(acc, acc.mean()))
    print('la %s %s' %(la, la.mean()))
    taskwise_fm = (taskwise_accs[:-1] - taskwise_accs[1:]).max(0)[:-1]/2
    print('fm %s %s' %(taskwise_fm, taskwise_fm.mean()))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_class', type=int, default=42)
    parser.add_argument('--max_class', type=int, default=51)
    parser.add_argument('--class_increment', type=int, default=5)
    parser.add_argument('--result_dir', type=str, default='./streaming_experiments/pkummdxsubbone_shuffled')
    parser.add_argument('--label_path', type=str, default='/home/ltj/datasets/PKUMMD/Process_data_300frame/pku/xsub/val_label.pkl')
    parser.add_argument('--indices_path', type=str, default='./files/indices/pkummdxsub_val_labels.npy')
    args = parser.parse_args()

    main(args)