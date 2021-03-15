'''
Generates sampling index according to class order from text file.
Expects preprocessed data based on 2s-AGCN method.
'''

import argparse
import os
import numpy as np
import pickle

import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_pickle(file_path):
    try:
        with open(file_path) as f:
            sample_name, label = pickle.load(f)
    except:
        # for pickle file from python2
        with open(file_path, 'rb') as f:
            sample_name, label = pickle.load(f, encoding='latin1')
    
    return sample_name, label


def get_class_order(unpickled_dataset):
    
    # get unique classes from imagefolder class names
    default_class_order = []
    for v in unpickled_dataset:
        if v not in default_class_order:
            default_class_order.append(v)

    return default_class_order


def generate_index_npy(default_labels, user_desired_order, output_file):
    
    default_class_order = get_class_order(default_labels)
    print(f'\nDefault class order {default_class_order}')
    
    # compute mapping from default pytorch order to user order
    # e.g. dclass 0,1,2,3, wclass 2,3,1,0, loading index 3,2,0,1
    # e.g. dclass 2,3,1,0, wclass 0,1,2,3, loading index 2,3,1,0
    # map = []
    # for v in default_class_order:
    #     ix = user_desired_order.index(v)
    #     map.append(ix)
    # print(f'\nMapping {map}')

    # relabel all samples and save to numpy files
    new_labels = np.empty_like(default_labels)
    for i in range(len(new_labels)):
        new_labels[i] = user_desired_order.index(default_labels[i])

    print(f'Saving numpy file to {output_file}')
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    np.save(output_file, np.array(new_labels))


def main(args):

    print('\nloading the data...')
    _, l_tr = load_pickle(args.l_tr_path)
    _, l_te = load_pickle(args.l_te_path)
    print(f'\nloading {args.l_tr_path} done. Labels found for training {np.unique(l_tr)}')
    print(f'\nloading {args.l_te_path} done. Labels found for test {np.unique(l_te)}')

    # load in user desired order from text file
    print(' Opening user desired order...')
    with open(args.class_order_text_file) as f:
        lines = [line.rstrip() for line in f]  # grab each class name from line in text file
    lines = [int(i) for i in lines]
    print(f'\nloading user desired order {args.class_order_text_file} done. Labels found {lines}')

    save_path = os.path.join(args.labels_dir, '%s_train_labels' %args.output_prefix)
    generate_index_npy(l_tr, lines, save_path)
    save_path = os.path.join(args.labels_dir, '%s_val_labels' %args.output_prefix)
    generate_index_npy(l_te, lines, save_path)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--l_tr_path', default='/home/ltj/codes/MS-G3D/data/ntu_60/xview/train_label.pkl', help='nturgbd60 testing data .npy')
    parser.add_argument('--l_te_path', default='/home/ltj/codes/MS-G3D/data/ntu_60/xview/val_label.pkl', help='nturgbd60 testing labels .pkl')
    parser.add_argument('--labels_dir', type=str, default='./files/indices')
    parser.add_argument('--class_order_text_file', type=str, default='./files/nturgbd60_class_order_50_10.txt')
    parser.add_argument('--output_prefix', type=str, default='nturgbd60xview')
    args = parser.parse_args()

    main(args)
