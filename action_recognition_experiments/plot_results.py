import json
import argparse
import pprint

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def main(args):

    with open(args.result_file, 'r') as f:
        results = json.load(f)
    
    seen_classes_accuracy = results['seen_classes_top1'][1:]
    xlabs = np.arange(args.base_class_init, args.max_class)

    assert len(xlabs) == len(seen_classes_accuracy), 'Number of classes %s does not match number of result outputs %s.' %(
        len(xlabs), len(seen_classes_accuracy)
    )

    pprint.pprint(seen_classes_accuracy)

    plt.ylim([0, 100])
    plt.plot(xlabs, seen_classes_accuracy)
    plt.savefig('test.png')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_class_init', type=int, default=41)
    parser.add_argument('--max_class', type=int, default=51)
    parser.add_argument('--result_file', type=str, default='./streaming_experiments/pkummdxsubjoint_shuffled/accuracies_min_trained_0_max_trained_51.json')
    args = parser.parse_args()

    main(args)