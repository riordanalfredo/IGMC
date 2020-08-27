import torch
import numpy as np
import sys
import copy
import math
import time
import pdb
import warnings
import traceback
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_functions import *
from data_utils import *
from preprocessing import *
from train_eval import *
from models import *
from igcmf_functions import *

import traceback
import warnings
import sys


def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

    log = file if hasattr(file, 'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(
        message, category, filename, lineno, line))


warnings.showwarning = warn_with_traceback


def parsers():
  # Arguments
    parser = argparse.ArgumentParser(
        description='Inductive Graph-based Collective Matrix Factorization')
    # general settings
    parser.add_argument('--testing', action='store_true', default=False,
                        help='if set, use testing mode which splits all ratings into train/test;\
                    otherwise, use validation model which splits all ratings into \
                    train/val/test and evaluate on val only')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='if set, skip the training and directly perform the \
                    transfer/ensemble/visualization')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='turn on debugging mode which uses a small number of data')
    parser.add_argument(
        '--data-name', default='ml_100k', help='dataset name')
    parser.add_argument('--data-appendix', default='',
                        help='what to append to save-names when saving datasets')
    parser.add_argument('--save-appendix', default='',
                        help='what to append to save-names when saving results')
    parser.add_argument('--max-train-num', type=int, default=None,
                        help='set maximum number of train data to use')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--data-seed', type=int, default=1234, metavar='S',
                        help='seed to shuffle data (1234,2341,3412,4123,1324 are used), \
                    valid only for ml_1m and ml_10m')
    parser.add_argument('--reprocess', action='store_true', default=False,
                        help='if True, reprocess data instead of using prestored .pkl data')
    parser.add_argument('--dynamic-dataset', action='store_true', default=False,
                        help='if True, extract enclosing subgraphs on the fly instead of \
                    storing in disk; works for large datasets that cannot fit into memory')
    parser.add_argument('--keep-old', action='store_true', default=False,
                        help='if True, do not overwrite old .py files in the result folder')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='save model states every # epochs ')
    # subgraph extraction settings
    parser.add_argument('--hop', default=1, metavar='S',
                        help='enclosing subgraph hop number')
    parser.add_argument('--sample-ratio', type=float, default=1.0,
                        help='if < 1, subsample nodes per hop according to the ratio')
    parser.add_argument('--max-nodes-per-hop', default=10000,
                        help='if > 0, upper bound the # nodes per hop by another subsampling')
    parser.add_argument('--use-features', action='store_true', default=False,
                        help='whether to use node features (side information)')
    # IGCMF args
    parser.add_argument('--use-cmf', action='store_true', default=False,
                        help='use collective matrix factorization or skip')
    # edge dropout settings
    parser.add_argument('--adj-dropout', type=float, default=0.2,
                        help='if not 0, random drops edges from adjacency matrix with this prob')
    parser.add_argument('--force-undirected', action='store_true', default=False,
                        help='in edge dropout, force (x, y) and (y, x) to be dropped together')
    # optimization settings
    parser.add_argument('--continue-from', type=int, default=None,
                        help="from which epoch's checkpoint to continue training")
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--lr-decay-step-size', type=int, default=50,
                        help='decay lr by factor A every B steps')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1,
                        help='decay lr by factor A every B steps')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                        help='batch size during training')
    parser.add_argument('--ARR', type=float, default=0.001,
                        help='The adjacenct rating regularizer. If not 0, regularize the \
                    differences between graph convolution parameters W associated with\
                    adjacent ratings')

    # transfer learning, ensemble, and visualization settings
    parser.add_argument('--transfer', default='',
                        help='if not empty, load the pretrained models in this path')
    parser.add_argument('--num-relations', type=int, default=5,
                        help='if transfer, specify num_relations in the transferred model')
    parser.add_argument('--multiply-by', type=int, default=1,
                        help='if transfer, specify how many times to multiply the predictions by')
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='if True, load a pretrained model and do visualization exps')
    parser.add_argument('--ensemble', action='store_true', default=False,
                        help='if True, load a series of model checkpoints and ensemble the results')
    parser.add_argument('--standard-rating', action='store_true', default=False,
                        help='if True, maps all ratings to standard 1, 2, 3, 4, 5 before training')
    # sparsity experiment settings
    parser.add_argument('--ratio', type=float, default=1.0,
                        help="For ml datasets, if ratio < 1, downsample training data to the\
                    target ratio")
    return parser


def main():
    parser = parsers()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.hop = int(args.hop)

    rating_map, post_rating_map = None, {
        x: int(i // (5 / args.num_relations))
        for i, x in enumerate(np.arange(1, 6).tolist())
    }  # refer to IGMC to use rating map

    '''
      1. Prepare train/test (testmode) or train/val/test (valmode) splits
    '''
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))
    if args.testing:
        val_test_appendix = 'testmode'
    else:
        val_test_appendix = 'valmode'
    args.res_dir = os.path.join(
        args.file_dir, 'results/{}{}_{}'.format(
            args.data_name, args.save_appendix, val_test_appendix
        )
    )

    '''
      2. Transfer Learning
    '''
    if args.transfer == '':
        args.model_pos = os.path.join(
            args.res_dir, 'model_checkpoint{}.pth'.format(args.epochs))
    else:
        args.model_pos = os.path.join(
            args.transfer, 'model_checkpoint{}.pth'.format(args.epochs))

    '''
      3. Load data (only ml_100k for now)
    '''
    datasets = []  # objects of data
    if args.data_name == 'ml_100k':
        print("Using official MovieLens dataset split u1.base/u1.test with 20% validation \
              set size...")
        # TODO: Please make it into an object instead of returning multiple variables like this.
        mainData = load_official_trainvaltest_split(
            args.data_name, args.testing, rating_map, post_rating_map, args.ratio, is_cmf=True
        )
        datasets.append(mainData)
    '''
      4. Get the side feature data, uses data extraction from matlab file (Monti et al)
    '''
    if args.use_cmf:
        loaded_data = igcmf_loader(args.data_name, 'user')
        userFeaturesData = load_data_monti(
            loaded_data, args.testing, is_cmf=True, is_debug=args.debug)

        loaded_data = igcmf_loader(args.data_name, 'item')
        itemFeaturesData = load_data_monti(
            loaded_data, args.testing, is_cmf=True, is_debug=args.debug)

        # add to objects of dataset
        datasets.append(userFeaturesData)
        datasets.append(itemFeaturesData)

    '''
          Extract enclosing subgraphs to build the train/test or train/val/test graph datasets. (Note that we must extract enclosing subgraphs for testmode and valmode separately, since the adj_train is different.)
      '''
    train_graphs, val_graphs, test_graphs = None, None, None
    data_combo = (args.data_name, args.data_appendix, val_test_appendix)

    # use preprocessed graph datasets (stored on disk)
    if not args.dynamic_dataset:
        if args.reprocess or not os.path.isdir('data/{}{}/{}/train'.format(*data_combo)):
            # if reprocess=True, delete the previously cached data and reprocess.
            if os.path.isdir('data/{}{}/{}/train'.format(*data_combo)):
                rmtree('data/{}{}/{}/train'.format(*data_combo))
            if os.path.isdir('data/{}{}/{}/val'.format(*data_combo)):
                rmtree('data/{}{}/{}/val'.format(*data_combo))
            if os.path.isdir('data/{}{}/{}/test'.format(*data_combo)):
                rmtree('data/{}{}/{}/test'.format(*data_combo))
            # extract enclosing subgraphs and build the datasets
            train_graphs, val_graphs, test_graphs = collective_links2subgraphs(
                datasets,
                args.hop,
                args.sample_ratio,
                args.max_nodes_per_hop,
                None,
                None,
                args.hop*2+1,
                None,
                args.testing)

            # train_graphs, val_graphs, test_graphs = links2subgraphs(
            #     adj_train,
            #     train_indices,
            #     val_indices,
            #     test_indices,
            #     train_labels,
            #     val_labels,
            #     test_labels,
            #     args.hop,
            #     args.sample_ratio,
            #     args.max_nodes_per_hop,
            #     u_features,
            #     v_features,
            #     args.hop*2+1,
            #     class_values,
            #     args.testing

        if not args.testing:
            val_graphs = MyDataset(
                val_graphs, root='data/{}{}/{}/val'.format(*data_combo))
        test_graphs = MyDataset(
            test_graphs, root='data/{}{}/{}/test'.format(*data_combo))
        train_graphs = MyDataset(
            train_graphs, root='data/{}{}/{}/train'.format(*data_combo))


if __name__ == "__main__":
    main()
