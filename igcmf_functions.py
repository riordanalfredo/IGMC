from tqdm import tqdm
from util_functions import neighbors, one_hot, parallel_worker
import os
import sys
import pdb
import math
import time
import multiprocessing as mp
import scipy.sparse as ssp
import networkx as nx
import numpy as np


class ImportedDataset:
    def __init__(self, variables):
        # TODO: create variables validator

        # features
        self.u_features = variables['u_features']
        self.v_features = variables['v_features']
        # matrix
        self.adj_train = variables['adj_train']

        # training labels & indices
        self.train_labels = variables['train_labels']
        self.train_u_indices = variables['train_u_indices']
        self.train_v_indices = variables['train_v_indices']

        # validation labels & indices
        self.val_labels = variables['val_labels']
        self.val_u_indices = variables['val_u_indices']
        self.val_v_indices = variables['val_v_indices']

        # testing labels & indices
        self.test_labels = variables['test_labels']
        self.test_u_indices = variables['test_u_indices']
        self.test_v_indices = variables['test_v_indices']

        # class_values
        self.class_values = variables['class_values']


'''
    A method to split side matrix into train, val, and test data.
    Slightly similar to load_monti. The difference is the train, val, and test data are depending on the labels and indices of the relation/main matrix.
'''


def load_side_matrix(loaded_data, train_labels, train_u_indices, train_v_indices,
                     val_labels, val_u_indices, val_v_indices, test_labels, test_u_indices,
                     test_v_indices, is_testing=False, is_cmf=True):
    M = loaded_data['M']
    Otraining = loaded_data['Otraining']
    Otest = loaded_data['Otest']
    num_users = loaded_data['num_users']
    num_items = loaded_data['num_items']
    u_features = loaded_data['u_features']
    v_features = loaded_data['v_features']

    # get labels
    u_nodes_ratings = np.where(M)[0]  # nonzeros label u
    v_nodes_ratings = np.where(M)[1]  # nonzeros label v
    ratings = M[np.where(M)]  # dot product result

    # specify types
    u_nodes_ratings, v_nodes_ratings = u_nodes_ratings.astype(
        np.int64), v_nodes_ratings.astype(np.int32)
    ratings = ratings.astype(np.float64)

    # re-assign
    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    print('number of u nodes = ', len(set(u_nodes)))
    print('number of v nodes = ', len(set(v_nodes)))

    return u_nodes


'''
    Method to convert networkx graph to PyGGraph format using pytorch. 
'''


def nx_to_PyGGraph(g, graph_label, node_labels, node_features, max_node_label, class_values):
    # convert networkx graph to pytorch_geometric data format
    y = torch.FloatTensor([class_values[graph_label]])
    if len(g.edges()) == 0:
        i, j = [], []
    else:
        i, j = zip(*g.edges())
    edge_index = torch.LongTensor([i+j, j+i])
    edge_type_dict = nx.get_edge_attributes(g, 'type')
    edge_type = torch.LongTensor(
        [edge_type_dict[(ii, jj)] for ii, jj in zip(i, j)])
    edge_type = torch.cat([edge_type, edge_type], 0)
    edge_attr = torch.FloatTensor(
        class_values[edge_type]
    ).unsqueeze(1)  # continuous ratings, num_edges * 1
    x = torch.FloatTensor(one_hot(node_labels, max_node_label+1))
    if node_features is not None:
        if type(node_features) == list:
            # node features are only provided for target user and item
            u_feature, v_feature = node_features
        else:
            # node features are provided for all nodes
            x2 = torch.FloatTensor(node_features)
            x = torch.cat([x, x2], 1)

    data = Data(x, edge_index, edge_attr=edge_attr, y=y)
    data.edge_type = edge_type
    if type(node_features) == list:
        data.u_feature = torch.FloatTensor(u_feature).unsqueeze(0)
        data.v_feature = torch.FloatTensor(v_feature).unsqueeze(0)
    return data


'''
    Method to extract the subgraph from collective matrices
'''


def collective_subgraph_extraction_labeling(ind, A, h=1, sample_ratio=1.0, max_nodes_per_hop=None,
                                            u_features=None, v_features=None, class_values=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    num_types = len(ind) + 2  # for now add 2 number of node types, M notation
    nodes = [ind[i] for i in range(num_types)]  # O(M)
    distances = [[0] for j in range(num_types)]  # O(M)
    visited_sets = [set([ind[k]]) for k in range(num_types)]  # O(M)
    fringe_sets = [set([ind[l]]) for l in range(num_types)]  # O(M)

    # TODO: DONT DELETE! USE THIS AS REFERENCE!
    # u_nodes, v_nodes = [ind[0]], [ind[1]]
    # u_dist, v_dist = [0], [0]
    # u_visited, v_visited = set([ind[0]]), set([ind[1]])
    # u_fringe, v_fringe = set([ind[0]]), set([ind[1]])

    conns = {
        # row-col format / u and v
        0: [ind[0], ind[1], A],
        1: [ind[0], u_features.shape[1], u_features],
        2: [ind[1], v_features.shape[1], v_features],
    }

    for dist in range(1, h+1):
        for i in range()
        fringe_sets[]
        v_fringe, u_fringe = neighbors(
            u_fringe, A, True), neighbors(v_fringe, A, False)
        u_fringe = u_fringe - u_visited
        v_fringe = v_fringe - v_visited
        u_visited = u_visited.union(u_fringe)
        v_visited = v_visited.union(v_fringe)
        if sample_ratio < 1.0:
            u_fringe = random.sample(u_fringe, int(sample_ratio*len(u_fringe)))
            v_fringe = random.sample(v_fringe, int(sample_ratio*len(v_fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                u_fringe = random.sample(u_fringe, max_nodes_per_hop)
            if max_nodes_per_hop < len(v_fringe):
                v_fringe = random.sample(v_fringe, max_nodes_per_hop)
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = u_nodes + list(u_fringe)
        v_nodes = v_nodes + list(v_fringe)
        u_dist = u_dist + [dist] * len(u_fringe)
        v_dist = v_dist + [dist] * len(v_fringe)
    subgraph = A[u_nodes, :][:, v_nodes]
    # remove link between target nodes
    subgraph[0, 0] = 0

    # construct nx graph
    g = nx.Graph()
    g.add_nodes_from(range(len(u_nodes)), bipartite='u')
    g.add_nodes_from(range(len(u_nodes), len(
        u_nodes)+len(v_nodes)), bipartite='v')
    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)
    r = r.astype(int)
    v += len(u_nodes)
    #g.add_weighted_edges_from(zip(u, v, r))
    g.add_edges_from(zip(u, v))

    edge_types = dict(zip(zip(u, v), r-1))  # transform r back to rating label
    nx.set_edge_attributes(g, name='type', values=edge_types)
    # get structural node labels
    node_labels = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]

    # get node features
    if u_features is not None:
        u_features = u_features[u_nodes]
    if v_features is not None:
        v_features = v_features[v_nodes]
    node_features = None
    if False:
        # directly use padded node features
        if u_features is not None and v_features is not None:
            u_extended = np.concatenate(
                [u_features, np.zeros(
                    [u_features.shape[0], v_features.shape[1]])], 1
            )
            v_extended = np.concatenate(
                [np.zeros([v_features.shape[0], u_features.shape[1]]),
                 v_features], 1
            )
            node_features = np.concatenate([u_extended, v_extended], 0)
    if False:
        # use identity features (one-hot encodings of node idxes)
        u_ids = one_hot(u_nodes, A.shape[0]+A.shape[1])
        v_ids = one_hot([x+A.shape[0] for x in v_nodes], A.shape[0]+A.shape[1])
        node_ids = np.concatenate([u_ids, v_ids], 0)
        #node_features = np.concatenate([node_features, node_ids], 1)
        node_features = node_ids
    if True:
        # only output node features for the target user and item
        if u_features is not None and v_features is not None:
            node_features = [u_features[0], v_features[0]]

    return g, node_labels, node_features


def collective_links2subgraphs(datasets, h=1,
                               sample_ratio=1.0,
                               max_nodes_per_hop=None,
                               u_features=None,
                               v_features=None,
                               max_node_label=None,
                               class_values=None,
                               testing=False,
                               parallel=True):
    # extract enclosing subgraphs
    if max_node_label is None:  # if not provided, infer from graphs
        max_n_label = {'max_node_label': 0}

    def helper(A, links, g_labels):
        g_list = []
        if not parallel or max_node_label is None:
            with tqdm(total=len(links[0])) as pbar:
                # TODO: how to make it dynamic?
                for i, j, g_label in zip(links[0], links[1], g_labels):
                    # create labelling graph

                    # i and j are indices
                    g, n_labels, n_features = collective_subgraph_extraction_labeling(
                        (i, j), A, h, sample_ratio, max_nodes_per_hop, u_features,
                        v_features, class_values
                    )

                    #
                    if max_node_label is None:
                        max_n_label['max_node_label'] = max(
                            max(n_labels), max_n_label['max_node_label']
                        )
                        g_list.append((g, g_label, n_labels, n_features))
                    else:
                        g_list.append(nx_to_PyGGraph(
                            g, g_label, n_labels, n_features, max_node_label, class_values
                        ))
                    pbar.update(1)
        else:
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(
                parallel_worker,
                [
                    (g_label, (i, j), A, h, sample_ratio, max_nodes_per_hop, u_features,
                        v_features, class_values)
                    for i, j, g_label in zip(links[0], links[1], g_labels)
                ]
            )
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready():
                    break
                remaining = results._number_left
                time.sleep(1)
            results = results.get()
            pool.close()
            pbar.close()
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
            print("Transforming to pytorch_geometric graphs...".format(end-start))
            g_list += [
                nx_to_PyGGraph(g, g_label, n_labels, n_features,
                               max_node_label, class_values)
                for g_label, g, n_labels, n_features in tqdm(results)
            ]
            del results
            end2 = time.time()
            print("Time eplased for transforming to pytorch_geometric graphs: {}s".format(
                end2-end))
        return g_list

    print('Enclosing subgraph extraction begins...')
    # TODO: I have to do something with datasets

    # train_graphs = helper(A, train_indices, train_labels)
    # if not testing:
    #     val_graphs = helper(A, val_indices, val_labels)
    # else:
    #     val_graphs = []
    # test_graphs = helper(A, test_indices, test_labels)

    # if max_node_label is None:
    #     train_graphs = [
    #         nx_to_PyGGraph(*x, **max_n_label, class_values=class_values) for x in train_graphs
    #     ]
    #     val_graphs = [
    #         nx_to_PyGGraph(*x, **max_n_label, class_values=class_values) for x in val_graphs
    #     ]
    #     test_graphs = [
    #         nx_to_PyGGraph(*x, **max_n_label, class_values=class_values) for x in test_graphs
    #     ]

    # return train_graphs, val_graphs, test_graphs
