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
import random
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import warnings

warnings.simplefilter("ignore", ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
torch.multiprocessing.set_sharing_strategy("file_system")


class ImportedDataset:
    def __init__(self, variables):
        # TODO: create variables validator

        # features
        self.u_features = variables["u_features"]
        self.v_features = variables["v_features"]
        # matrix
        self.adj_train = variables["adj_train"]

        # training labels & indices
        self.train_labels = variables["train_labels"]
        self.train_u_indices = variables["train_u_indices"]
        self.train_v_indices = variables["train_v_indices"]

        # validation labels & indices
        self.val_labels = variables["val_labels"]
        self.val_u_indices = variables["val_u_indices"]
        self.val_v_indices = variables["val_v_indices"]

        # testing labels & indices
        self.test_labels = variables["test_labels"]
        self.test_u_indices = variables["test_u_indices"]
        self.test_v_indices = variables["test_v_indices"]

        # class_values
        self.class_values = variables["class_values"]


"""
    A method to split side matrix into train, val, and test data.
    Slightly similar to load_monti. The difference is the train, val, and test data are depending on the labels and indices of the relation/main matrix.
"""


def load_side_matrix(
    loaded_data,
    train_labels,
    train_u_indices,
    train_v_indices,
    val_labels,
    val_u_indices,
    val_v_indices,
    test_labels,
    test_u_indices,
    test_v_indices,
    is_testing=False,
    is_cmf=True,
):
    M = loaded_data["M"]
    Otraining = loaded_data["Otraining"]
    Otest = loaded_data["Otest"]
    num_users = loaded_data["num_users"]
    num_items = loaded_data["num_items"]
    u_features = loaded_data["u_features"]
    v_features = loaded_data["v_features"]

    # get labels
    u_nodes_ratings = np.where(M)[0]  # nonzeros label u
    v_nodes_ratings = np.where(M)[1]  # nonzeros label v
    ratings = M[np.where(M)]  # dot product result

    # specify types
    u_nodes_ratings, v_nodes_ratings = (
        u_nodes_ratings.astype(np.int64),
        v_nodes_ratings.astype(np.int32),
    )
    ratings = ratings.astype(np.float64)

    # re-assign
    u_nodes = u_nodes_ratings
    v_nodes = v_nodes_ratings

    print("number of u nodes = ", len(set(u_nodes)))
    print("number of v nodes = ", len(set(v_nodes)))

    return u_nodes


"""
    Method to convert networkx graph to PyGGraph format using pytorch.
"""


def nx_to_PyGGraph(
    g, graph_label, node_labels, node_features, max_node_label, class_values
):
    # convert networkx graph to pytorch_geometric data format
    # TODO: bug for the side matrix/side feature graph
    # Where it should handle
    y = torch.FloatTensor([class_values[graph_label]])
    if len(g.edges()) == 0:
        i, j = [], []
    else:
        i, j = zip(*g.edges())
    edge_index = torch.LongTensor([i + j, j + i])
    edge_type_dict = nx.get_edge_attributes(g, "type")
    edge_type = torch.LongTensor([edge_type_dict[(ii, jj)] for ii, jj in zip(i, j)])
    edge_type = torch.cat([edge_type, edge_type], 0)
    edge_attr = torch.FloatTensor(class_values[edge_type]).unsqueeze(
        1
    )  # continuous ratings, num_edges * 1
    x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))

    data = Data(x, edge_index, edge_attr=edge_attr, y=y)
    data.edge_type = edge_type

    return data


"""
    Method to extract the subgraph from collective matrices
"""


def subgraph_extraction(
    inds, A, h=1, sample_ratio=1.0, max_nodes_per_hop=None, class_values=None
):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    u_index = inds[0]
    i_index = inds[1]

    nodes = [[u_index], [i_index]]
    distances = [[0], [0]]  # O(M)
    visited_sets = [set([u_index]), set([i_index])]  # O(M)
    fringe_sets = [set([u_index]), set([i_index])]  # O(M)
    for dist in range(1, h + 1):  # for now, we only focus in 1-hop
        # get fringe with neighbors(node,A,is_row)
        fringe_sets[1], fringe_sets[0] = neighbors(fringe_sets[0], A, True), neighbors(
            fringe_sets[1], A, False
        )

        # update fringe based on visited sets
        fringe_sets[0] = fringe_sets[0] - visited_sets[0]
        fringe_sets[1] = fringe_sets[1] - visited_sets[1]

        # get visited nodes based on u and v fringes
        visited_sets[0] = visited_sets[0].union(fringe_sets[0])
        visited_sets[1] = visited_sets[1].union(fringe_sets[1])

        # use sample ratio (if defined)
        if sample_ratio < 1.0:
            fringe_sets[0] = random.sample(
                fringe_sets[0], int(sample_ratio * len(fringe_sets[0]))
            )
            fringe_sets[1] = random.sample(
                fringe_sets[1], int(sample_ratio * len(fringe_sets[1]))
            )

        # limiting the number of nodes_per hop (if defined)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe_sets[0]):
                fringe_sets[0] = random.sample(fringe_sets[0], max_nodes_per_hop)
            if max_nodes_per_hop < len(fringe_sets[1]):
                fringe_sets[1] = random.sample(fringe_sets[1], max_nodes_per_hop)

        # stop if there are no fringes on u and v
        if len(fringe_sets[0]) == 0 and len(fringe_sets[1]) == 0:
            break

        # update u and v nodes
        nodes[0] = nodes[0] + list(fringe_sets[0])
        nodes[1] = nodes[1] + list(fringe_sets[1])

        # update u and v distances
        distances[0] = distances[0] + [dist] * len(fringe_sets[0])
        distances[1] = distances[1] + [dist] * len(fringe_sets[1])

    return nodes, distances


def subgraph_labeling(raw_nodes, raw_distances, matrices):
    main_matrix = matrices[0]
    side_matrix = matrices[1]
    u_nodes, v_nodes, w_nodes = raw_nodes[0][0], raw_nodes[0][1], raw_nodes[1][1]
    u_dist, v_dist, w_dist = (
        raw_distances[0][0],
        raw_distances[0][1],
        raw_distances[1][1],
    )
    nodes = [u_nodes, v_nodes, w_nodes]
    distances = [u_dist, v_dist, w_dist]
    # combined_nodes = np.concatenate([u_nodes, v_nodes], axis=None)

    subgraphs = []
    for i in range(1, len(matrices) + 1):
        subgraph = matrices[i - 1][nodes[i - 1], :][:, nodes[i]]
        subgraph[0, 0] = 0
        subgraphs.append(subgraph)

    g = nx.Graph()
    g.add_nodes_from(range(len(u_nodes)), bipartite="u")
    g.add_nodes_from(range(len(u_nodes), len(u_nodes) + len(v_nodes)), bipartite="v")
    g.add_nodes_from(
        range(len(u_nodes) + len(v_nodes), len(u_nodes) + len(v_nodes) + len(w_nodes)),
        bipartite="w",
    )
    u, v, r = ssp.find(subgraphs[0])  # r is 1, 2... (rating labels + 1)
    x, w, rs = ssp.find(subgraphs[1])  # y is 1 (genre exist)
    r = r.astype(int)
    rs = rs.astype(int)

    v += len(u_nodes)
    w += len(u_nodes) + len(v_nodes)
    # g.add_weighted_edges_from(zip(u, v, r))
    g.add_edges_from(zip(u, v))
    g.add_edges_from(zip(v, w))

    edge_types = dict(zip(zip(u, v), r - 1))  # transform r back to rating label
    edge_types.update(zip(zip(v, w), rs))
    nx.set_edge_attributes(g, name="type", values=edge_types)

    # Node-labeling process
    node_labels = []
    for i in range(len(distances)):
        node_labels += [h * len(distances) + i for h in distances[i]]

    return g, node_labels


def random_nonzero(index, matrix):
    tpl = np.nonzero(matrix[index])
    return random.choice(tpl[1])  # because the first index will always be 0


def collective_subgraph_extraction_labeling(
    inds,
    A,
    h=1,
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    u_features=None,
    v_features=None,
    class_values=None,
):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0
    """
    TODO: must make it dynamic later: u indices, i indices,  i_feature cols, u_feature cols. Also, order matters here.
     """

    # randomize the number of item_features column index
    item_index = random_nonzero(inds[1], v_features)
    ind = (inds[0], inds[1], inds[1], np.int64(item_index))

    num_node_types = len(ind)  # M notation
    # O(M) need to add u_feature.shape[1]
    nodes = [[ind[i]] for i in range(num_node_types)]
    distances = [[0] for _ in range(num_node_types)]  # O(M)
    visited_sets = [set([ind[k]]) for k in range(num_node_types)]  # O(M)
    fringe_sets = [set([ind[l]]) for l in range(num_node_types)]  # O(M)

    conns = {
        # connections/ relations: u, v, adj_matrix
        # 0: [ind[1], ind[0], u_features],  # u_features-user
        0: [ind[0], ind[1], A],  # user-item
        2: [ind[2], ind[3], v_features],  # item-i_features
    }
    for i in range(0, len(ind), 2):  # up to 2 loops
        for dist in range(1, h + 1):  # for now, we only focus in 1-hop
            # get fringe with neighbors(node,A,is_row)
            fringe_sets[i + 1], fringe_sets[i] = (
                neighbors(fringe_sets[i], conns[i][2], True),
                neighbors(fringe_sets[i + 1], conns[i][2], False),
            )

            # update fringe based on visited sets
            fringe_sets[i] = fringe_sets[i] - visited_sets[i]
            fringe_sets[i + 1] = fringe_sets[i + 1] - visited_sets[i + 1]

            # get visited nodes based on u and v fringes
            visited_sets[i] = visited_sets[i].union(fringe_sets[i])
            visited_sets[i + 1] = visited_sets[i + 1].union(fringe_sets[i + 1])

            # use sample ratio (if defined)
            if sample_ratio < 1.0:
                fringe_sets[i] = random.sample(
                    fringe_sets[i], int(sample_ratio * len(fringe_sets[i]))
                )
                fringe_sets[i + 1] = random.sample(
                    fringe_sets[i + 1], int(sample_ratio * len(fringe_sets[i + 1]))
                )

            # limiting the number of nodes_per hop (if defined)
            if max_nodes_per_hop is not None:
                if max_nodes_per_hop < len(fringe_sets[i]):
                    fringe_sets[i] = random.sample(fringe_sets[i], max_nodes_per_hop)
                if max_nodes_per_hop < len(fringe_sets[i + 1]):
                    fringe_sets[i + 1] = random.sample(
                        fringe_sets[i + 1], max_nodes_per_hop
                    )

            # stop if there are no fringes on u and v
            if len(fringe_sets[i]) == 0 and len(fringe_sets[i + 1]) == 0:
                break

            # update u and v nodes
            nodes[i] = nodes[i] + list(fringe_sets[i])
            nodes[i + 1] = nodes[i + 1] + list(fringe_sets[i + 1])

            # update u and v distances
            distances[i] = distances[i] + [dist] * len(fringe_sets[i])
            distances[i + 1] = distances[i + 1] + [dist] * len(fringe_sets[i + 1])

    # combine similar nodes
    nodes[2] = nodes[1]
    # ----------------------------------------------

    # number of subraphs == connections dictionary size
    num_of_subgraphs = range(0, len(ind), 2)
    subgraphs = []

    # TODO: should I make subgraphs into dictionary?
    subgraphs = [conns[j][2][nodes[j], :][:, nodes[j + 1]] for j in num_of_subgraphs]

    # g.add_nodes_from(range(len(u_nodes)), bipartite='u')
    # g.add_nodes_from(range(len(u_nodes), len(
    #     u_nodes)+len(v_nodes)), bipartite='v')
    graphs = []
    node_labels = []
    for i in range(len(subgraphs)):
        g = nx.Graph()
        """
            TODO: IGCMF — current problem in here is the library said it is bipartite. How to make it multipartite graph? or can we do better by building multiple bipartite graphs instead?
        """
        # construct nx graph
        subgraphs[i][0, 0] = 0  # remove link between target nodes
        g.add_nodes_from(range(len(nodes[i])))
        g.add_nodes_from(range(len(nodes[i + 1]), len(nodes[i]) + len(nodes[i + 1])))
        u, v, r = ssp.find(subgraphs[i])  # r is 1, 2... (rating labels + 1)
        r = r.astype(int)
        v += len(nodes[i])
        # g.add_weighted_edges_from(zip(u, v, r))
        g.add_edges_from(zip(u, v))
        # transform r back to rating label
        edge_types = dict(zip(zip(u, v), r - 1))
        nx.set_edge_attributes(g, name="type", values=edge_types)
        graphs.append(g)
        node_labels.append(
            [x * 2 for x in distances[i]] + [x * 2 + 1 for x in distances[i + 1]]
        )

    # for i in range(3):
    #     node_labels += [h*3+i for h in distances[i]]

    # get structural node labels with (h * N + 1) formula, where 'h' is number of hop, 'N' is number of node types (4 for now)

    # NOTE: HARDCODED! IGCMF does not require this anymore.
    # get node features
    # if u_features is not None:
    #     u_features = u_features[u_nodes]
    # if v_features is not None:
    #     v_features = v_features[v_nodes]
    # node_features = None
    # NOTE: IGCMF does not require this anymore. (ABLATION)
    # if False:
    #     # directly use padded node features
    #     if u_features is not None and v_features is not None:
    #         u_extended = np.concatenate(
    #             [u_features, np.zeros(
    #                 [u_features.shape[0], v_features.shape[1]])], 1
    #         )
    #         v_extended = np.concatenate(
    #             [np.zeros([v_features.shape[0], u_features.shape[1]]),
    #                 v_features], 1
    #         )
    #         node_features = np.concatenate([u_extended, v_extended], 0)
    # NOTE: IGCMF does not require this anymore. (ABLATION)
    # if False:
    #     # use identity features (one-hot encodings of node idxes)
    #     u_ids = one_hot(u_nodes, A.shape[0]+A.shape[1])
    #     v_ids = one_hot([x+A.shape[0]
    #                      for x in v_nodes], A.shape[0]+A.shape[1])
    #     node_ids = np.concatenate([u_ids, v_ids], 0)
    #     # node_features = np.concatenate([node_features, node_ids], 1)
    #     node_features = node_ids
    # NOTE: IGCMF does not require this anymore.(ABLATION)
    # if True:
    #     # only output node features for the target user and item
    #     if u_features is not None and v_features is not None:
    #         node_features = [u_features[0], v_features[0]]
    node_features = None

    return graphs, node_labels, node_features


def collective_links2subgraphs(
    datasets,
    h=1,
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    u_features=None,
    v_features=None,
    max_node_label=None,
    class_values=None,
    testing=False,
    parallel=False,
    is_debug=False,
):  # to debug/reduce all size

    # TODO: HARDCODED =========
    main_obj = datasets[0]
    A = main_obj.adj_train
    class_values = main_obj.class_values
    if is_debug:  # use a small number of data to debug
        num_data = 1000
        main_obj.train_u_indices, main_obj.train_v_indices = (
            main_obj.train_u_indices[:num_data],
            main_obj.train_v_indices[:num_data],
        )
        main_obj.val_u_indices, main_obj.val_v_indices = (
            main_obj.val_u_indices[:num_data],
            main_obj.val_v_indices[:num_data],
        )
        main_obj.test_u_indices, main_obj.test_v_indices = (
            main_obj.test_u_indices[:num_data],
            main_obj.test_v_indices[:num_data],
        )
    train_indices = (main_obj.train_u_indices, main_obj.train_v_indices)
    val_indices = (main_obj.val_u_indices, main_obj.val_v_indices)
    test_indices = (main_obj.test_u_indices, main_obj.test_v_indices)
    train_labels = main_obj.train_labels
    val_labels = main_obj.val_labels
    test_labels = main_obj.test_labels
    u_features = main_obj.u_features
    v_features = main_obj.v_features

    # ------------------------

    # extract enclosing subgraphs
    if max_node_label is None:  # if not provided, infer from graphs
        max_n_label = {"max_node_label": 0}

    # Set max node label (2 matrix, 3 relations, 1 hop)
    max_node_label = h * (2 * 3)

    def helper(A, links, g_labels):
        g_list = []
        if not parallel or max_node_label is None:
            with tqdm(total=len(links[0])) as pbar:
                # TODO: how to make it dynamic?
                for i, j, g_label in zip(links[0], links[1], g_labels):
                    # create labelling graph
                    n_features = [u_features[0], v_features[0]]
                    tpl_list = []
                    # i and j are indices
                    nodes, distances = subgraph_extraction(
                        (i, j), A, h, sample_ratio, max_nodes_per_hop
                    )
                    tpl_list.append((nodes, distances))
                    # side matrix index
                    k = random_nonzero(j, v_features)
                    nodes, distances = subgraph_extraction(
                        (j, k), v_features, h, sample_ratio, max_nodes_per_hop
                    )
                    tpl_list.append((nodes, distances))

                    nodes = [tpl[0] for tpl in tpl_list]
                    distances = [tpl[1] for tpl in tpl_list]
                    matrices = [A, v_features]  # main, side
                    g, n_label = subgraph_labeling(nodes, distances, matrices)
                    # gs, n_labels, n_features = collective_subgraph_extraction_labeling(
                    #     (i, j), A, h, sample_ratio, max_nodes_per_hop, u_features,
                    #     v_features, class_values
                    # )

                    #
                    if max_node_label is None:
                        max_n_label["max_node_label"] = max(
                            max(n_labels), max_n_label["max_node_label"]
                        )
                        # NOTE: loop each bipartite graph
                        g_list.append((g, g_label, n_label, n_features))
                    else:
                        # NOTE: loop each bipartite graph
                        g_list.append(
                            nx_to_PyGGraph(
                                g,
                                g_label,
                                n_label,
                                n_features,
                                max_node_label,
                                class_values,
                            )
                        )

                    pbar.update(1)
        else:
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(
                cmf_parallel_worker,
                [
                    (
                        g_label,
                        (i, j),
                        A,
                        h,
                        sample_ratio,
                        max_nodes_per_hop,
                        u_features,
                        v_features,
                        class_values,
                    )
                    for i, j, g_label in zip(links[0], links[1], g_labels)
                ],
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
            print("Time eplased for subgraph extraction: {}s".format(end - start))
            print("Transforming to pytorch_geometric graphs...".format(end - start))
            g_list += [
                nx_to_PyGGraph(
                    g, g_label, n_label, n_features, max_node_label, class_values
                )
                for g_label, gs, n_labels, n_features in tqdm(results)
            ]
            del results
            end2 = time.time()
            print(
                "Time eplased for transforming to pytorch_geometric graphs: {}s".format(
                    end2 - end
                )
            )
        return g_list

    print("Enclosing subgraph extraction begins...")

    train_graphs = helper(A, train_indices, train_labels)
    if not testing:
        val_graphs = helper(A, val_indices, val_labels)
    else:
        val_graphs = []
    test_graphs = helper(A, test_indices, test_labels)

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

    return train_graphs, val_graphs, test_graphs
    # return train_graphs


def cmf_parallel_worker(
    g_label,
    ind,
    A,
    h=1,
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    u_features=None,
    v_features=None,
    class_values=None,
):
    gs, node_labels, node_features = collective_subgraph_extraction_labeling(
        ind, A, h, sample_ratio, max_nodes_per_hop, u_features, v_features, class_values
    )

    return g_label, gs, node_labels, node_features