from tqdm import tqdm
from util_functions import neighbors, one_hot
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


class MyDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        A,
        links,
        labels,
        h,
        sample_ratio,
        max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        max_num=None,
        parallel=True,
    ):
        self.A = A
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.class_values = class_values
        self.parallel = parallel
        self.max_num = max_num
        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]
        super(MyDataset, self).__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        name = "data.pt"
        if self.max_num is not None:
            name = "data_{}.pt".format(self.max_num)
        return [name]

    def process(self):

        # Extract enclosing subgraphs and save to disk
        data_list = collective_links2subgraphs(
            self.A,
            self.links,
            self.labels,
            self.h,
            self.sample_ratio,
            self.max_nodes_per_hop,
            self.u_features,
            self.v_features,
            self.class_values,
            self.parallel,
        )
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        del data_list


class MyDynamicDataset(Dataset):
    def __init__(
        self,
        root,
        A,
        links,
        labels,
        h,
        sample_ratio,
        max_nodes_per_hop,
        u_features,
        v_features,
        class_values,
        max_num=None,
    ):
        super(MyDynamicDataset, self).__init__(root)
        self.A = A
        self.links = links
        self.labels = labels
        self.h = h
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.u_features = u_features
        self.v_features = v_features
        self.class_values = class_values
        if max_num is not None:
            np.random.seed(123)
            num_links = len(links[0])
            perm = np.random.permutation(num_links)
            perm = perm[:max_num]
            self.links = (links[0][perm], links[1][perm])
            self.labels = labels[perm]

    def __len__(self):
        return len(self.links[0])

    def get(self, idx):
        # node_features = [self.u_features[0], self.v_features[0]]
        nodes_distances_tpl = []
        i, j = self.links[0][idx], self.links[1][idx]
        g_label = self.labels[idx]

        # i and j indices
        nodes, distances = subgraph_extraction(
            (i, j), self.A, self.h, self.sample_ratio, self.max_nodes_per_hop
        )
        nodes_distances_tpl.append((nodes, distances))

        # side matrix index (item features)
        k = random_nonzero(j, self.v_features)
        nodes, distances = subgraph_extraction(
            (j, k),
            self.v_features,
            self.h,
            self.sample_ratio,
            self.max_nodes_per_hop,
            g_label,
        )
        nodes_distances_tpl.append((nodes, distances))
        nodes = [nd[0] for nd in nodes_distances_tpl]
        distances = [nd[1] for nd in nodes_distances_tpl]
        matrices = [self.A, self.v_features]  # main, side

        # node labeling
        tmp = subgraph_labeling(
            nodes,
            distances,
            matrices,
            class_values=self.class_values,
            h=self.h,
            g_label=g_label,
        )

        return construct_pyg_graph(*tmp)


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


def subgraph_labeling(raw_nodes, raw_distances, matrices, class_values, h=1, g_label=1):
    # TODO: make it dynamic by handling multiple matrices later
    main_matrix = matrices[0]
    side_matrix = matrices[1]
    y_genre = 0  # genre is always because it was selected to be 1 before.
    u_nodes, v_nodes, w_nodes = raw_nodes[0][0], raw_nodes[0][1], raw_nodes[1][1]
    u_dist, v_dist, w_dist = (
        raw_distances[0][0],
        raw_distances[0][1],
        raw_distances[1][1],
    )
    nodes = [u_nodes, v_nodes, w_nodes]
    distances = [u_dist, v_dist, w_dist]

    subgraphs = []
    for i in range(1, len(matrices) + 1):
        subgraph = matrices[i - 1][nodes[i - 1], :][:, nodes[i]]
        subgraph[0, 0] = 0
        subgraphs.append(subgraph)

    u, v, r = ssp.find(subgraphs[0])  # r is 1, 2... (rating labels + 1)
    item_side_ids, genre_ids, genre_values = ssp.find(
        subgraphs[1]
    )  # y is 1 (genre exist)
    r = r.astype(int)
    genre_values = genre_values.astype(int)

    v += len(u_nodes)  # starting point
    item_side_ids += len(u_nodes)
    genre_ids += len(u_nodes) + len(v_nodes)  # starting point

    y = class_values[g_label]

    r = r - 1  # transform r back to rating label
    genre_values = genre_values  # transform rating side back to original label

    # Node-labeling process
    node_labels = []
    for i in range(len(distances)):
        node_labels += [h * len(distances) + i for h in distances[i]]

    # Set max node label (2 matrix, 3 relations, 1 hop)
    max_node_label = h * (len(matrices) * len(distances))
    indices = {
        "user_ids": u,
        "item_ids": v,
        "genre_ids": genre_ids,
        "item_side_ids": item_side_ids,
    }
    scores = {"r": r, "g": genre_values}

    return indices, scores, node_labels, max_node_label, y


def construct_pyg_graph(indices, scores, node_labels, max_node_label, y):
    u, v, w, z = (
        indices["user_ids"],
        indices["item_ids"],
        indices["genre_ids"],
        indices["item_side_ids"],
    )
    r, g = scores["r"], scores["g"]
    u, v, w, z = (
        torch.LongTensor(u),
        torch.LongTensor(v),
        torch.LongTensor(w),
        torch.LongTensor(z),
    )
    r, g = torch.LongTensor(r), torch.LongTensor(g)
    edge_index = torch.stack([torch.cat([u, v, z, w]), torch.cat([v, u, w, z])], 0)
    edge_type = torch.cat([r, r, g, g])
    x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
    y = torch.FloatTensor([y])
    data = Data(x, edge_index, edge_type=edge_type, y=y)
    return data


def random_nonzero(index, matrix):
    tpl = np.nonzero(matrix[index])
    return random.choice(tpl[1])  # because the first index will always be 0


def collective_links2subgraphs(
    A,
    links,
    labels,
    h=1,
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    u_features=None,
    v_features=None,
    class_values=None,
    parallel=True,
):
    # extract enclosing subgraphs
    print("Enclosing subgraph extraction begins...")

    # Set max node label (2 matrix, 3 relations, 1 hop)
    max_node_label = h * (2 * 3)
    g_list = []
    # create labelling graph
    if not parallel or max_node_label is None:
        with tqdm(total=len(links[0])) as pbar:
            for i, j, g_label in zip(links[0], links[1], labels):
                tmp = subgraph_extraction_labeling(
                    (i, j),
                    A,
                    h,
                    sample_ratio,
                    max_nodes_per_hop,
                    u_features,
                    v_features,
                    class_values,
                    g_label,
                )
                data = construct_pyg_graph(*tmp)
                g_list.append(data)
                pbar.update(1)
    else:
        start = time.time()
        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap_async(
            subgraph_extraction_labeling,
            [
                (
                    (i, j),
                    A,
                    h,
                    sample_ratio,
                    max_nodes_per_hop,
                    u_features,
                    v_features,
                    class_values,
                    g_label,
                )
                for i, j, g_label in zip(links[0], links[1], labels)
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
        print("Transforming to pytorch_geometric graphs... {}s".format(end - start))
        g_list = []
        while results:
            tmp = results.pop()
            g_list.append(construct_pyg_graph(*tmp))
            pbar.update(1)
        pbar.close()
        end2 = time.time()
        print(
            "Time eplased for transforming to pytorch_geometric graphs: {}s".format(
                end2 - end
            )
        )
    return g_list


def subgraph_extraction_labeling(
    ind,
    A,
    h=1,
    sample_ratio=1.0,
    max_nodes_per_hop=None,
    u_features=None,
    v_features=None,
    class_values=None,
    g_label=1,
):
    node_features = [u_features[0], v_features[0]]
    nodes_distances_tpl = []
    i, j = ind[0], ind[1]

    # i and j indices
    nodes, distances = subgraph_extraction(
        (i, j), A, h, sample_ratio, max_nodes_per_hop
    )
    nodes_distances_tpl.append((nodes, distances))

    # side matrix index
    k = random_nonzero(j, v_features)
    nodes, distances = subgraph_extraction(
        (j, k), v_features, h, sample_ratio, max_nodes_per_hop
    )
    nodes_distances_tpl.append((nodes, distances))

    nodes = [nd[0] for nd in nodes_distances_tpl]
    distances = [nd[1] for nd in nodes_distances_tpl]
    matrices = [A, v_features]  # main, side

    # node labeling
    res = subgraph_labeling(nodes, distances, matrices, class_values, h, g_label)
    # data = construct_pyg_graph(*res)
    return res
