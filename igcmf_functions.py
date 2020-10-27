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

# from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset, Dataset
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


class PairData(Data):
    def __init__(
        self, x1, ui_edge_index, ui_edge_type, x2, ig_edge_index, ig_edge_type, y1, y2
    ):
        super(PairData, self).__init__()
        self.x1 = x1
        self.ui_edge_index = ui_edge_index
        self.ui_edge_type = ui_edge_type
        self.y1 = y1

        self.x2 = x2
        self.ig_edge_index = ig_edge_index
        self.ig_edge_type = ig_edge_type
        self.y2 = y2

    def __inc__(self, key, value):
        if key == "ui_edge_index":
            return self.x1.size(0)
        if key == "ig_edge_index":
            return self.x2.size(0)
        else:
            return super(PairData, self).__inc__(key, value)


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

    # def __getitem__(self, idx):
    #     return self.datasetA[idx], self.datasetB[idx]

    def get(self, idx):
        # rating matrix (user-item)
        i, j = self.links[0][idx], self.links[1][idx]
        score = self.labels[idx]
        nodes, distances = subgraph_extraction(
            (i, j), self.A, self.h, self.sample_ratio, self.max_nodes_per_hop
        )  # i and j indices

        user_item_subgraph = subgraph_labeling(
            nodes,
            distances,
            self.A,
            class_values=self.class_values,
            h=self.h,
            score=score,
        )  # node labeling

        # side matrix index (item features)
        k = random_nonzero(j, self.v_features)  # TODO: could this be an issue?
        nodes, distances = subgraph_extraction(
            (j, k),
            self.v_features,
            self.h,
            self.sample_ratio,
            self.max_nodes_per_hop,
        )

        item_genre_subgraph = subgraph_labeling(
            nodes,
            distances,
            self.v_features,
            class_values=[1.0],  # show that it exists
            h=self.h,
            score=0,  # always be 1
            is_neg_sampling=True,
        )  # node labeling

        subgraphs_dict = {
            "user_item": user_item_subgraph,
            "item_genre": item_genre_subgraph,
        }
        return construct_pyg_graph(subgraphs_dict)

"""
    Method to extract the subgraph from collective matrices
"""
def subgraph_extraction(inds, A, h=1, sample_ratio=1.0, max_nodes_per_hop=None):
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


def negative_sampling_coordinates(A):
    A = ssp.coo_matrix(A, copy=True)
    A.sum_duplicates()
    return A.row, A.col, A.data

"""
    Get negative sample nodes
"""
def get_neg_nodes(u_node, v_nodes, A, neg_ratio=3):
    l = len(v_nodes)
    max_l = A.get_shape()[1]
    neg_l = l * neg_ratio
    neg_nodes = [x for x in range(max_l) if x not in v_nodes]
    res_list = []

    if(neg_l < max_l - l):
        res_list = random.choices(neg_nodes, k=neg_l)
    else:
        res_list = neg_nodes
    u_list = [u_node for x in range(len(res_list))]
    v_list = res_list    
    return u_list, v_list

def find_with_neg_samples(subgraph, ori_nodes, neg_nodes):
    u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)    
    val = float(0)
    r_neg = [val for _ in range(len(neg_nodes))]
    u_neg_ind = [0 for _ in range(len(neg_nodes))]
    v_neg_ind = [x for x in range(1, len(neg_nodes)+1)]
    u = np.append(u,u_neg_ind)
    v = np.append(v,v_neg_ind)
    r = np.append(r,r_neg)
    return u,v,r

def subgraph_labeling(
    nodes, distances, adj_matrix, class_values, h=1, score=1, is_neg_sampling=False, neg_ratio=3
):
    u_nodes, v_nodes = nodes[0], nodes[1]
    u_dist, v_dist = distances[0], distances[1]
    subgraph = adj_matrix[u_nodes, :][:, v_nodes]
    subgraph[0, 0] = 0

    if not is_neg_sampling:
        u, v, r = ssp.find(subgraph)  # r is 1, 2... (rating labels + 1)
    else:
        # NOTE: we assign v_nodes = genre nodes
        u_node = u_nodes[0]
        u_neg_nodes, v_neg_nodes = get_neg_nodes(u_node, v_nodes, adj_matrix, neg_ratio)
        v_dist.extend([1 for x in range(len(u_neg_nodes))]) # new genre nodes are new neighbours
        u, v, r = find_with_neg_samples(subgraph, v_nodes, v_neg_nodes)
       
    # transform r back to rating label
    r = r.astype(int)
    r = r - 1

    v += len(u_nodes)  # to avoid overlapping indices

    # returned variables
    triplet = {"u": u, "v": v, "r": r}
    node_labels = [x * 2 for x in u_dist] + [x * 2 + 1 for x in v_dist]
    max_node_label = 2 * h + 1
    y = class_values[score]  # find actual rating from the expected label

    return triplet, node_labels, max_node_label, y


def construct_pyg_graph(subgraphs):
    user_item = subgraphs["user_item"]
    item_genre = subgraphs["item_genre"]

    def graph_variables(triplet, node_labels, max_node_label, y):
        u, v, r = (triplet["u"], triplet["v"], triplet["r"])
        u, v, r = (
            torch.LongTensor(u),
            torch.LongTensor(v),
            torch.LongTensor(r),
        )
        edge_index = torch.stack([torch.cat([u, v]), torch.cat([v, u])], 0)
        edge_type = torch.cat([r, r])
        x = torch.FloatTensor(one_hot(node_labels, max_node_label + 1))
        y = torch.FloatTensor([y])
        return x, edge_index, edge_type, y

    x1, ui_edge_index, ui_edge_type, y1 = graph_variables(*user_item)
    x2, ig_edge_index, ig_edge_type, y2 = graph_variables(*item_genre)

    data = PairData(
        x1, ui_edge_index, ui_edge_type, x2, ig_edge_index, ig_edge_type, y1=y1, y2=y2
    )
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
