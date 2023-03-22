###############################################################
# @Author  : Suhyun Yoon, Yonghoon Kang
# @Contact : artfjqm2@skku.edu , evegreen96@skku.edu
#############################################################

import numpy as np
import networkx as nx
import pandas as pd
import os
import csv
import json
from tqdm import tqdm

import pickle as pkl
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple
import copy


def get_key(dict_1, value):
    return [k for k, v in dict_1.items() if v == value]

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def build_train_test(G, n_edges, train_fold=10, ratio=0.1):

    org_adj = nx.to_numpy_matrix(G)
    edges = np.array(G.edges)
    num_test = int(np.floor(n_edges * ratio))

    # true links
    all_true_edges_idx = list(range(n_edges))
    np.random.shuffle(all_true_edges_idx)
    all_true_edges = edges[all_true_edges_idx]
    test_true_edges = all_true_edges[-num_test:]
    train_true_edges = all_true_edges[:-num_test]

    train_true_edges_split = np.array_split(np.array(train_true_edges), train_fold)

    test_G = G.copy()
    test_G.remove_edges_from(test_true_edges)

    # train_Gs, train_adjs = [], []
    # for i in range(train_fold):
    #     temp = G.copy()
    #     train_Gs.append(temp.remove_edges_from(train_true_edges[i]))

    # adj = nx.adjacency_matrix(new_G, nodelist=sorted(org_G.nodes()))

    # false links
    zero_elems = np.where(org_adj == 0)
    zero_edges = np.concatenate([zero_elems[0].reshape(-1, 1), zero_elems[1].reshape(-1, 1)], axis=1)
    all_false_edges = zero_edges[zero_edges[:, 0] < zero_edges[:, 1]]
    
    all_false_edges_idxs = list(range(all_false_edges.shape[0]))
    np.random.shuffle(all_false_edges_idxs)
    all_false_edges = all_false_edges[all_false_edges_idxs]

    test_false_edges = all_false_edges[-num_test:]

    train_false_edges = all_false_edges[:len(train_true_edges)]
    train_false_edges_split = np.array_split(np.array(train_false_edges), train_fold)


    print(f"# train true edges : {len(train_true_edges_split[0])}\t\t# train false edges : {len(train_false_edges_split[0])}")
    print(f"# test true edges : {len(test_true_edges)}\t\t# test false edges : {len(test_false_edges)}")
    # return train_Gs, train_true_edges_split, train_false_edges_split, \
    #        test_G, test_true_edges, test_false_edges

    return test_G, train_true_edges_split, train_false_edges_split, test_true_edges, test_false_edges


def get_facebook_features(ego_nodes, dataset_directory):
    if dataset_directory[-1] != '/':
        dataset_directory = dataset_directory + '/'

    features = {}
    gender_featnum = []
    for ego_node in ego_nodes:
        featnames = open(dataset_directory + str(ego_node) + '.featnames')
        features_node = []
        for line in featnames:
            features_node.append(int(line.split(';')[-1].split()[-1]))
            if line.split(';')[0].split()[1] == 'gender':
                if int(line.split(';')[-1].split()[-1]) not in gender_featnum:
                    gender_featnum.append(int(line.split(';')[-1].split()[-1]))
        features[ego_node] = features_node
        featnames.close()
    gender_featnum.sort()
    return features, gender_featnum

def load_facebook_dataset(data_path, scale=True, test_ratio=0.1):
    # Read edges & construct whole graph
    edge = os.path.join(data_path, 'facebook_combined.txt')
    edges = []
    with open(edge) as f:
        print('file opened')
        for i, line in enumerate(f):
            words = line.split()
            edges.append((int(words[0]), int(words[1])))
        print('Reading edges finished')

    G = nx.Graph(edges)
    nodes = ['0', '107', '348', '414', '686', '698', '1684', '1912', '3437', '3980']
    # get gender feature index
    features_idx, gender_featnum = get_facebook_features(nodes, os.path.join(data_path, 'facebook'))
    # numpy features, # Nodes * # Features
    X = np.zeros((len(G.nodes()), max(map(max, features_idx.values()))+1))
    # read all features(targets)
    target = dict()
    for n in nodes:
        gender_idx = [features_idx[n].index(gender_featnum[0]),
                      features_idx[n].index(gender_featnum[1])]
        # read all targets
        feat = os.path.join(data_path, 'facebook', f'{n}.feat')
        with open(feat) as f:
            for i, line in enumerate(f):
                feats = line.split()
                target[int(feats[0])] = feats[1:]
                X[int(feats[0])][features_idx[n]] = np.array(feats[1:], dtype=np.float64)
        # read ego targets
        egofeat = os.path.join(data_path, 'facebook', f'{n}.egofeat')
        with open(egofeat) as f:
            for i, line in enumerate(f):
                feats = line.split()
                target[int(n)] = feats
                X[int(n)][features_idx[n]] = np.array(feats, dtype=np.float64)
    # sensitive
    sensitive = X[:, gender_featnum[-1]]
    # communities = [set(np.where(sens == sensitive)[0].tolist()) for sens in np.unique(sensitive)]
    # nodelist and data split
    # nodelist = {idx: node for idx, node in enumerate(target.keys())}

    # updated sensitive by splitted G
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"# nodes : {n_nodes}\t # edges : {n_edges}")

    train_G, train_true_edges_split, train_false_edges_split, test_true_edges, test_false_edges = build_train_test(G, n_edges)
    adj = nx.adjacency_matrix(train_G, nodelist=sorted(G.nodes()))


    return train_G, adj, X, sensitive, train_true_edges_split, train_false_edges_split, test_true_edges, test_false_edges



def load_cora_dataset(data_path, scale=True, test_ratio=0.1):

    cora_label = {"Genetic_Algorithms": 0,
                  "Reinforcement_Learning": 1,
                  "Neural_Networks": 2,
                  "Rule_Learning": 3,
                  "Case_Based": 4,
                  "Theory": 5,
                  "Probabilistic_Methods": 6
                  }
    feat_path = os.path.join(data_path, "cora.content")
    edge_path = os.path.join(data_path, "cora.cites")

    idx_features_labels = np.genfromtxt(feat_path, dtype=np.dtype(str))
    idx_features_labels = idx_features_labels[idx_features_labels[:, 0].astype(np.int32).argsort()]

    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    nodelist = {node: idx for idx, node in enumerate(idx)}
    X = np.array(idx_features_labels[:, 1:-1], dtype=np.float32)
    sensitive = np.array(list(map(cora_label.get, idx_features_labels[:, -1])))

    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    # edges 
    edges = np.genfromtxt(edge_path, dtype=np.dtype(int))
    src = np.array(list(map(nodelist.get, edges[:, 0])))
    dst = np.array(list(map(nodelist.get, edges[:, 1])))
    new_edges = np.concatenate([src.reshape(-1, 1), dst.reshape(-1, 1)], axis=1)

    # construct graph
    G = nx.Graph() 
    G.add_edges_from(new_edges, weight=1)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"# nodes : {n_nodes}\t # edges : {n_edges}")

    # org_G = copy.deepcopy(G)
    # new_G, test_true_edges, test_false_edges = build_test(org_G, n_nodes, n_edges, new_edges)

    
    train_G, train_true_edges_split, train_false_edges_split, test_true_edges, test_false_edges = build_train_test(G, n_edges)
    adj = nx.adjacency_matrix(train_G, nodelist=sorted(G.nodes()))


    return train_G, adj, X, sensitive, train_true_edges_split, train_false_edges_split, test_true_edges, test_false_edges


def load_citeseer_dataset(data_path, scale=True, test_ratio=0.1):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []

    for i in range(len(names)):
        with open(os.path.join(data_path, "ind.citeseer.{}".format(names[i])), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file(os.path.join(data_path, "ind.citeseer.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    tx_extended[test_idx_range - min(test_idx_range), :] = tx
    tx = tx_extended
    ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    ty_extended[test_idx_range - min(test_idx_range), :] = ty
    ty = ty_extended

    X = sp.vstack((allx, tx)).tolil()
    X[test_idx_reorder, :] = X[test_idx_range, :]
    X = X.toarray()
    onehot_labels = np.vstack((ally, ty))
    onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]

    sensitive = np.argmax(onehot_labels, 1)

    G = nx.from_dict_of_lists(graph)
    G = nx.convert_node_labels_to_integers(G)

    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

    # nodes = sorted(G.nodes())
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"# nodes : {n_nodes}\t # edges : {n_edges}")
    
    train_G, train_true_edges_split, train_false_edges_split, test_true_edges, test_false_edges = build_train_test(G, n_edges)
    adj = nx.adjacency_matrix(train_G, nodelist=sorted(G.nodes()))


    return train_G, adj, X, sensitive, train_true_edges_split, train_false_edges_split, test_true_edges, test_false_edges



# Directory path which contains dataset directory(ex. data_path/Facebook/)
def load_dataset(dataname, data_path='./dataset'):
    if dataname == 'facebook':
        return load_facebook_dataset(os.path.join(data_path, 'Facebook'))
    elif dataname == 'cora':
        return load_cora_dataset(os.path.join(data_path, 'Cora'))
    elif dataname == 'citeseer':
        return load_citeseer_dataset(os.path.join(data_path, 'Citeseer'))
    else:
        print("No datasets...")






# num_test = int(np.floor(n_edges * ratio))

#     # generate false links for testing

#     test_false_edges = []
#     while len(test_false_edges) < num_test:
#         idx_u = np.random.randint(0, n_nodes - 1)
#         idx_v = np.random.randint(0, n_nodes - 1)

#         if idx_u == idx_v:
#             pass
#         elif (idx_u, idx_v) in G.edges(idx_u):
#             pass
#         elif [idx_u, idx_v, 0] in test_false_edges:
#             pass
#         else:
#             test_false_edges.append([idx_u, idx_v, 0])

#     # generate true links for testing
#     test_true_edges = []
#     all_edges_idx = list(range(n_edges))
#     np.random.shuffle(all_edges_idx)
#     test_true_edges_idx = all_edges_idx[:num_test]
#     for test_idx in test_true_edges_idx:
#         u, v = edges[test_idx]
#         try:
#             G.remove_edge(u, v)
#             test_true_edges.append([u, v, 1])
#         except:
#             pass

#     print(f"# test true edges : {len(test_true_edges)}\t\t# test false edges : {len(test_false_edges)}")
#     return G, test_true_edges, test_false_edges
