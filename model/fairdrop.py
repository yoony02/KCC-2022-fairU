###############################################################
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# reference github link : https://github.com/brandeis-machine-learning/FairAdj
#
# @modify : chri0220@skku.edu
# add function fairdropper
#############################################################


import numpy as np
import torch
from model.gae import preprocess_graph


class fairdropper():
    def __init__(self, G, adj, sensitive, n_epochs, device, delta=0.16):
        self.delta = delta
        self.device = device
        self.n_epochs = n_epochs
        self.G = G
        self.adj = adj
        self.sensitive = sensitive

    def build_drop_map(self):
        edges = np.array(list(self.G.edges()))
        src = edges[:, 0]
        dst = edges[:, 1]

        sensitive = torch.LongTensor(self.sensitive)
        sens_diff = (sensitive[src] != sensitive[dst])
        randomization = torch.FloatTensor(self.n_epochs, sens_diff.size(0)).uniform_() < 0.5 + self.delta
        self.sens_diff = sens_diff
        self.randomization = randomization 

    def drop_fairly(self, epoch):
        # G_new = self.G.copy()
        adj_new = self.adj.copy()

        keep = torch.where(self.randomization[epoch], self.sens_diff, ~self.sens_diff)
        remove_edges = np.array(list(self.G.edges()))[keep]
        remove_edges = [(edge[0], edge[1]) for edge in remove_edges]

        
        # G_new.remove_edges_from(remove_edges)
        for edge in remove_edges:
            adj_new[edge[0], edge[1]] = 0


        adj, adj_norm, adj_label = preprocess_graph(adj_new)
        adj_norm = adj_norm.to(self.device)
        adj_label = adj_label.to(self.device)
        
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        pos_weight = torch.Tensor([pos_weight]).to(self.device)
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # return G_new, adj_new.to_sparse().to(self.device)
        return adj_norm, adj_label, pos_weight, norm