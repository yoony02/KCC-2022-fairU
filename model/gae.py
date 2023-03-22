import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.loss
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import numpy as np
import scipy.sparse as sp



def grl_hook():
    def fun1(grad):
        return -1 * grad.clone()

    return fun1



def preprocess_graph(adj):
    """ D^(-1) * A """

    new_adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    adj_label = torch.FloatTensor(adj.toarray())
    
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))

    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -1.).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_).tocoo()

    adj_norm = sparse_mx_to_torch_sparse_tensor(adj_normalized)

    return adj, adj_norm, torch.FloatTensor(adj_label)
    
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor """

    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = self.gc2(x, adj)
        return self.dc(x), x

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_adv": 1.}]



class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, args):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, args.hidden1, args.dropout, act=F.relu)
        self.gc2 = GraphConvolution(args.hidden1, args.hidden2, args.dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(args.hidden1, args.hidden2, args.dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(args.dropout, act=lambda x: x)
        self.optimizer = optim.Adam(self.get_parameters(), lr=args.lr)

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_adv": 1.}]

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)
        # return hidden1, hidden1

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), z, mu, logvar
    
    def decode(self, z, pos_edges, neg_edges):
        edges = torch.cat([torch.LongTensor(pos_edges), torch.LongTensor(neg_edges)], dim=0)
        logits = (z[edges[:, 0]] * z[edges[:, 1]]).sum(dim=-1)
        targets = torch.cat([torch.ones(len(pos_edges)), torch.zeros(len(neg_edges))])
        return logits, targets 



class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class AdversarialNetwork(nn.Module):
    def __init__(self, args, sensitive):
        super(AdversarialNetwork, self).__init__()
        self.in_feature = args.hidden2
        self.hidden_size = args.hidden2
        if max(np.unique(sensitive)) == 1:
            self.dim_out = 1
            self.binary = True
        else:
            self.dim_out = len(np.unique(sensitive))
            self.binary = False
        
        self.lr = args.lr_adv

        self.ad_layer1 = nn.Linear(self.in_feature, self.hidden_size)
        self.ad_layer2 = nn.Linear(self.hidden_size, self.dim_out)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.1)
    
    def get_parameters(self):
        return [{"params": self.parameters(), "lr_adv": self.lr}]

    def forward(self, x):
        x = x * 1.0
        x.register_hook(grl_hook())
        x = self.ad_layer1(x)
        x = self.relu(x)
        x = self.ad_layer2(x)
        return x

    def loss_function(self, preds, targets, pos_weight=None):
        if self.binary:
            targets = targets.reshape(-1, 1)
            if pos_weight:
                return F.binary_cross_entropy_with_logits(preds, targets, pos_weight=pos_weight)
            else:
                return F.binary_cross_entropy_with_logits(preds, targets)
        else:
            targets = targets.long()
            return F.cross_entropy(preds, targets)


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):
    cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

    return cost + KLD


