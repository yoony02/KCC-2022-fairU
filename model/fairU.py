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


class GCNLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GCNLayer, self).__init__()
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


class GCNModelVAE(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2, dropout, lr):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GCNLayer(input_dim, hidden1, dropout, act=F.relu)
        self.gc2 = GCNLayer(hidden1, hidden2, dropout, act=lambda x: x)
        self.gc3 = GCNLayer(hidden1, hidden2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.optimizer = optim.Adam(self.get_parameters(), lr=lr)

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

    
    
class AdversarialNetwork(nn.Module):
    def __init__(self, hidden, dim_out, sensitive_class):
        super(AdversarialNetwork, self).__init__()
        self.in_feature = hidden
        self.hidden_size = hidden
        self.dim_out = dim_out
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

    

class FairU(nn.Module):

    def __init__(self, input_dim, args, sensitive):
        super(FairU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = args.hidden1
        self.hidden_dim2 = args.hidden2
        self.dropout = args.dropout
        self.lr = args.lr
        self.device = args.device
        self.adv = args.adv
        
        self.alpha = args.alpha
        self.beta = args.beta

        self.sensitive = sensitive
        self.sensitive_class = np.unique(sensitive)
        
        self.gvae = GCNModelVAE(self.input_dim, self.hidden_dim1, self.hidden_dim2, self.dropout, self.lr)
        
        if self.adv:
            if max(self.sensitive_class) == 1:
                self.dim_out = 1
                self.binary = True
            else:
                self.dim_out = len(self.sensitive_class)
                self.binary = False
            self.adv_net = AdversarialNetwork(self.hidden_dim2, self.dim_out, self.sensitive_class)

        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            
        
    def forward(self, feats, adj, edges, train=True):
        recov, z, mu, logvar = self.gvae(feats, adj)
    
        if train:
            link_preds = (z[edges[:, 0]] * z[edges[:, 1]]).sum(dim=-1)
            if self.adv:
                adv_preds = self.adv_net(z)
            else:
                adv_preds = None
            return recov, mu, logvar, link_preds, adv_preds
        else:
            return recov

    
    def loss_function(self, adj_preds, adj_label, for_reconloss, 
                      link_preds, test_links, 
                      adv_preds):
        
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        mu, logvar, n_nodes, norm, pos_weight = for_reconloss
        cost = norm * F.binary_cross_entropy_with_logits(adj_preds, adj_label, pos_weight=pos_weight)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        recon_loss = cost + KLD

        targets = torch.FloatTensor(test_links[:, 2]).to(self.device)
        link_loss = F.binary_cross_entropy_with_logits(link_preds, targets)

        if self.adv:
            if self.binary:
                adv_targets = torch.FloatTensor(self.sensitive.reshape(-1, 1)).to(self.device)
                adv_loss = F.binary_cross_entropy_with_logits(adv_preds, adv_targets)
            else:
                adv_targets = torch.LongTensor(self.sensitive).to(self.device)
                adv_loss = F.cross_entropy(adv_preds, adv_targets)
        else:
            adv_loss = 0

        return recon_loss + (self.alpha * adv_loss) + (self.beta * link_loss) 
