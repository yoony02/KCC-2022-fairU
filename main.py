###############################################################
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# reference github link : https://github.com/brandeis-machine-learning/FairAdj
#
# @modify : artfjqm2@skku.edu, evegreen96@skku.edu, maya0707@skku.edu
# add function 
# 1. dataloader 
# 2. fairdrop
# 3. greedy
# 4. evaluation 
#############################################################


import torch
import numpy as np
import random
import networkx as nx

from args import parse_args
from dataloader import load_dataset
from model.fairdrop import *
# from model.gae import GCNModelVAE, loss_function, preprocess_graph
from model.fairU import FairU, preprocess_graph
from model.greedy import greedy_pp
from evals import get_scores, result_print
import torch.nn.functional as F


def main(args):
    G, adj, features, sensitive, train_true_edges_split, train_false_edges_split, test_true_edges, test_false_edges = load_dataset(args.dataset, 'data')
    n_nodes, feat_dim = features.shape
    features = torch.FloatTensor(features).to(args.device)
    sensitive_save = sensitive.copy()

    model = FairU(feat_dim, args, sensitive) #.to(args.device)
    model = model.to(args.device)
    
    print('start train' + '-'*50)
    model.train()
    for epoch in range(args.n_epochs):

        # drop edges fairly
        for fold in range(len(train_true_edges_split)):
            train_true_edges, train_false_edges = train_true_edges_split[fold], train_false_edges_split[fold]
            train_G = G.copy()
            train_G.remove_edges_from(train_true_edges)
            train_adj = nx.adjacency_matrix(train_G, nodelist=sorted(G.nodes()))

            train_true_edges = np.concatenate([train_true_edges, np.ones(train_true_edges.shape[0]).reshape(-1, 1)], axis=1).astype(int)
            train_false_edges = np.concatenate([train_false_edges, np.zeros(train_false_edges.shape[0]).reshape(-1, 1)], axis=1).astype(int)
            train_edges = np.concatenate([train_true_edges, train_false_edges])

            fairdrop = fairdropper(train_G, train_adj.copy(), sensitive, args.n_epochs, args.device)
            fairdrop.build_drop_map()

            if args.fairdrop:
                if epoch % args.fairdrop_term == 0:
                    adj_norm, adj_label, pos_weight, norm = fairdrop.drop_fairly(epoch)
            else:
                adj, adj_norm, adj_label = preprocess_graph(train_adj)
                adj_norm = adj_norm.to(args.device)
                adj_label = adj_label.to(args.device)
                pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
                pos_weight = torch.Tensor([pos_weight]).to(args.device)
                norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)       

            model.optimizer.zero_grad()
            adj_preds, mu, logvar, link_preds, adv_preds = model(features, adj_norm, train_edges)
                
            for_reconloss = [mu, logvar, n_nodes, norm, pos_weight]
            loss = model.loss_function(adj_preds, adj_label, for_reconloss, link_preds, train_edges, adv_preds)
            loss.backward()
            cur_loss = loss.item()
            model.optimizer.step()

        if (epoch+1) % 50 == 0:
            print(f"Epoch: [{epoch+1:d} / {args.n_epochs}]; Loss: {cur_loss:.3f};")


    model.eval()
    with torch.no_grad():
        _, adj_norm, _ = preprocess_graph(adj)
        adj_norm = adj_norm.to(args.device)
        z = model(features, adj_norm, None, train=False)	

    hidden_emb = z.data.cpu().numpy()
    preds = np.array(np.dot(hidden_emb, hidden_emb.T), dtype=np.float128)
    
    thresh = np.median(preds)
    if args.greedy:
        test_true_edges = np.concatenate([test_true_edges, np.ones(test_true_edges.shape[0]).reshape(-1, 1)], axis=1).astype(int)
        test_false_edges = np.concatenate([test_false_edges, np.zeros(test_false_edges.shape[0]).reshape(-1, 1)], axis=1).astype(int)
        test_edges = np.concatenate([test_true_edges, test_false_edges])

        pred_temp = [preds[i, j] for i , j, _ in test_edges]
        new_preds = greedy_pp(G, sensitive, test_edges, pred_temp, thresh, args.greedy_change_pct)
        scores = get_scores(test_true_edges, test_false_edges, new_preds, G, sensitive)
    else:
        scores = get_scores(test_true_edges, test_false_edges, preds, G, sensitive)
    result_print(scores)


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    args.device = torch.device(args.device)
    fix_seed(args.seed)
    main(args)
