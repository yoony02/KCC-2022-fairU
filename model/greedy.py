###############################################################
# Reference github link : https://github.com/farzmas/FLIP/blob/master/algs/greedy.py
# @modify : chri0220@skku.edu
###############################################################



import numpy as np
import networkx as nx


def greedy_pp(G, sensitive, labels, predicts, thresh, percent):

    preds_g = G.copy()
    preds = np.array(predicts)
    labels_w_edge = np.array(labels)

    preds_w_edge = labels_w_edge.copy()
    preds_w_edge[:, 2] = preds > thresh
    
    # add predict edges 
    for edge in preds_w_edge:
        src, dst, weight = edge
        if weight == 1:
            preds_g.add_edge(src, dst, weight=weight)
    
    preds_adj = nx.convert_matrix.to_numpy_array(preds_g)

    # make ground truth graph 
    gt_adj = np.zeros_like(preds_adj)
    for edge in labels_w_edge:
        src, dst, weight = edge
        if weight == 1:
            gt_adj[src, dst] = 1
            gt_adj[dst, src] = 1

    preds_greedy = good_greedy(preds_adj, gt_adj, sensitive.reshape(-1, 1), percent)

    # import pdb
    # pdb.set_trace()

    # # extract prediction after greedy post processing
    # final_preds = []
    # for edge in labels:
    #     pred = preds_greedy[edge[0], edge[1]]
    #     final_preds.append(pred)
    
    # return np.array(final_preds)
    return preds_greedy


def good_greedy(preds_adj, ground_truth_adj, sensitive, percent):
    # make sure a is a column vector w/ length = number of vertices in graph
    try:
        assert len(sensitive.shape) == 2
        assert sensitive.shape[1] == 1
        assert preds_adj.shape[0] == sensitive.shape[0]
    except:
        import pdb
        pdb.set_trace()
        
    d = np.sum(preds_adj, axis=1, keepdims=True)
    m = np.sum(preds_adj) / 2

    score_pair = np.multiply((d + d.T - 1) / (2 * m) - 1, (sensitive == sensitive.T))
    
    #compute degree sum for all vertices with each protected
    score_other = np.zeros_like(d)
    class_d_sum = {}
    for c in np.unique(sensitive):
        class_d_sum[c] = np.sum(d[sensitive == c])
        score_other[sensitive == c] = class_d_sum[c]
        score_other -= d

    score_other = score_other + score_other.T - np.diag(np.squeeze(score_other))
    score_other = score_other / (2 * m)
    score = score_pair + score_other
    
    score[(1 - preds_adj) > 0] *= -1
    score[(1 - ground_truth_adj) > 0] = 9999999
    
    mod_percent = percent * ground_truth_adj.sum() / (preds_adj.size)

    thresh = np.quantile(score, mod_percent)
    flip_inds = score < thresh
    preds_adj[flip_inds] = 1 - preds_adj[flip_inds]
    
    return preds_adj








