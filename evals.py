##########################################################################
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
#
# @modify : chri0220@skku.edu, ghkim10202@skku.edu, maya0707@skku.edu
# add fairness scores 
##########################################################################

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score

import numpy as np
from scipy import stats
import networkx as nx


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_scores(test_true_edges, test_false_edges, preds, G, sensitive, threshold=0.5):
    
    
    if G.is_directed():
        G = G.to_undirected()
    G_ground = G.copy()
    G_new = G.copy()

    preds_pos_intra = []
    preds_pos_inter = []

    for e in test_true_edges:
        G_ground.add_edge(e[0], e[1], weight=1)
        if preds[e[0], e[1]] > threshold:
            G_new.add_edge(e[0], e[1], weight=1)

        if sensitive[e[0]] == sensitive[e[1]]:
            preds_pos_intra.append(sigmoid(preds[e[0], e[1]]))
        else:
            preds_pos_inter.append(sigmoid(preds[e[0], e[1]]))
        # if sensitive[e[0]] == sensitive[e[1]]:
        #     preds_pos_intra.append(preds[e[0], e[1]])
        # else:
        #     preds_pos_inter.append(preds[e[0], e[1]])


    preds_neg_intra = []
    preds_neg_inter = []
    for e in test_false_edges:
        if preds[e[0], e[1]] > threshold:
            G_new.add_edge(e[0], e[1], weight=1)
        
        if sensitive[e[0]] == sensitive[e[1]]:
            preds_neg_intra.append(sigmoid(preds[e[0], e[1]]))
        else:
            preds_neg_inter.append(sigmoid(preds[e[0], e[1]]))
        # if sensitive[e[0]] == sensitive[e[1]]:
        #     preds_neg_intra.append(preds[e[0], e[1]])
        # else:
        #     preds_neg_inter.append(preds[e[0], e[1]])

    
    # homophily
    cnt = 0
    for edge in np.array(G_new.edges):
        src_class = sensitive[edge[0]]
        dst_class = sensitive[edge[1]]
        if src_class == dst_class:
            # print(f"src : {edge[0], src_class}\t dst : {edge[1], dst_class}")
            cnt += 1

    print(f"New network homophily : {cnt / len(G_new.edges) :.2f}")


    res = {}
    for preds_pos, preds_neg, type in zip((preds_pos_intra, preds_pos_inter, preds_pos_intra + preds_pos_inter),
                                          (preds_neg_intra, preds_neg_inter, preds_neg_intra + preds_neg_inter),
                                          ("intra", "inter", "overall")):
        preds_all = np.hstack([preds_pos, preds_neg])
        labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
        roc_score = roc_auc_score(labels_all, preds_all)
        ap_score = average_precision_score(labels_all, preds_all)
        err = (np.sum(list(map(lambda x: x >= threshold, preds_pos))) + np.sum(
            list(map(lambda x: x < threshold, preds_neg)))) / (len(preds_pos) + len(preds_neg))

        score_avg = (sum(preds_pos) + sum(preds_neg)) / (len(preds_pos) + len(preds_neg))
        pos_avg, neg_avg = sum(preds_pos) / len(preds_pos), sum(preds_neg) / len(preds_neg)

        res[type] = [roc_score, ap_score, err, score_avg, pos_avg, neg_avg]


    ks_pos = stats.ks_2samp(preds_pos_intra, preds_pos_inter)[0]
    ks_neg = stats.ks_2samp(preds_neg_intra, preds_neg_inter)[0]

    # calculate modularity
    communities = [set(np.where(sens == sensitive)[0].tolist()) for sens in np.unique(sensitive)]
    modularity_new = nx.community.modularity(G_new, communities)
    modularity_ground = nx.community.modularity(G_ground, communities)
    modred = (modularity_ground - modularity_new) / np.abs(modularity_ground)

    scores = res["overall"][0:2] + [modred] + [abs(res["intra"][i] - res["inter"][i]) for i in range(3, 6)] + [ks_pos, ks_neg]

    return scores

def result_print(scores):
    print("Evaluation Results " + '-'*50)
    print('METRIC\t' + ' '.join(
            '{0:>8}'.format(metric) for metric in ["auc", "ap", "modred", "dp", "true", "false", "fnr", "tnr"]))
    print('VALUE\t' + ' '.join('{0:>8.4f}'.format(value) for value in scores))
    print('-'*50)
