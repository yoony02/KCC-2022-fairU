###############################################################
# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com
# reference github link : https://github.com/brandeis-machine-learning/FairAdj
#
# @modify : artfjqm2@skku.edu , chri0220@skku.edu, maya0707@skku.edu
#############################################################

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Fair Adjacency Graph Embedding for Link Prediction")

    # experiment
    parser.add_argument('--seed', type=int, default=1, help="seed")
    # parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--dataset', type=str, default="cora")
    parser.add_argument('--scale', action="store_false", help='normalize the data')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='proportion of testing edges')

    # model
    parser.add_argument('--fairdrop', type=bool, default=True, help='Fairdrop cycle num')
    parser.add_argument('--fairdrop_term', type=int, default=10, help='Fairdrop cycle num')
    parser.add_argument('--delta', type=float, default=0.16, help='fairdrop drop ratio')
    parser.add_argument('--beta', type=float, default=0.5, help='hyperparameter for link prediction')

    
    parser.add_argument('--hidden1', type=int, default=32, help='Number of units in hidden layer 1.')
    parser.add_argument('--hidden2', type=int, default=16, help='Number of units in hidden layer 2.')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout rate (1 - keep probability).')

    # optimize
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train.')  # 4
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--eta', type=float, default=0.2, help='Learning rate for adjacency matrix.')

    # for greedy
    parser.add_argument('--greedy_change_pct', type=float, default=0.02, help='Greedy preprocessing change percent')
    parser.add_argument('--greedy', default=False, type=bool,  help='select greedy preprocessing')

    # for adversarial
    parser.add_argument('--adv', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--alpha', type=float, default=10., help='hyperparameter for adversarial loss')
    parser.add_argument('--lr_adv', type=float, default=1., help='learning rate multiple for adversarial net')

    return parser.parse_args()
