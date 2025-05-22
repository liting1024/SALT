#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import dgl
import math
# import wandb
import torch
import random
import argparse
import numpy as np


from tqdm import tqdm
from model import WinGNN
from test_new import test
from train_new import train
from model.config import cfg
from deepsnap.graph import Graph
from model.Logger import getLogger
from dataset_prep import load, load_r
from model.utils import create_optimizer
from deepsnap.dataset import GraphDataset

import warnings
warnings.filterwarnings("ignore")


def get_node_map(node_set):
    node_idxs = sorted(list(node_set))
    node_map = {}
    node_cnt = 0
    for node_idx in node_idxs:
        node_map[node_idx] = node_cnt
        node_cnt += 1
    return node_map


if __name__ == '__main__':
    dataset = 'INVIS15'
    
    if dataset in ['DC', 'HMob', 'T-Drive']:
        win_size = 5
    else:
        win_size = 10
        
    if dataset in ['LyonSchool', 'IoT','INVIS13', 'Thiers13', 'LH10', 'INVIS15']:
        data_level = 'l2'
    else:
        data_level = 'l1'
        
    print(f'------- dataset: {dataset} --------')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset, help='Dataset: uci-msg, DC, HMob, IoT, LyonSchool, INVIS13')

    parser.add_argument('--cuda_device', type=int,
                        default=0, help='Cuda device no -1')

    parser.add_argument('--seed', type=int, default=2023, help='split seed')

    parser.add_argument('--repeat', type=int, default=1, help='number of repeat model')

    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs to train.')

    parser.add_argument('--out_dim', type=int, default=64,
                        help='model output dimension.')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type')

    parser.add_argument('--lr', type=float, default=0.0005,
                        help='initial learning rate.')

    parser.add_argument('--maml_lr', type=float, default=0.0005,
                        help='meta learning rate')

    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay (L2 loss on parameters).')

    parser.add_argument('--drop_rate', type=float, default=0.16, help='drop meta loss')

    parser.add_argument('--num_layers', type=int,
                        default=2, help='GNN layer num')

    parser.add_argument('--num_hidden', type=int, default=512,
                        help='number of hidden units of MLP')

    parser.add_argument('--window_num', type=float, default=win_size,
                        help='windows size')

    parser.add_argument('--dropout', type=float, default=0.1,
                        help='GNN dropout')

    parser.add_argument('--residual', type=bool, default=False,
                        help='skip connection')

    parser.add_argument('--beta', type=float, default=0.89,
                        help='The weight of adaptive learning rate component accumulation')



    args = parser.parse_args()

    save_path = os.path.join(
            f'num_layers={args.num_layers},num_hidden={args.num_hidden},decoder={cfg.model.edge_decoding}',
            f'lr={args.lr},maml_lr={args.maml_lr},weightdecay={args.weight_decay},beta={args.beta}',
            f'winnum={args.window_num},droprate={args.dropout}',
        )
    logger = getLogger( os.path.join(cfg.log_path, args.dataset, save_path) ) 

    # load datasets
    if args.dataset == 'dblp':
        dataset = args.dataset
        e_feat = np.load('dataset/{0}/ml_{0}.npy'.format(dataset))
        n_feat_ = np.load('dataset/{0}/ml_{0}_node.npy'.format(dataset))
        train_data, train_e_feat, train_n_feat, test_data, test_e_feat, test_n_feat = load("Norandom", len(n_feat_))
        graphs = []
        for tr in train_data:
            graphs.append(tr)
        for te in test_data:
            graphs.append(te)
        n_feat = [n_feat_ for i in range(len(graphs))]
    elif args.dataset in ["reddit_body", "reddit_title", "as_733",
                          "uci-msg", "bitcoinotc", "bitcoinalpha",
                          'stackoverflow_M',
                          ]:
        graphs, e_feat, e_time, n_feat = load_r(args.dataset)
    elif args.dataset in [
                          'HMob', 'DC', 'IoT', 'INVIS13', 'LyonSchool','T-Drive', 'Thiers13', 'LH10', 'INVIS15'
                          ]:
        graphs, n_feat, dataset_info, node_set_seq = load_r(args.dataset, data_level)
        num_node = int(dataset_info['num_nodes'])
        max_thres = float(dataset_info['max_thres'])
    else:
        raise ValueError
    
    
    model_save_path = os.path.join('model_parameter', args.dataset, save_path,'e_time_feat.pkl')
    model_load_path = 'model_parameter/' + args.dataset


    n_dim = n_feat[0].shape[1]
    n_node = n_feat[0].shape[0]


    device = torch.device(f'cuda:{args.cuda_device}' if args.cuda_device >= 0 else 'cpu')

    all_avg_mae_rmse = 0.0
    best_mae_rmse = 1e5
    best_model = 0

    for rep in range(args.repeat):

        logger.info('num_layers:{}, num_hidden: {}, lr: {}, maml_lr:{}, window_num:{}, drop_rate:{}, Fixed negative sample sampling'.
                    format(args.num_layers, args.num_hidden, args.lr, args.maml_lr, args.window_num, args.drop_rate))
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

        graph_l = []
        # Data set processing
        for idx, graph in tqdm(enumerate(graphs)):  # 700
            graph_d = dgl.from_scipy(graph)

            if n_feat[idx].shape[0] != n_node or n_feat[idx].shape[1] != n_dim: # 
                n_feat_t = graph_l[idx - 1].node_feature
                graph_d.node_feature = torch.Tensor(n_feat_t)
            else:
                graph_d.node_feature = torch.Tensor(n_feat[idx])


            edges = graph_d.edges()
            weights = graph.data
            row = edges[0].numpy()
            col = edges[1].numpy()
            n_e = graph_d.num_edges() - graph_d.num_nodes()

            edge_label_index = list()
            postive_edge_label_index = list()

            
            if data_level == 'l1':
                x_coords, y_coords =  zip(*[(x, y) for x in range(num_node) for y in range(num_node)])
                edge_label_index.append(list(x_coords))
                edge_label_index.append(list(y_coords))
                y = graph.todense().reshape(1,-1)
                postive_edge_label_index.append(row.tolist())
                postive_edge_label_index.append(col.tolist())
                
            elif data_level == 'l2':
                x_coords, y_coords =  zip(*[(x, y) for x in range(num_node) for y in range(num_node)])
                edge_label_index.append(list(x_coords))
                edge_label_index.append(list(y_coords))
                y = graph.todense().reshape(1,-1)
                node_set = node_set_seq[idx]
                node_map = get_node_map(node_set)
                real_row = list(node_map.keys())
                real_col = list(node_map.keys())
                real_row_list = [i for i in real_row for j in real_col]
                real_col_list = [j for i in real_row for j in real_col]
                postive_edge_label_index.append(real_row_list)
                postive_edge_label_index.append(real_col_list)

            graph_d.edge_label = torch.Tensor(y) / max_thres
            graph_d.edge_label_index = torch.LongTensor(edge_label_index)
            graph_d.postive_edge_label_index = torch.LongTensor(postive_edge_label_index)

            graph_l.append(graph_d)


        # model initialization
        model = WinGNN.Model(n_dim, args.out_dim, args.num_hidden, args.num_layers, args.dropout)
        model.train()

        # LightDyG optimizer
        optimizer = create_optimizer(args.optimizer, model, args.lr, args.weight_decay)

        model = model.to(device)

        n = len(graph_l) - 50 

        best_param = train(args, model, optimizer, device, graph_l, logger, n, max_thres, data_level, num_node)

        model.load_state_dict(best_param['best_state'])
        S_dw = best_param['best_s_dw']

        # test
        model.eval()
        avg_mae, avg_rmse = test(graph_l, model, args, logger, n, S_dw, device, max_thres, data_level, num_node)

        if avg_mae+avg_rmse < best_mae_rmse:
            best_model = best_param['best_state']
            best_mae_rmse = avg_mae+avg_rmse
        all_avg_mae_rmse += avg_mae+avg_rmse


    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(best_model, model_save_path)
    all_avg_mae_rmse = all_avg_mae_rmse / args.repeat
    print(all_avg_mae_rmse)
