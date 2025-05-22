#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :load datasets

import os
import copy
import math
import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix


def load_info(data_name, params=None):
    data_info = {
        'T-Drive': {
            'num_nodes': 1279,
            'num_snaps': 300,
            'max_thres': 5000,
            'noise_dim': 512,
            'feat_dim': 32,
            'pos_dim': 256,
            'GNN_feat_dim': 256
        },
        'DC': {
            'num_nodes': 128,
            'num_snaps': 700,
            'max_thres': 5000,
            'noise_dim': 100,
            'feat_dim': 32,
            'pos_dim': 32,
            'GNN_feat_dim': 64
        },
        'HMob': {
            'num_nodes': 92,
            'num_snaps': 500,
            'max_thres': 250,
            'noise_dim': 64,
            'feat_dim': None, 
            'pos_dim': 32,
            'GNN_feat_dim': 32
        },
        'IoT': {
            'num_nodes': 668,
            'num_snaps': 144,
            'max_thres': 1024,
            'noise_dim': 48,
            'feat_dim': 32,
            'pos_dim': 32
        },
        'LyonSchool': {
            'num_nodes': 242,
            'num_snaps': 157,
            'max_thres': 20,
            'noise_dim': 32,
            'feat_dim': 11,
            'pos_dim': 32
        },
        'Mesh': {
            'num_nodes': 38,
            'num_snaps': 445,
            'max_thres': 2000,
            'noise_dim': params.Noise_dims if params else None,
            'feat_dim': 32,
            'pos_dim': 16
        },
        'INVIS13': {
            'num_nodes': 100,
            'num_snaps': 403,
            'max_thres': 50,
            'noise_dim': 5,
            'feat_dim': 5,
            'pos_dim': 5
        },
        'INVIS15': {
            'num_nodes': 232,
            'num_snaps': 431,
            'max_thres': 50,
            'noise_dim': 5,
            'feat_dim': 12,
            'pos_dim': 12
        },
        'LH10': {
            'num_nodes': 81,
            'num_snaps': 253,
            'max_thres': 50,
            'noise_dim': 2,
            'feat_dim': 5,
            'pos_dim': 5
        },
        'Thiers13': {
            'num_nodes': 329,
            'num_snaps': 179,
            'max_thres': 50,
            'noise_dim': 16,
            'feat_dim': 9,
            'pos_dim': 16
        }
    }
    
    if data_name not in data_info:
        raise ValueError('Invalid data name')
    
    info = data_info[data_name].copy()
    if data_name == 'Mesh':
        info['GNN_feat_dim'] = info['feat_dim'] + info['pos_dim']
    else:
        info['GNN_feat_dim'] = info.get('GNN_feat_dim', info['pos_dim'])
    
    return info

def load(nodes_num):
    """
    load_dataset
    :param nodes_num:
    :return:
    """
    path = "dataset/dblp_timestamp/"

    train_e_feat_path = path + 'train_e_feat/' + type + '/'
    test_e_feat_path = path + 'test_e_feat/' + type + '/'

    train_n_feat_path = path + type + '/' + 'train_n_feat/'
    test_n_feat_path = path + type + '/' + 'test_n_feat/'


    path = path + type
    train_path = path + '/train/'
    test_path = path + '/test/'

    train_n_feat = read_e_feat(train_n_feat_path)
    test_n_feat = read_e_feat(test_n_feat_path)

    train_e_feat = read_e_feat(train_e_feat_path)
    test_e_feat = read_e_feat(test_e_feat_path)

    num = 0
    train_graph = read_graph(train_path, nodes_num, num)
    num = num + len(train_graph)
    test_graph = read_graph(test_path, nodes_num, num)
    return train_graph, train_e_feat, train_n_feat, test_graph, test_e_feat, test_n_feat


def load_r(name, data_level):
    path = "dataset/" + name
    path_ei = path + '/' + 'edge_index/'
    path_nf = path + '/' + 'node_feature/'
    path_ef = path + '/' + 'edge_feature/'
    path_et = path + '/' + 'edge_time/'
    path_ew = path + '/' + 'edge_weight/'

    dataset_info = load_info(name)
    
    if data_level == 'l2':
        node_set_seq = np.load(f'./WinDataset/dataset/{name}/{name}_node_seq.npy', allow_pickle=True)
    else:
        node_set_seq = None

    edge_index = read_npz(path_ei)
    node_feature = read_npz(path_nf)
    edge_weight = read_npz(path_ew)



    nodes_num = dataset_info['num_nodes']
    if node_feature[0].shape[0] !=nodes_num:
        node_feature[0] = node_feature[0][:nodes_num]
    
    T = len(edge_index)
    node_feature_list = node_feature * T
    
    sub_graph = []
    for index, e_i in enumerate(edge_index):
        row = np.concatenate((e_i[0], e_i[1]))
        col = np.concatenate((e_i[1], e_i[0]))
        weights = np.concatenate((edge_weight[index], edge_weight[index]))
        if name == 'HMob':
            row = e_i[0]
            col = e_i[1]
            weights = edge_weight[index]
        sub_g = coo_matrix((weights, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)
    return sub_graph, node_feature_list, dataset_info, node_set_seq


def read_npz(path):
    filesname = os.listdir(path)
    npz = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('.')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        npz.append(np.load(path+filename))

    return npz


def read_e_feat(path):
    filesname = os.listdir(path)
    e_feat = []
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id)
        file_s[id] = filename
    for filename in file_s:
        e_feat.append(np.load(path+filename))

    return e_feat


def read_graph(path, nodes_num, num):
    filesname = os.listdir(path)
    file_s = filesname.copy()
    for filename in filesname:
        id = filename.split('_')[0]
        id = int(id) - num
        file_s[id] = filename

    sub_graph = []
    for file in file_s:
        sub_ = pd.read_csv(path + file)
        row = sub_.src_l.values
        col = sub_.dst_l.values
        node_m = set(row).union(set(col))
        ts = [1] * len(row)
        sub_g = coo_matrix((ts, (row, col)), shape=(nodes_num, nodes_num))
        sub_graph.append(sub_g)
    return sub_graph


