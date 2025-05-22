import tqdm
import torch
import dgl
import random
import math
import time
import numpy as np
from copy import deepcopy
from model.loss import get_MAE, get_RMSE, mse_mae_loss
from model.utils import mae_rmse_based_eval_meta


def train(args, model, optimizer, device, graph_l, logger, n, max_thres, data_level, num_node):

    best_param = {'mae_rmse': 1e5, 'best_state': None, 'best_s_dw': None}
    earl_stop_c = 0
    epoch_count = 0

    for epoch in range(args.epochs):
        graph_l_cpy = deepcopy(graph_l)
        all_mae_rmse = 0.0

        i = 0
        fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
        S_dw = [0] * len(fast_weights)
        train_count = 0
        while i < (n - args.window_num):
            if i != 0:
                i = random.randint(i, i + args.window_num)
            if i >= (n - args.window_num):
                break
            graph_train = graph_l[i: i + args.window_num]
            i = i + 1
            features = [graph_unit.node_feature.to(device) for graph_unit in graph_train]

            fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
            window_mae_rmse = 0.0
            losses = torch.tensor(0.0).to(device)
            count = 0
            for idx, graph in enumerate(graph_train):
                if idx == args.window_num - 1:
                    break
                feature_train = deepcopy(features[idx])
                graph = graph.to(device)
                pred = model(graph, feature_train, fast_weights)
                loss = mse_mae_loss(pred, graph.edge_label)
                grad = torch.autograd.grad(loss, fast_weights)

                graph = graph.to('cpu')
                feature_train = feature_train.to('cpu')

                beta = args.beta
                S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0] * p[0], zip(grad, S_dw)))

                fast_weights = list(
                    map(lambda p: p[1] - args.maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw))
                    )

                graph_train[idx + 1] = graph_train[idx + 1].to(device)
                pred = model(graph_train[idx + 1], features[idx + 1], fast_weights)
                loss = mse_mae_loss(pred, graph_train[idx + 1].edge_label)

                edge_label = graph_train[idx + 1].edge_label
                edge_label_index = graph_train[idx + 1].edge_label_index
                mae, rmse = mae_rmse_based_eval_meta(model, graph_train[idx + 1], features[idx+1],
                            fast_weights,max_thres=max_thres, data_level=data_level, num_node=num_node)
                
                graph_train[idx + 1].edge_label = edge_label
                graph_train[idx + 1].edge_label_index = edge_label_index
                

                droprate = torch.FloatTensor(np.ones(shape=(1)) * args.drop_rate)
                masks = torch.bernoulli(1. - droprate).unsqueeze(1) 

                if masks[0][0]:
                    losses = losses + loss
                    count += 1
                    window_mae_rmse += (mae+rmse)/2
                
               

            if losses:
                losses = losses / count
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
            if count:
                all_mae_rmse += window_mae_rmse / count
            train_count += 1

        all_mae_rmse = all_mae_rmse / train_count
        epoch_count += 1
        
        if all_mae_rmse < best_param['mae_rmse']:
            best_param = {'mae_rmse': all_mae_rmse, 'best_state': deepcopy(model.state_dict()), 'best_s_dw': S_dw}
            logger.info('meta epoch:{}, all_mae_rmse: {:.5f},'. format(epoch, all_mae_rmse))
            earl_stop_c = 0
        else:
            earl_stop_c += 1
            if earl_stop_c == 10:
                break
    return best_param