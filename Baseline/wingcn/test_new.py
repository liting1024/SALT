import numpy as np
import torch
from copy import deepcopy
from model.loss import get_MAE, get_RMSE, mse_mae_loss
from model.utils import mae_rmse_based_eval_meta


def test(graph_l, model, args, logger, n, S_dw, device, max_thres, data_level, num_node):
    beta = args.beta
    avg_mae = 0.0
    avg_rmse = 0.0

    graph_test = graph_l[n:]
    fast_weights = list(map(lambda p: p[0], zip(model.parameters())))
    for idx, g_test in enumerate(graph_test):

        if idx == len(graph_test) - 1:
            break

        graph_train = deepcopy(g_test.node_feature)
        graph_train = graph_train.to(device)
        g_test = g_test.to(device)

        pred = model(g_test, graph_train, fast_weights)
        loss = mse_mae_loss(pred, g_test.edge_label)

        graph_train = graph_train.to('cpu')
        grad = torch.autograd.grad(loss, fast_weights)
        g_test = g_test.to('cpu')

        S_dw = list(map(lambda p: beta * p[1] + (1 - beta) * p[0].pow(2), zip(grad, S_dw)))

        fast_weights = list(map(lambda p: p[1] - args.maml_lr / (torch.sqrt(p[2]) + 1e-8) * p[0], zip(grad, fast_weights, S_dw)))

        graph_test[idx + 1] = graph_test[idx + 1].to(device)
        graph_test[idx + 1].node_feature = graph_test[idx + 1].node_feature.to(device)
        pred = model(graph_test[idx + 1], graph_test[idx + 1].node_feature, fast_weights)

        loss = mse_mae_loss(pred, graph_test[idx + 1].edge_label)

        if data_level == 'l2':
            postive_idx = graph_test[idx + 1].postive_edge_label_index[0] * num_node + graph_test[idx + 1].postive_edge_label_index[1]
            pred_re_norm = pred[postive_idx] * max_thres
            label_re_norm = graph_test[idx + 1].edge_label[:,postive_idx] * max_thres
        else:
            pred_re_norm = pred * max_thres
            label_re_norm = graph_test[idx + 1].edge_label* max_thres
            
        mae = get_MAE(pred_re_norm, label_re_norm)
        rmse = get_RMSE(pred_re_norm, label_re_norm)
        
        avg_mae += mae
        avg_rmse += rmse
    avg_mae /= len(graph_test) - 1
    avg_rmse /= len(graph_test) - 1
    logger.info({'avg_mae': avg_mae, 'avg_rmse': avg_rmse})
    return avg_mae, avg_rmse
