import os
import sys
import torch
import argparse
import numpy as np
import torch.optim as optim

from tqdm import trange
from dataset.dataset import *
from utils.paramset import Paramset
from utils.seeds import setup_seed
from utils.file import dynamic_import_lib
from utils.logger import Logger
from utils.metrics import get_RMSE, get_MAE

print(os.getcwd())

EPSILON = 0.01 
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
LOG_PATH = './Exp'
DATASET_PATH = './Data'


setup_seed(0)
metrics_max = 1e+9
epoch_start = 0

args_parse = argparse.ArgumentParser()
args_parse.add_argument('--config',
                        type=str,
                        default='./SALT/config/HMob/SALT.yaml')
args_parse.add_argument('--reproduction',
                        type=lambda x: x.lower() in ['true', '1', 'yes'],
                        default='true')
args = args_parse.parse_args()

params = Paramset(args.config)

logger = Logger(params, LOG_PATH, tensorboard=True)

dataset = Dataset(params, DATASET_PATH)

Model = dynamic_import_lib(params.model)
net = Model(params).to(params.device)

if args.reproduction:
    model_state = torch.load(f'./Pretrained/{params.data_name}/{params.data_name}_pretrain.pth')['model_dict']
    net.load_state_dict(model_state, True)

opt = optim.RMSprop(net.parameters(), lr=float(params.learning_rate), weight_decay=1e-5)


def train():
    net.train()
    train_cnt = 0
    loss_list = []
    
    for tau in trange(params.win_size, dataset.num_train_snaps - params.pred_size + 1):
        sup_list, noise_list, feat_list= dataset.get_win_data(tau-params.win_size, tau)
        gnd_list = dataset.get_win_gnd_data(tau-params.win_size + 1, tau + params.pred_size)
        
        adj_est_list = net(sup_list,feat_list,noise_list,pred_flag=False)

        loss = net.causal_training_loss(adj_est_list,
                                        gnd_list,
                                        params.Loss['alpha'],
                                        params.Loss['theta'])

        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_list.append(loss.item())
        train_cnt += 1
        if train_cnt % 100 == 0:
            print('-Train %d / %d / %d' % (train_cnt, dataset.num_train_snaps, np.mean(loss_list)))
            
    gen_loss_mean = np.mean(loss_list)
    result = {'G-Loss': gen_loss_mean}
    print('#%d Train G-Loss %f' % (epoch, gen_loss_mean))
    logger.write(result, 'Train', epoch) 


@torch.no_grad()
def evaluate(start, end, mode):
    net.eval()

    results_list_dict = {'RMSE': [],
                         'MAE': []}
    
    metrics_dict = {'RMSE': get_RMSE,
                    'MAE': get_MAE}
    
    for tau in trange(start, end, params.pred_size):
        sup_list, noise_list,feat_list = dataset.get_win_data(tau - params.win_size, tau)
        
        adj_est_list = net(sup_list, feat_list, noise_list, pred_flag=True)
        
        for t in range(tau, tau + params.pred_size):
            adj_est = adj_est_list[t - tau]
            adj_est = adj_est.cpu().numpy()
            adj_est *= dataset.max_thres

            np.fill_diagonal(adj_est, 0)
            adj_est[adj_est <= EPSILON] = 0
            
            edges = dataset.edge_seq[t]
            if dataset.data_level == 2:
                num_nodes = dataset.num_nodes
                gnd = get_adj_wei(edges, num_nodes, dataset.max_thres)
                real_row = list(dataset.node_map_seq_gbl[t].keys())
                real_col = list(dataset.node_map_seq_gbl[t].keys())
                gnd = gnd[np.ix_(real_row, real_col)]
                adj_est = adj_est[np.ix_(real_row, real_col)]
                num_nodes = dataset.num_nodes_seq_gbl[t]
            else:
                num_nodes = dataset.num_nodes
                gnd = get_adj_wei(edges, num_nodes, dataset.max_thres)

            for metric, func in metrics_dict.items():
                result = func(adj_est, gnd, num_nodes)
                results_list_dict[metric].append(result)
            

    results_dict = {}
    for metric, result_list in results_list_dict.items():
        results_dict[metric] = np.mean(result_list)
        results_dict[metric+'_std'] = np.std(result_list, ddof=1)

    if mode in ['Val', 'Test']:
        print(f'{mode} #{epoch} RMSE {results_dict["RMSE"]:.4f} {results_dict["RMSE_std"]:.4f} '
          f'MAE {results_dict["MAE"]:.4f} {results_dict["MAE_std"]:.4f}')
        log_list = ['RMSE', 'MAE']
        log_results = {key: results_dict[key] for key in log_list}
        logger.write(log_results, mode, epoch)
    return results_dict


for epoch in range(epoch_start, params.num_epochs):
    if args.reproduction:
            test_res = evaluate(dataset.num_snaps-dataset.num_test_snaps,
                        dataset.num_snaps,
                        'Test')
            test_rmse, test_mae = test_res['RMSE'], test_res['MAE']
            print(f'TEST RMSE:  {test_rmse},    MAE:    {test_mae}')
            sys.exit(0)
    else:
        train()
        val_res = evaluate(dataset.num_snaps-dataset.num_test_snaps-dataset.num_val_snaps,
                        dataset.num_snaps-dataset.num_test_snaps,
                            'Val')
        
        test_res = evaluate(dataset.num_snaps-dataset.num_test_snaps,
                            dataset.num_snaps,
                            'Test')

        val_rmse, val_mae = val_res['RMSE'], val_res['MAE']
        test_rmse, test_mae = test_res['RMSE'], test_res['MAE']
        metrics_mean = (val_rmse + val_mae).mean()
        if metrics_mean < metrics_max:
            metrics_max = metrics_mean
            logger.save_model(net, opt, epoch, f'{round(test_rmse, 3)}_{round(test_mae, 3)}')