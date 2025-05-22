# Demonstration of STGSN
import sys
import os


import os
import random
from tensorboardX import SummaryWriter
from tqdm import trange

import torch
import torch.optim as optim
from model.STGSN.modules import *
from model.STGSN.loss import *
from model.utils import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(0)

def get_node_map(node_set):
    node_idxs = sorted(list(node_set))
    node_map = {}
    node_cnt = 0
    for node_idx in node_idxs:
        node_map[node_idx] = node_cnt
        node_cnt += 1
    return node_map

def save_model(model, optimizer, save_dir, epoch, rmse, mae):
    rmse = round(rmse, 3)
    mae = round(mae, 3)
    save_path = f'{save_dir}/{epoch}_{rmse}_{mae}.pth'
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f'Model saved to {save_path}')

# ====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================
data_name = 'INVIS15'
num_nodes = 232 # Number of nodes (Level-1 w/ fixed node set)
num_snaps = 431 # Number of snapshots
max_thres = 50 # Threshold for maximum edge weight
feat_dim = 12 # Dimensionality of feature input
enc_dims = [feat_dim, 32, 32, 32] # Layer configuration of encoder
emb_dim = enc_dims[-1] # Dimensionality of dynamic embedding
win_size = 10 # Window size of historical snapshots
theta = 0.1 # Hyper-parameter for collapsed graph

# ====================
base_dataset_path = './Dataset'
edge_seq = np.load(f'{base_dataset_path}/{data_name}/{data_name}_edge_seq.npy', allow_pickle=True)
feat = np.load(f'{base_dataset_path}/{data_name}/{data_name}_feat.npy', allow_pickle=True)
feat_tnr = torch.FloatTensor(feat).to(device)
feat_list = []
for i in range(win_size):
    feat_list.append(feat_tnr)
node_set_seq = np.load(f'{base_dataset_path}/{data_name}/{data_name}_node_seq.npy', allow_pickle=True)

# ====================
dropout_rate = 0.2 # Dropout rate
epsilon = 1e-2 # Threshold of zero-refining
batch_size = 1 # Batch size
num_epochs = 300 # Number of training epochs
num_val_snaps = 10 # Number of validation snapshots
num_test_snaps = 50 # Number of test snapshots
num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

# ====================
# Define the model
model_name = 'STGSN'
model = STGSN(enc_dims, dropout_rate).to(device)
# ==========
# Define the optimizer
opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

# ====================

save_base_path = './Exp'
log_dir = f'{save_base_path}/{data_name}/{model_name}/tb/'
pt_dir = f'{save_base_path}/{data_name}/{model_name}/pt/'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(pt_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

test_max_rmse = 1e+9


export = True
def get_latest_file(directory):
    if not os.path.exists(directory):
        raise ValueError(f"The directory {directory} does not exist.")
    files = [os.path.join(directory, file) for file in os.listdir(directory)]
    files = [file for file in files if os.path.isfile(file)]
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file
# ====================
for epoch in range(num_epochs):
    
    if export:
        model_path = get_latest_file(pt_dir)
        model_state = torch.load(model_path)['model_state_dict']
        model.load_state_dict(model_state, True)
        print('load')
        model.eval()
        est_list = []
        gnd_list = []
        
        RMSE_list = []
        MAE_list = []
        

        for tau in range(num_snaps-num_test_snaps, num_snaps):
                # ====================
                sup_list = []  # List of GNN support (tensor)
                col_net = np.zeros((num_nodes, num_nodes))
                coef_sum = 0.0
                for t in range(tau-win_size, tau):
                    # ==========
                    edges = edge_seq[t]
                    adj = get_adj_wei(edges, num_nodes, max_thres)
                    adj_norm = adj/max_thres
                    sup = get_gnn_sup_d(adj_norm)
                    sup_sp = sp.sparse.coo_matrix(sup)
                    sup_sp = sparse_to_tuple(sup_sp)
                    idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                    vals = torch.FloatTensor(sup_sp[1]).to(device)
                    sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
                    sup_list.append(sup_tnr)
                    # ==========
                    coef = (1-theta)**(tau-t)
                    col_net += coef*adj_norm
                    coef_sum += coef
                # ==========
                col_net /= coef_sum
                col_sup = get_gnn_sup_d(col_net)
                col_sup_sp = sp.sparse.coo_matrix(col_sup)
                col_sup_sp = sparse_to_tuple(col_sup_sp)
                idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(device)
                vals = torch.FloatTensor(col_sup_sp[1]).to(device)
                col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
                # ==========
                # Get the prediction result
                adj_est = model(sup_list, feat_list, col_sup_tnr, feat_tnr, num_nodes)
                if torch.cuda.is_available():
                    adj_est = adj_est.cpu().data.numpy()
                else:
                    adj_est = adj_est.data.numpy()
                adj_est *= max_thres  # Rescale the edge weights to the original value range
                # ==========
                # Refine the prediction result
                adj_est = (adj_est+adj_est.T)/2
                for r in range(num_nodes):
                    adj_est[r, r] = 0
                for r in range(num_nodes):
                    for c in range(num_nodes):
                        if adj_est[r, c] <= epsilon:
                            adj_est[r, c] = 0

                # ====================
                # Get the ground-truth
                edges = edge_seq[tau]
                gnd = get_adj_wei(edges, num_nodes, max_thres)
                # ====================
                # Evaluate the prediction result
                # ---------------------------------- Add: small matrix ---------------------------------- #
                node_set = node_set_seq[t]
                node_map = get_node_map(node_set)
                
                real_row = list(node_map.keys())
                real_col = list(node_map.keys())
                est_list.append(adj_est)
                gnd_list.append(gnd)
                adj_est = adj_est[np.ix_(real_row, real_col)]
                gnd = gnd[np.ix_(real_row, real_col)]
                num_nodes_t = len(node_set)
                # ------------------------------------------------------------------------- #
                RMSE = get_RMSE(adj_est, gnd, num_nodes_t)
                MAE = get_MAE(adj_est, gnd, num_nodes_t)
                # ==========
                RMSE_list.append(RMSE)
                MAE_list.append(MAE)

        # ====================
        RMSE_mean = np.mean(RMSE_list)
        RMSE_std = np.std(RMSE_list, ddof=1)
        MAE_mean = np.mean(MAE_list)
        MAE_std = np.std(MAE_list, ddof=1)
        print('Test Epoch %d RMSE %f %f MAE %f %f' % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))

        os.makedirs(f'./Visual/{model_name}/{data_name}', exist_ok=True)

        np.save(f'./Visual/{model_name}/{data_name}/est_adj.npy', np.stack(est_list, axis=0)) 
        np.save(f'./Visual/{model_name}/{data_name}/gnd.npy', np.stack(gnd_list, axis=0)) 
        break
    
    # ====================
    # Pre-train the model
    model.train()
    num_batch = int(np.ceil(num_train_snaps/batch_size))  # Number of batch
    total_loss = 0.0
    for b in range(num_batch):
        start_idx = b*batch_size
        end_idx = (b+1)*batch_size
        if end_idx>num_train_snaps:
            end_idx = num_train_snaps
        # ====================
        # Training for current batch
        batch_loss = 0.0
        for tau in range(start_idx, end_idx):
            # ==========
            sup_list = []  # List of GNN support (tensor)
            col_net = np.zeros((num_nodes, num_nodes))
            coef_sum = 0.0
            for t in range(tau-win_size, tau):
                # ==========
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj/max_thres
                sup = get_gnn_sup_d(adj_norm)
                sup_sp = sp.sparse.coo_matrix(sup)
                sup_sp = sparse_to_tuple(sup_sp)
                idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                vals = torch.FloatTensor(sup_sp[1]).to(device)
                sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
                sup_list.append(sup_tnr)
                # ==========
                coef = (1-theta)**(tau-t)
                col_net += coef*adj_norm
                coef_sum += coef
            # ==========
            col_net /= coef_sum
            col_sup = get_gnn_sup_d(col_net)
            col_sup_sp = sp.sparse.coo_matrix(col_sup)
            col_sup_sp = sparse_to_tuple(col_sup_sp)
            idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(device)
            vals = torch.FloatTensor(col_sup_sp[1]).to(device)
            col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
            # ==========
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres) # Training ground-truth
            gnd_norm = gnd/max_thres # Normalize the edge weights (in ground-truth) to [0, 1]
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
            # ==========
            adj_est = model(sup_list, feat_list, col_sup_tnr, feat_tnr, num_nodes)
            loss_ = get_STGSN_loss_wei(adj_est, gnd_tnr)
            batch_loss = batch_loss + loss_
        # ===========
        # Update model parameter according to batch loss
        opt.zero_grad()
        batch_loss.backward()
        opt.step()
        total_loss = total_loss + batch_loss
    print('Epoch %d Total Loss %f' % (epoch, total_loss))
    writer.add_scalar(f'Train/Loss', total_loss, epoch)

    # ====================
    # Validate the model
    model.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        col_net = np.zeros((num_nodes, num_nodes))
        coef_sum = 0.0
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj / max_thres
            sup = get_gnn_sup_d(adj_norm)
            sup_sp = sp.sparse.coo_matrix(sup)
            sup_sp = sparse_to_tuple(sup_sp)
            idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
            vals = torch.FloatTensor(sup_sp[1]).to(device)
            sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
            sup_list.append(sup_tnr)
            # ==========
            coef = (1-theta)**(tau-t)
            col_net += coef*adj_norm
            coef_sum += coef
        # ==========
        col_net /= coef_sum
        col_sup = get_gnn_sup_d(col_net)
        col_sup_sp = sp.sparse.coo_matrix(col_sup)
        col_sup_sp = sparse_to_tuple(col_sup_sp)
        idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(device)
        vals = torch.FloatTensor(col_sup_sp[1]).to(device)
        col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
        # ==========
        # Get the prediction result
        adj_est = model(sup_list, feat_list, col_sup_tnr, feat_tnr, num_nodes)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres  # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        adj_est = (adj_est+adj_est.T)/2
        for r in range(num_nodes):
            adj_est[r, r] = 0
        for r in range(num_nodes):
            for c in range(num_nodes):
                if adj_est[r, c] <= epsilon:
                    adj_est[r, c] = 0

        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the prediction result
        # ---------------------------------- Add: small matrix ---------------------------------- #
        node_set = node_set_seq[t]
        node_map = get_node_map(node_set)
        real_row = list(node_map.keys())
        real_col = list(node_map.keys())
        adj_est = adj_est[np.ix_(real_row, real_col)]
        gnd = gnd[np.ix_(real_row, real_col)]
        num_nodes_t = len(node_set)
        # ------------------------------------------------------------------------- #
        
        RMSE = get_RMSE(adj_est, gnd, num_nodes_t)
        MAE = get_MAE(adj_est, gnd, num_nodes_t)
        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)

    # ====================
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)
    print('Val #%d RMSE %f %f MAE %f %f' % (epoch, RMSE_mean, RMSE_std, MAE_mean, MAE_std))
    writer.add_scalar(f'Val/RMSE_mean', RMSE_mean, epoch)
    writer.add_scalar(f'Val/RMSE_std', RMSE_std, epoch)
    writer.add_scalar(f'Val/MAE_mean', MAE_mean, epoch)
    writer.add_scalar(f'Val/MAE_std', MAE_std, epoch)
    # ====================
    # Test the model
    model.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    for tau in range(num_snaps-num_test_snaps, num_snaps):
        # ====================
        sup_list = []  # List of GNN support (tensor)
        col_net = np.zeros((num_nodes, num_nodes))
        coef_sum = 0.0
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres
            sup = get_gnn_sup_d(adj_norm)
            sup_sp = sp.sparse.coo_matrix(sup)
            sup_sp = sparse_to_tuple(sup_sp)
            idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
            vals = torch.FloatTensor(sup_sp[1]).to(device)
            sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
            sup_list.append(sup_tnr)
            # ==========
            coef = (1-theta)**(tau-t)
            col_net += coef*adj_norm
            coef_sum += coef
        # ==========
        col_net /= coef_sum
        col_sup = get_gnn_sup_d(col_net)
        col_sup_sp = sp.sparse.coo_matrix(col_sup)
        col_sup_sp = sparse_to_tuple(col_sup_sp)
        idxs = torch.LongTensor(col_sup_sp[0].astype(float)).to(device)
        vals = torch.FloatTensor(col_sup_sp[1]).to(device)
        col_sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, col_sup_sp[2]).float().to(device)
        # ==========
        # Get the prediction result
        adj_est = model(sup_list, feat_list, col_sup_tnr, feat_tnr, num_nodes)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres  # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        adj_est = (adj_est+adj_est.T)/2
        for r in range(num_nodes):
            adj_est[r, r] = 0
        for r in range(num_nodes):
            for c in range(num_nodes):
                if adj_est[r, c] <= epsilon:
                    adj_est[r, c] = 0

        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the prediction result
        # ---------------------------------- Add: small matrix ---------------------------------- #
        node_set = node_set_seq[t]
        node_map = get_node_map(node_set)
        real_row = list(node_map.keys())
        real_col = list(node_map.keys())
        adj_est = adj_est[np.ix_(real_row, real_col)]
        gnd = gnd[np.ix_(real_row, real_col)]
        num_nodes_t = len(node_set)
        # ------------------------------------------------------------------------- #
        RMSE = get_RMSE(adj_est, gnd, num_nodes_t)
        MAE = get_MAE(adj_est, gnd, num_nodes_t)
        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)

    # ====================
    RMSE_mean = np.mean(RMSE_list)
    RMSE_std = np.std(RMSE_list, ddof=1)
    MAE_mean = np.mean(MAE_list)
    MAE_std = np.std(MAE_list, ddof=1)
    writer.add_scalar(f'Test/RMSE_mean', RMSE_mean, epoch)
    writer.add_scalar(f'Test/RMSE_std', RMSE_std, epoch)
    writer.add_scalar(f'Test/MAE_mean', MAE_mean, epoch)
    writer.add_scalar(f'Test/MAE_std', MAE_std, epoch)
    
    mrx_mean = RMSE_mean + MAE_mean
    if mrx_mean < test_max_rmse:
        test_max_rmse = mrx_mean
        save_model(model, opt, pt_dir, epoch, RMSE_mean, MAE_mean)

