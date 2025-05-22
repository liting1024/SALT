import os
import random
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import torch
import torch.optim as optim
from modules import *
from loss import *
from utils import *

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
data_name = 'INVIS13'
num_nodes = 100 # Number of nodes (Level-1 w/ fixed node set)
num_snaps = 403 # Number of snapshots
max_thres = 50 # Threshold for maximum edge weight
struc_dims = [num_nodes, 32] # Layer configuration of structural encoder (FC)
temp_dims = [struc_dims[-1], 16, 16] # Layer configuration of temporal encoder (RNN)
dec_dims = [temp_dims[-1], 32, num_nodes] # Layer configuration of decoder (FC)
beta = 0.1 # Hyper-parameter of loss

# ====================
base_dataset_path = './Dataset'
edge_seq = np.load(f'{base_dataset_path}/{data_name}/{data_name}_edge_seq.npy', allow_pickle=True)
node_set_seq = np.load(f'{base_dataset_path}/{data_name}/{data_name}_node_seq.npy', allow_pickle=True)

# ====================
dropout_rate = 0.2 # Dropout rate
epsilon = 1e-2 # Threshold of zero-refining
win_size = 10 # Window size of historical snapshots
batch_size = 1 # Batch size
num_epochs = 300 # Number of training epochs
num_val_snaps = 10 # Number of validation snapshots
num_test_snaps = 50 # Number of test snapshots
num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

# ====================
# Define the model
model_name = 'D2V'
model = dyngraph2vec(struc_dims, temp_dims, dec_dims, dropout_rate).to(device)
# ==========
# Define the optimizer
opt = optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)

# ====================

save_base_path = './Exp'
log_dir = f'{save_base_path}/{data_name}/{model_name}/tb/'
pt_dir = f'{save_base_path}/{data_name}/{model_name}/pt/'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(pt_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

val_max_rmse = 1e+9

# ====================
for epoch in range(num_epochs):
    # ------------------------------ Train the model ----------------------------- #
    model.train()
    num_batch = int(np.ceil(num_train_snaps/batch_size)) # Number of batch
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
            adj_list = []  # List of historical adjacency matrices
            for t in range(tau-win_size, tau):
                edges = edge_seq[t]
                adj = get_adj_wei(edges, num_nodes, max_thres)
                adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
                adj_tnr = torch.FloatTensor(adj_norm).to(device)
                adj_list.append(adj_tnr)
            # ==========
            edges = edge_seq[tau]
            gnd = get_adj_wei(edges, num_nodes, max_thres) # Training ground-truth
            gnd_norm = gnd/max_thres  # Normalize the edge weights (in ground-truth) to [0, 1]
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
            # ==========
            adj_est = model(adj_list)
            loss_ = get_d2v_loss(adj_est, gnd_tnr, beta)
            batch_loss = batch_loss + loss_
        # ==========
        # Update model parameter according to batch loss
        opt.zero_grad()
        batch_loss.backward()
        opt.step()
        total_loss = total_loss + batch_loss
    print('Epoch %d Total Loss %f' % (epoch, total_loss))
    writer.add_scalar(f'Train/Loss', total_loss, epoch)

    # ====================
    # ---------------------------- Validate the model ---------------------------- #
    model.eval()
    RMSE_list = []
    MAE_list = []
    for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
        # ====================
        adj_list = [] # List of historical adjacency matrices
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            adj_tnr = torch.FloatTensor(adj_norm).to(device)
            adj_list.append(adj_tnr)
        # ====================
        # Get the prediction result
        adj_est = model(adj_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres # Rescale edge weights to the original value range
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
        # Get ground-truth
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the quality of current prediction operation
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
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
    # ====================
    RMSE_mean_val = np.mean(RMSE_list)
    RMSE_std_val  = np.std(RMSE_list, ddof=1)
    MAE_mean_val  = np.mean(MAE_list)
    MAE_std_val  = np.std(MAE_list, ddof=1)
    print('Val Epoch %d RMSE %f %f MAE %f %f' % (epoch, RMSE_mean_val , RMSE_std_val , MAE_mean_val , MAE_std_val))
    writer.add_scalar(f'Val/RMSE_mean', RMSE_mean_val, epoch)
    writer.add_scalar(f'Val/RMSE_std', RMSE_std_val, epoch)
    writer.add_scalar(f'Val/MAE_mean', MAE_mean_val, epoch)
    writer.add_scalar(f'Val/MAE_std', MAE_std_val, epoch)
    
    # ====================
    # ------------------------------ Test the model ------------------------------ #
    model.eval()
    RMSE_list = []
    MAE_list = []
    for tau in range(num_snaps-num_test_snaps, num_snaps):
        # ====================
        adj_list = []  # List of historical adjacency matrices
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            adj_tnr = torch.FloatTensor(adj_norm).to(device)
            adj_list.append(adj_tnr)
        # ====================
        # Get the prediction result
        adj_est = model(adj_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres # Rescale the edge weights to the original value range
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
        # Evaluate the quality of current prediction operation
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
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
    # ====================
    RMSE_mean_test = np.mean(RMSE_list)
    RMSE_std_test = np.std(RMSE_list, ddof=1)
    MAE_mean_test = np.mean(MAE_list)
    MAE_std_test = np.std(MAE_list, ddof=1)
    print('Test Epoch %d RMSE %f %f MAE %f %f' % (epoch, RMSE_mean_test, RMSE_std_test, MAE_mean_test, MAE_std_test))
    writer.add_scalar(f'Test/RMSE_mean', RMSE_mean_test, epoch)
    writer.add_scalar(f'Test/RMSE_std', RMSE_std_test, epoch)
    writer.add_scalar(f'Test/MAE_mean', MAE_mean_test, epoch)
    writer.add_scalar(f'Test/MAE_std', MAE_std_test, epoch)
    
    mrx_mean = RMSE_mean_val + MAE_mean_val
    if mrx_mean < val_max_rmse:
        val_max_rmse = mrx_mean
        save_model(model, opt, pt_dir, epoch, RMSE_mean_test, MAE_mean_test)
