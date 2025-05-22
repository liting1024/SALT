import os
import random
from tensorboardX import SummaryWriter
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
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ====================
data_name = 'LyonSchool'
num_nodes = 242 # Number of nodes (Level-1 w/ fixed node set)
num_snaps = 157 # Number of snapshots
max_thres = 20 # Threshold for maximum edge weight
noise_dim = 11 # Dimension of noise input
struc_dims = [noise_dim, 32, 16] # Layer configuration of structural encoder
temp_dims = [num_nodes*struc_dims[-1], 1024] # Layer configuration of temporal encoder
dec_dims = [temp_dims[-1], num_nodes*num_nodes] # Layer configuration of decoder
disc_dims = [num_nodes*num_nodes, 512, 256, 64, 1] # Layer configuration of discriminator
win_size = 10 # Window size of historical snapshots
alpha = 10 # Hyper-parameter to adjust the contribution of the MSE loss

# ====================
base_dataset_path = './Dataset'
edge_seq = np.load(f'{base_dataset_path}/{data_name}/{data_name}_edge_seq.npy', allow_pickle=True)
node_set_seq = np.load(f'{base_dataset_path}/{data_name}/{data_name}_node_seq.npy', allow_pickle=True)

# ====================
dropout_rate = 0.2 # Dropout rate
epsilon = 1e-2 # Threshold of zero-refining
c = 0.01 # Threshold of the clipping step (for parameters of discriminator)
num_epochs = 500 # Number of training epochs
num_val_snaps = 10 # Number of validation snapshots
num_test_snaps = 50 # Number of test snapshots
num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

model_name = 'GCN_GAN'
gen_net = GCN_GAN(struc_dims, temp_dims, dec_dims, dropout_rate).to(device) # Generator
disc_net = DiscNet(disc_dims, dropout_rate).to(device) # Discriminator
# ==========
# Define the optimizer
gen_opt = optim.RMSprop(gen_net.parameters(), lr=1e-4, weight_decay=1e-5)
disc_opt = optim.RMSprop(disc_net.parameters(), lr=1e-4, weight_decay=1e-5)

save_base_path = './Exp'
log_dir = f'{save_base_path}/{data_name}/{model_name}/tb/'
pt_dir = f'{save_base_path}/{data_name}/{model_name}/pt/'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(pt_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

val_max_rmse = 1e+9


# ====================
for epoch in range(num_epochs):

    # Training the model
    gen_net.train()
    disc_net.train()
    # ==========
    train_cnt = 0
    disc_loss_list = []
    gen_loss_list = []
    for tau in range(win_size, num_train_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of noise input
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            # ==========
            # Transfer the GNN support to a sparse tensor
            sup = get_gnn_sup(adj_norm)
            sup_sp = sp.sparse.coo_matrix(sup)
            sup_sp = sparse_to_tuple(sup_sp)
            idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
            vals = torch.FloatTensor(sup_sp[1]).to(device)
            sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
            sup_list.append(sup_tnr)
            # =========
            # Generate random noise
            noise_feat = gen_noise(num_nodes, noise_dim)
            noise_feat_tnr = torch.FloatTensor(noise_feat).to(device)
            noise_list.append(noise_feat_tnr)
        # ==========
        edges = edge_seq[tau]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        gnd_norm = gnd/max_thres # Normalize the edge weights (in ground-truth) to [0, 1]
        gnd_tnr = torch.FloatTensor(gnd_norm).to(device)

        for _ in range(1):
            # ====================
            # Train the discriminator
            adj_est = gen_net(sup_list, noise_list)
            disc_real, disc_fake = disc_net(gnd_tnr, adj_est, num_nodes)
            disc_loss = get_disc_loss(disc_real, disc_fake) # Loss of the discriminator
            disc_opt.zero_grad()
            disc_loss.backward()
            disc_opt.step()
            # ===========
            # Clip parameters of discriminator
            for param in disc_net.parameters():
                param.data.clamp_(-c, c)
            # ==========
            # Train the generative network
            adj_est = gen_net(sup_list, noise_list)
            _, disc_fake = disc_net(gnd_tnr, adj_est, num_nodes)
            gen_loss = get_gen_loss(adj_est, gnd_tnr, disc_fake, alpha) # Loss of the generative network
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

        # ====================
        gen_loss_list.append(gen_loss.item())
        disc_loss_list.append(disc_loss.item())
        train_cnt += 1
        if train_cnt % 100 == 0:
            print('-Train %d / %d' % (train_cnt, num_train_snaps))
    gen_loss_mean = np.mean(gen_loss_list)
    disc_loss_mean = np.mean(disc_loss_list)
    print('#%d Train G-Loss %f D-Loss %f' % (epoch, gen_loss_mean, disc_loss_mean))
    writer.add_scalar(f'Train/Loss', gen_loss_mean, epoch)
    
    # ====================
    # Validate the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    EW_KL_list = []
    MR_list = []
    for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of noise input
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres # Normalize the edge weights to [0, 1]
            sup = get_gnn_sup(adj_norm)
            # ==========
            # Transfer the GNN support to a sparse tensor
            sup_sp = sp.sparse.coo_matrix(sup)
            sup_sp = sparse_to_tuple(sup_sp)
            idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
            vals = torch.FloatTensor(sup_sp[1]).to(device)
            sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
            sup_list.append(sup_tnr)
            # ==========
            # Generate random noise
            noise_feat = gen_noise(num_nodes, noise_dim)
            noise_feat_tnr = torch.FloatTensor(noise_feat).to(device)
            noise_list.append(noise_feat_tnr)
        # ====================
        # Get the prediction result
        adj_est = gen_net(sup_list, noise_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres  # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        #adj_est = (adj_est+adj_est.T)/2
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
        EW_KL = get_EW_KL(adj_est, gnd, num_nodes_t)
        MR = get_MR(adj_est, gnd, num_nodes_t)
        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        EW_KL_list.append(EW_KL)
        MR_list.append(MR)
    # ====================
    RMSE_mean_val = np.mean(RMSE_list)
    RMSE_std_val  = np.std(RMSE_list, ddof=1)
    MAE_mean_val  = np.mean(MAE_list)
    MAE_std_val  = np.std(MAE_list, ddof=1)
    EW_KL_mean_val  = np.mean(EW_KL_list)
    EW_KL_std_val  = np.std(EW_KL_list, ddof=1)
    MR_mean_val  = np.mean(MR_list)
    MR_std_val  = np.std(MR_list, ddof=1)
    print('Val Epoch %d RMSE %f %f MAE %f %f EW-KL %f %f MR %f %f'
          % (epoch, RMSE_mean_val , RMSE_std_val , MAE_mean_val , MAE_std_val , EW_KL_mean_val , EW_KL_std_val , MR_mean_val , MR_std_val))
    writer.add_scalar(f'Val/RMSE_mean', RMSE_mean_val, epoch)
    writer.add_scalar(f'Val/RMSE_std', RMSE_std_val, epoch)
    writer.add_scalar(f'Val/MAE_mean', MAE_mean_val, epoch)
    writer.add_scalar(f'Val/MAE_std', MAE_std_val, epoch)
    
    # ====================
    # Test the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list = []
    MAE_list = []
    EW_KL_list = []
    MR_list = []
    for t in range(num_snaps-num_test_snaps, num_snaps):
        # ====================
        sup_list = []  # List of GNN support (tensor)
        noise_list = []
        for k in range(t-win_size, t):
            # ==========
            edges = edge_seq[k]
            adj = get_adj_wei(edges, num_nodes, max_thres)
            adj_norm = adj/max_thres  # Normalize the edge weights to [0, 1]
            sup = get_gnn_sup(adj_norm)
            # ==========
            # Transfer the GNN support to a sparse tensor
            sup_sp = sp.sparse.coo_matrix(sup)
            sup_sp = sparse_to_tuple(sup_sp)
            idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
            vals = torch.FloatTensor(sup_sp[1]).to(device)
            sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
            sup_list.append(sup_tnr)
            # ==========
            # Generate random noise
            noise_feat = gen_noise(num_nodes, noise_dim)
            noise_feat_tnr = torch.FloatTensor(noise_feat).to(device)
            noise_list.append(noise_feat_tnr)
        # ====================
        # Get the prediction result
        adj_est = gen_net(sup_list, noise_list)
        if torch.cuda.is_available():
            adj_est = adj_est.cpu().data.numpy()
        else:
            adj_est = adj_est.data.numpy()
        adj_est *= max_thres  # Rescale the edge weights to the original value range
        # ==========
        # Refine the prediction result
        #adj_est = (adj_est + adj_est.T)/2
        for r in range(num_nodes):
            adj_est[r, r] = 0
        for r in range(num_nodes):
            for c in range(num_nodes):
                if adj_est[r, c] <= epsilon:
                    adj_est[r, c] = 0
        # ====================
        # Get the ground-truth
        edges = edge_seq[t]
        gnd = get_adj_wei(edges, num_nodes, max_thres)
        # ====================
        # Evaluate the prediction result
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
        EW_KL = get_EW_KL(adj_est, gnd, num_nodes_t)
        MR = get_MR(adj_est, gnd, num_nodes_t)
        # ==========
        RMSE_list.append(RMSE)
        MAE_list.append(MAE)
        EW_KL_list.append(EW_KL)
        MR_list.append(MR)

    # ====================
    RMSE_mean_test = np.mean(RMSE_list)
    RMSE_std_test = np.std(RMSE_list, ddof=1)
    MAE_mean_test = np.mean(MAE_list)
    MAE_std_test = np.std(MAE_list, ddof=1)
    EW_KL_mean_test = np.mean(EW_KL_list)
    EW_KL_std_test = np.std(EW_KL_list, ddof=1)
    MR_mean_test = np.mean(MR_list)
    MR_std_test = np.std(MR_list, ddof=1)
    print('Test Epoch %d RMSE %f %f MAE %f %f EW-KL %f %f MR %f %f'
          % (epoch, RMSE_mean_test, RMSE_std_test, MAE_mean_test, MAE_std_test, EW_KL_mean_test, EW_KL_std_test, MR_mean_test, MR_std_test))
    writer.add_scalar(f'Test/RMSE_mean', RMSE_mean_test, epoch)
    writer.add_scalar(f'Test/RMSE_std', RMSE_std_test, epoch)
    writer.add_scalar(f'Test/MAE_mean', MAE_mean_test, epoch)
    writer.add_scalar(f'Test/MAE_std', MAE_std_test, epoch)
    
    mrx_mean = RMSE_mean_test + MAE_mean_test
    if mrx_mean < val_max_rmse:
        val_max_rmse = mrx_mean
        save_model(gen_net, gen_opt, pt_dir, epoch, RMSE_mean_test, MAE_mean_test)