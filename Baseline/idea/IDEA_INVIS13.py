import torch
import torch.optim as optim
from IDEA import GenNet_tanh
from IDEA import DiscNet
from tensorboardX import SummaryWriter
from loss import get_pre_gen_loss
from loss import get_gen_loss
from loss import get_disc_loss
from utils import *
import scipy.sparse
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(100)

# ====================
data_name = 'INVIS13'
num_nodes_glb = int(100)
num_nodes = 100 # Number of nodes (Level-1 w/ fixed node set)
num_snaps = 403 # Number of snapshots
max_thres = 50 # Threshold for maximum edge weight
noise_dim = 5 # Dimensionality of noise input
feat_dim = 5 # Dimensionality of node feature
pos_dim = 5 # Dimensionality of positional embedding
GNN_feat_dim = feat_dim + pos_dim
FEM_dims = [GNN_feat_dim, 64, 32] # Layer configuration of feature extraction module (FEM)
EDM_dims = [(FEM_dims[-1]+noise_dim), 64, 64] # Layer configuration of embedding derivation module (EDM)
EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 64, 64] # Layer configuration of embedding aggregation module (EAM)
disc_dims = [FEM_dims[-1], 32, 16, 8] # Layer configuration of discriminator
save_flag = False # Flag whether to save the trained model (w.r.t. each epoch)

# ====================
alpha = 40 # Parameter to balance the ER loss
beta = 0.1 # Parameter to balance the SDM loss
lambd = 0.4 # Parameter of attentive aligning unit
theta = 0.1 # Decaying factor

# ====================
base_dataset_path = './Dataset'
edge_seq = np.load(f'{base_dataset_path}/{data_name}/{data_name}_edge_seq.npy', allow_pickle=True)
node_set_seq = np.load(f'{base_dataset_path}/{data_name}/{data_name}_node_seq.npy', allow_pickle=True)
mod_seq = np.load(f'{base_dataset_path}/{data_name}/{data_name}_mod_seq.npy', allow_pickle=True)
align_seq_gbl = np.load(f'{base_dataset_path}/{data_name}/{data_name}_align_seq.npy', allow_pickle=True)
# =========
node_map_seq_gbl = [] # Node correspondence map between two successive snapshots
num_nodes_seq_gbl = []
for t in range(num_snaps):
    node_set = node_set_seq[t] # 实际编号
    node_map = get_node_map(node_set)  # 左边是实际编号 右边是虚假编号
    node_map_seq_gbl.append(node_map)
    num_nodes_seq_gbl.append(len(node_set))

# ==========
# Get pos emb
pos_emb = None
for p in range(num_nodes):
    if p==0:
        pos_emb = get_pos_emb(p, pos_dim)
    else:
        pos_emb = np.concatenate((pos_emb, get_pos_emb(p, pos_dim)), axis=0)

# ==========
# Get global node features
feat_gbl = np.load(f'{base_dataset_path}/{data_name}/{data_name}_feat.npy', allow_pickle=True)
feat_lcl_seq = []
for t in range(num_snaps):
    node_set = node_set_seq[t]
    node_idxs = sorted(list(node_set))
    feat_lcl = feat_gbl[node_idxs, :] # 按实际编号做索引节点特征
    pos_feat_lcl = pos_emb[node_idxs, :]
    feat_cat = np.concatenate([feat_lcl, pos_feat_lcl], axis=-1)
    feat_lcl_seq.append(feat_cat)

# ====================
dropout_rate = 0.0 # Dropout rate
win_size = 10 # Window size of historical snapshots
epsilon = 1e-5 # Threshold of the zero-refining
num_pre_epochs = 30 # Number of pre-training epochs
num_epochs = 200 # Number of training epochs
num_test_snaps = 50 # Number of test snapshots
num_val_snaps = 10 # Number of validation snapshots
num_train_snaps = num_snaps-num_test_snaps-num_val_snaps # Number of training snapshots

# ====================
# Define the model
gen_net = GenNet_tanh(FEM_dims, EDM_dims, EAM_dims, dropout_rate).to(device) # Generator
disc_net = DiscNet(FEM_dims, disc_dims, dropout_rate).to(device) # Discriminator
# ==========
# Define the optimizer
pre_gen_opt = optim.Adam(gen_net.parameters(), lr=1e-4, weight_decay=1e-5)
gen_opt = optim.Adam(gen_net.parameters(), lr=1e-4, weight_decay=1e-5)
disc_opt = optim.Adam(disc_net.parameters(), lr=1e-4, weight_decay=1e-5)

# ====================


model_name = 'IDEA'
save_base_path = './Exp'
log_dir = f'{save_base_path}/{data_name}/{model_name}/tb/'
pt_dir = f'{save_base_path}/{data_name}/{model_name}/pt/'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(pt_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

val_max_rmse = 1e+9


for epoch in range(num_pre_epochs):
    # ====================
    # Pre-train the model
    gen_net.train()
    disc_net.train()
    # ==========
    train_cnt = 0
    gen_loss_list = []
    # ==========
    for tau in range(win_size, num_train_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        align_list = [] # List of aligning matrices
        feat_list = [] # List of node feature inputs
        num_nodes_list = [] # List of #nodes in each snapshot
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            num_nodes = num_nodes_seq_gbl[t]
            node_map = node_map_seq_gbl[t]
            adj = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
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
            # Generate the random noise via random projection
            mod_tnr = torch.FloatTensor(mod_seq[t]).to(device)
            rand_mat = rand_proj(num_nodes, noise_dim)
            rand_tnr = torch.FloatTensor(rand_mat).to(device)
            noise_tnr = torch.mm(mod_tnr, rand_tnr)
            noise_list.append(noise_tnr)
            # ==========
            align_tnr = torch.FloatTensor(align_seq_gbl[t]).to(device)
            align_list.append(align_tnr)
            feat_lcl = feat_lcl_seq[t]
            feat_tnr = torch.FloatTensor(feat_lcl).to(device)
            feat_list.append(feat_tnr)
            num_nodes_list.append(num_nodes)
        # ==========
        gnd_list = []
        real_sup_list = []
        for t in range(tau-win_size+1, tau+1):
            # ==========
            edges = edge_seq[t]
            num_nodes = num_nodes_seq_gbl[t]
            node_map = node_map_seq_gbl[t]
            gnd = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
            gnd_norm = gnd/max_thres # Normalize the edge weights to [0, 1]
            gnd_norm += np.eye(num_nodes)
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
            gnd_list.append(gnd_tnr)
            # ==========
            # Transfer the GNN support to a sparse tensor
            sup = get_gnn_sup_woSE(gnd_norm)
            sup_tnr = torch.FloatTensor(sup).to(device)
            real_sup_list.append(sup_tnr)
            # ==========
            if t==tau:
                feat_lcl = feat_lcl_seq[t]
                feat_tnr = torch.FloatTensor(feat_lcl).to(device)
                feat_list.append(feat_tnr)
                num_nodes_list.append(num_nodes)

        # ====================
        # Train the generator
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=False)
        pre_gen_loss = get_pre_gen_loss(adj_est_list, gnd_list, theta)
        pre_gen_opt.zero_grad()
        pre_gen_loss.backward()
        pre_gen_opt.step()

        # ====================
        gen_loss_list.append(pre_gen_loss.item())
        train_cnt += 1
        if train_cnt % 100 == 0:
            print('-Train %d / %d' % (train_cnt, num_train_snaps))
    gen_loss_mean = np.mean(gen_loss_list)
    print('#%d Pre-Train G-Loss %f' % (epoch, gen_loss_mean))
    writer.add_scalar(f'Train/Loss', gen_loss_mean, epoch)

    # ====================
    # Validate the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list_L2 = []
    MAE_list_L2 = []
    MLSD_list_L2 = []
    MR_list_L2 = []
    # ==========
    RMSE_list_L3 = []
    MAE_list_L3 = []
    MLSD_list_L3 = []
    MR_list_L3 = []
    for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        align_list = [] # List of aligning matrices
        feat_list = [] # List of node feature inputs
        num_nodes_list = [] # List of #nodes in each snapshot
        pre_node_map_list = []
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            num_nodes = num_nodes_seq_gbl[t]
            node_map = node_map_seq_gbl[t]
            adj = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
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
            # Generate the random noise via random projection
            mod_tnr = torch.FloatTensor(mod_seq[t]).to(device)
            rand_mat = rand_proj(num_nodes, noise_dim)
            rand_tnr = torch.FloatTensor(rand_mat).to(device)
            noise_tnr = torch.mm(mod_tnr, rand_tnr)
            noise_list.append(noise_tnr)
            # ==========
            align_tnr = torch.FloatTensor(align_seq_gbl[t]).to(device)
            align_list.append(align_tnr)
            feat_lcl = feat_lcl_seq[t]
            feat_tnr = torch.FloatTensor(feat_lcl).to(device)
            feat_list.append(feat_tnr)
            num_nodes_list.append(num_nodes)
            pre_node_map_list.append(node_map)
        # ====================
        # Get the ground-truth
        edges = edge_seq[tau]
        # ==========
        # For L3
        num_nodes_L3 = num_nodes_seq_gbl[tau] # Number of nodes for L3
        node_map_L3 = node_map_seq_gbl[tau]
        gnd_L3 = get_adj_wei_map(edges, node_map_L3, num_nodes_L3, max_thres) # Ground-truth
        feat_lcl = feat_lcl_seq[tau]
        feat_tnr = torch.FloatTensor(feat_lcl).to(device)
        feat_list.append(feat_tnr)
        num_nodes_list.append(num_nodes_L3)
        # ==========
        # For L2
        node_idxs_L2 = get_node_idxs_L2(pre_node_map_list, node_map_L3)
        num_nodes_L2 = len(node_idxs_L2) # Number of nodes for L2
        gnd_L2 = gnd_L3[node_idxs_L2, :]
        gnd_L2 = gnd_L2[:, node_idxs_L2]

        # ====================
        # Get the prediction result
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=True)
        adj_est_L3 = adj_est_list[-1]
        adj_est_L2 = adj_est_L3[node_idxs_L2, :]
        adj_est_L2 = adj_est_L2[:, node_idxs_L2]
        if torch.cuda.is_available():
            adj_est_L2 = adj_est_L2.cpu().data.numpy()
            adj_est_L3 = adj_est_L3.cpu().data.numpy()
        else:
            adj_est_L2 = adj_est_L2.data.numpy()
            adj_est_L3 = adj_est_L3.data.numpy()
        # Rescale the edge weights to the original value range
        adj_est_L2 *= max_thres
        adj_est_L3 *= max_thres
        # ==========
        # Refine the prediction result
        for r in range(num_nodes_L3):
            if r<num_nodes_L2:
                adj_est_L2[r, r] = 0
            adj_est_L3[r, r] = 0
        for r in range(num_nodes_L3):
            for c in range(num_nodes_L3):
                if r<num_nodes_L2 and c<num_nodes_L2:
                    if adj_est_L2[r, c]<=epsilon:
                        adj_est_L2[r, c] = 0
                if adj_est_L3[r, c]<=epsilon:
                    adj_est_L3[r, c] = 0

        # ====================
        # Evaluate the prediction result
        RMSE_L2 = get_RMSE(adj_est_L2, gnd_L2, num_nodes_L2)
        MAE_L2 = get_MAE(adj_est_L2, gnd_L2, num_nodes_L2)
        MLSD_L2 = get_MLSD(adj_est_L2, gnd_L2, num_nodes_L2)
        MR_L2 = get_MR(adj_est_L2, gnd_L2, num_nodes_L2)
        # ==========
        RMSE_list_L2.append(RMSE_L2)
        MAE_list_L2.append(MAE_L2)
        MLSD_list_L2.append(MLSD_L2)
        MR_list_L2.append(MR_L2)
        # ==========
        RMSE_L3 = get_RMSE(adj_est_L3, gnd_L3, num_nodes_L3)
        MAE_L3 = get_MAE(adj_est_L3, gnd_L3, num_nodes_L3)
        MLSD_L3 = get_MLSD(adj_est_L3, gnd_L3, num_nodes_L3)
        MR_L3 = get_MR(adj_est_L3, gnd_L3, num_nodes_L3)
        # ==========
        RMSE_list_L3.append(RMSE_L3)
        MAE_list_L3.append(MAE_L3)
        MLSD_list_L3.append(MLSD_L3)
        MR_list_L3.append(MR_L3)

    # ====================
    RMSE_mean_L2_val = np.mean(RMSE_list_L2)
    RMSE_std_L2_val = np.std(RMSE_list_L2, ddof=1)
    MAE_mean_L2_val = np.mean(MAE_list_L2)
    MAE_std_L2_val = np.std(MAE_list_L2, ddof=1)
    MLSD_mean_L2_val = np.mean(MLSD_list_L2)
    MLSD_std_L2_val = np.std(MLSD_list_L2, ddof=1)
    MR_mean_L2_val = np.mean(MR_list_L2)
    MR_std_L2_val = np.std(MR_list_L2, ddof=1)
    print('(L2) Val Pre-#%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
          % (epoch, RMSE_mean_L2_val, RMSE_std_L2_val, MAE_mean_L2_val, MAE_std_L2_val,
             MLSD_mean_L2_val, MLSD_std_L2_val, MR_mean_L2_val, MR_std_L2_val))
    writer.add_scalar(f'Val/RMSE_mean_L2', RMSE_mean_L2_val, epoch)
    writer.add_scalar(f'Val/RMSE_std_L2', RMSE_std_L2_val, epoch)
    writer.add_scalar(f'Val/MAE_mean_L2', MAE_mean_L2_val, epoch)
    writer.add_scalar(f'Val/MAE_std_L2', MAE_std_L2_val, epoch)
    
    # ==========
    RMSE_mean_L3_val = np.mean(RMSE_list_L3)
    RMSE_std_L3_val = np.std(RMSE_list_L3, ddof=1)
    MAE_mean_L3_val = np.mean(MAE_list_L3)
    MAE_std_L3_val = np.std(MAE_list_L3, ddof=1)
    MLSD_mean_L3_val = np.mean(MLSD_list_L3)
    MLSD_std_L3_val = np.std(MLSD_list_L3, ddof=1)
    MR_mean_L3_val = np.mean(MR_list_L3)
    MR_std_L3_val = np.std(MR_list_L3, ddof=1)
    print('(L3) Val Pre-#%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
          % (epoch, RMSE_mean_L3_val, RMSE_std_L3_val, MAE_mean_L3_val, MAE_std_L3_val,
             MLSD_mean_L3_val, MLSD_std_L3_val, MR_mean_L3_val, MR_std_L3_val))
    writer.add_scalar(f'Val/RMSE_mean_L3', RMSE_mean_L3_val, epoch)
    writer.add_scalar(f'Val/RMSE_std_L3', RMSE_std_L3_val, epoch)
    writer.add_scalar(f'Val/MAE_mean_L3', MAE_mean_L3_val, epoch)
    writer.add_scalar(f'Val/MAE_std_L3', MAE_std_L3_val, epoch)

    # ====================
    # Test the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list_L2 = []
    MAE_list_L2 = []
    MLSD_list_L2 = []
    MR_list_L2 = []
    # ==========
    RMSE_list_L3 = []
    MAE_list_L3 = []
    MLSD_list_L3 = []
    MR_list_L3 = []
    for tau in range(num_snaps-num_test_snaps, num_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        align_list = [] # List of aligning matrices
        feat_list = [] # List of node feature inputs
        num_nodes_list = [] # List of #nodes in each snapshot
        pre_node_map_list = []
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            num_nodes = num_nodes_seq_gbl[t]
            node_map = node_map_seq_gbl[t]
            adj = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
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
            # Generate the random noise via random projection
            mod_tnr = torch.FloatTensor(mod_seq[t]).to(device)
            rand_mat = rand_proj(num_nodes, noise_dim)
            rand_tnr = torch.FloatTensor(rand_mat).to(device)
            noise_tnr = torch.mm(mod_tnr, rand_tnr)
            noise_list.append(noise_tnr)
            # ==========
            align_tnr = torch.FloatTensor(align_seq_gbl[t]).to(device)
            align_list.append(align_tnr)
            feat_lcl = feat_lcl_seq[t]
            feat_tnr = torch.FloatTensor(feat_lcl).to(device)
            feat_list.append(feat_tnr)
            num_nodes_list.append(num_nodes)
            pre_node_map_list.append(node_map)
        # ==========
        # Get the ground-truth
        edges = edge_seq[tau]
        # ==========
        # For L3
        num_nodes_L3 = num_nodes_seq_gbl[tau] # Number of nodes for L3
        node_map_L3 = node_map_seq_gbl[tau]
        gnd_L3 = get_adj_wei_map(edges, node_map_L3, num_nodes_L3, max_thres) # Ground-truth
        feat_lcl = feat_lcl_seq[tau]
        feat_tnr = torch.FloatTensor(feat_lcl).to(device)
        feat_list.append(feat_tnr)
        num_nodes_list.append(num_nodes_L3)
        # ==========
        # For L2
        node_idxs_L2 = get_node_idxs_L2(pre_node_map_list, node_map_L3)
        num_nodes_L2 = len(node_idxs_L2) # Number of nodes for L2
        gnd_L2 = gnd_L3[node_idxs_L2, :]
        gnd_L2 = gnd_L2[:, node_idxs_L2]

        # ==========
        # Get the prediction result
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=True)
        adj_est_L3 = adj_est_list[-1]
        adj_est_L2 = adj_est_L3[node_idxs_L2, :]
        adj_est_L2 = adj_est_L2[:, node_idxs_L2]
        if torch.cuda.is_available():
            adj_est_L2 = adj_est_L2.cpu().data.numpy()
            adj_est_L3 = adj_est_L3.cpu().data.numpy()
        else:
            adj_est_L2 = adj_est_L2.data.numpy()
            adj_est_L3 = adj_est_L3.data.numpy()
        adj_est_L2 *= max_thres # Rescale the edge weights to the original value range
        adj_est_L3 *= max_thres
        # ==========
        # Refine the prediction result
        for r in range(num_nodes_L3):
            if r<num_nodes_L2:
                adj_est_L2[r, r] = 0
            adj_est_L3[r, r] = 0
        for r in range(num_nodes_L3):
            for c in range(num_nodes_L3):
                if r<num_nodes_L2 and c<num_nodes_L2:
                    if adj_est_L2[r, c]<=epsilon:
                        adj_est_L2[r, c] = 0
                if adj_est_L3[r, c]<=epsilon:
                    adj_est_L3[r, c] = 0

        # ====================
        # Evaluate the prediction result
        RMSE_L2_test = get_RMSE(adj_est_L2, gnd_L2, num_nodes_L2)
        MAE_L2_test = get_MAE(adj_est_L2, gnd_L2, num_nodes_L2)
        MLSD_L2_test = get_MLSD(adj_est_L2, gnd_L2, num_nodes_L2)
        MR_L2_test = get_MR(adj_est_L2, gnd_L2, num_nodes_L2)
        # ==========
        RMSE_list_L2.append(RMSE_L2_test)
        MAE_list_L2.append(MAE_L2_test)
        MLSD_list_L2.append(MLSD_L2_test)
        MR_list_L2.append(MR_L2_test)
        # ==========
        RMSE_L3_test = get_RMSE(adj_est_L3, gnd_L3, num_nodes_L3)
        MAE_L3_test = get_MAE(adj_est_L3, gnd_L3, num_nodes_L3)
        MLSD_L3_test = get_MLSD(adj_est_L3, gnd_L3, num_nodes_L3)
        MR_L3_test = get_MR(adj_est_L3, gnd_L3, num_nodes_L3)
        # ==========
        RMSE_list_L3.append(RMSE_L3_test)
        MAE_list_L3.append(MAE_L3_test)
        MLSD_list_L3.append(MLSD_L3_test)
        MR_list_L3.append(MR_L3_test)

    # ====================
    RMSE_mean_L2_test = np.mean(RMSE_list_L2)
    RMSE_std_L2_test = np.std(RMSE_list_L2, ddof=1)
    MAE_mean_L2_test = np.mean(MAE_list_L2)
    MAE_std_L2_test = np.std(MAE_list_L2, ddof=1)
    MLSD_mean_L2_test = np.mean(MLSD_list_L2)
    MLSD_std_L2_test = np.std(MLSD_list_L2, ddof=1)
    MR_mean_L2_test = np.mean(MR_list_L2)
    MR_std_L2_test = np.std(MR_list_L2, ddof=1)
    print('(L2) Test Pre-#%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
            % (epoch, RMSE_mean_L2_test, RMSE_std_L2_test, MAE_mean_L2_test, MAE_std_L2_test,
               MLSD_mean_L2_test, MLSD_std_L2_test, MR_mean_L2_test, MR_std_L2_test))

    writer.add_scalar(f'Test/RMSE_mean_L2', RMSE_mean_L2_test, epoch)
    writer.add_scalar(f'Test/RMSE_std_L2', RMSE_std_L2_test, epoch)
    writer.add_scalar(f'Test/MAE_mean_L2', MAE_mean_L2_test, epoch)
    writer.add_scalar(f'Test/MAE_std_L2', MAE_std_L2_test, epoch)
    
    # ==========
    RMSE_mean_L3_test = np.mean(RMSE_list_L3)
    RMSE_std_L3_test = np.std(RMSE_list_L3, ddof=1)
    MAE_mean_L3_test = np.mean(MAE_list_L3)
    MAE_std_L3_test = np.std(MAE_list_L3, ddof=1)
    MLSD_mean_L3_test = np.mean(MLSD_list_L3)
    MLSD_std_L3_test = np.std(MLSD_list_L3, ddof=1)
    MR_mean_L3_test = np.mean(MR_list_L3)
    MR_std_L3_test = np.std(MR_list_L3, ddof=1)
    print('(L3) Test Pre-#%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
          % (epoch, RMSE_mean_L3_test, RMSE_std_L3_test, MAE_mean_L3_test, MAE_std_L3_test,
             MLSD_mean_L3_test, MLSD_std_L3_test, MR_mean_L3_test, MR_std_L3_test))
    writer.add_scalar(f'Test/RMSE_mean_L3', RMSE_mean_L3_test, epoch)
    writer.add_scalar(f'Test/RMSE_std_L3', RMSE_std_L3_test, epoch)
    writer.add_scalar(f'Test/MAE_mean_L3', MAE_mean_L3_test, epoch)
    writer.add_scalar(f'Test/MAE_std_L3', MAE_std_L3_test, epoch)
    
    mrx_mean = RMSE_mean_L3_val + MAE_mean_L3_val
    if mrx_mean < val_max_rmse:
        val_max_rmse = mrx_mean
        save_model(gen_net, gen_opt, pt_dir, epoch, RMSE_mean_L3_test, MAE_mean_L3_test)


# ====================
# Joint optimization of the generator & discriminator
for epoch in range(num_epochs):
    # ====================
    # Train the model
    gen_net.train()
    disc_net.train()
    # ==========
    train_cnt = 0
    gen_loss_list = []
    disc_loss_list = []
    # ==========
    for tau in range(win_size, num_train_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        align_list = [] # List of aligning matrices
        feat_list = [] # List of node feature inputs
        num_nodes_list = [] # List of #nodes in each snapshot
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            num_nodes = num_nodes_seq_gbl[t]
            node_map = node_map_seq_gbl[t]
            adj = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
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
            # Generate the random noise via random projection
            mod_tnr = torch.FloatTensor(mod_seq[t]).to(device)
            rand_mat = rand_proj(num_nodes, noise_dim)
            rand_tnr = torch.FloatTensor(rand_mat).to(device)
            noise_tnr = torch.mm(mod_tnr, rand_tnr)
            noise_list.append(noise_tnr)
            # ==========
            align_tnr = torch.FloatTensor(align_seq_gbl[t]).to(device)
            align_list.append(align_tnr)
            feat_lcl = feat_lcl_seq[t]
            feat_tnr = torch.FloatTensor(feat_lcl).to(device)
            feat_list.append(feat_tnr)
            num_nodes_list.append(num_nodes)
        # ==========
        gnd_list = []
        real_sup_list = []
        for t in range(tau-win_size+1, tau+1):
            # ==========
            edges = edge_seq[t]
            num_nodes = num_nodes_seq_gbl[t]
            node_map = node_map_seq_gbl[t]
            gnd = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
            gnd_norm = gnd/max_thres # Normalize the edge weights to [0, 1]
            gnd_norm += np.eye(num_nodes)
            gnd_tnr = torch.FloatTensor(gnd_norm).to(device)
            gnd_list.append(gnd_tnr)
            # ==========
            # Transfer the GNN support to a sparse tensor
            sup = get_gnn_sup_woSE(gnd_norm)
            sup_tnr = torch.FloatTensor(sup).to(device)
            real_sup_list.append(sup_tnr)
            # ==========
            if t==tau:
                feat_lcl = feat_lcl_seq[t]
                feat_tnr = torch.FloatTensor(feat_lcl).to(device)
                feat_list.append(feat_tnr)
                num_nodes_list.append(num_nodes)

        # ====================
        # Train the discriminator
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=False)
        disc_real_list, disc_fake_list = disc_net(real_sup_list, feat_list[1:], adj_est_list, feat_list[1:])
        disc_loss = get_disc_loss(disc_real_list, disc_fake_list, theta)
        disc_opt.zero_grad()
        disc_loss.backward()
        disc_opt.step()
        # ==========
        # Train the generator
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=False)
        _, disc_fake_list = disc_net(real_sup_list, feat_list[1:], adj_est_list, feat_list[1:])
        gen_loss = get_gen_loss(adj_est_list, gnd_list, disc_fake_list, max_thres, alpha, beta, theta)
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
    writer.add_scalar(f'Train/Loss', gen_loss_mean, epoch+num_pre_epochs)
    # ====================
    # Save the trained model (w.r.t. current epoch)
    if save_flag:
        torch.save(gen_net, 'pt/%s_IDEA_gen_%d.pkl' % (data_name, epoch))
        torch.save(disc_net, 'pt/%s_IDEA_disc_%d.pkl' % (data_name, epoch))

    # ====================
    # Validate the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list_L2 = []
    MAE_list_L2 = []
    MLSD_list_L2 = []
    MR_list_L2 = []
    # ==========
    RMSE_list_L3 = []
    MAE_list_L3 = []
    MLSD_list_L3 = []
    MR_list_L3 = []
    for tau in range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        align_list = [] # List of aligning matrices
        feat_list = [] # List of node feature inputs
        num_nodes_list = [] # List of #nodes in each snapshot
        pre_node_map_list = []
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            num_nodes = num_nodes_seq_gbl[t]
            node_map = node_map_seq_gbl[t]
            adj = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
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
            # Generate the random noise via random projection
            mod_tnr = torch.FloatTensor(mod_seq[t]).to(device)
            rand_mat = rand_proj(num_nodes, noise_dim)
            rand_tnr = torch.FloatTensor(rand_mat).to(device)
            noise_tnr = torch.mm(mod_tnr, rand_tnr)
            noise_list.append(noise_tnr)
            # ==========
            align_tnr = torch.FloatTensor(align_seq_gbl[t]).to(device)
            align_list.append(align_tnr)
            feat_lcl = feat_lcl_seq[t]
            feat_tnr = torch.FloatTensor(feat_lcl).to(device)
            feat_list.append(feat_tnr)
            num_nodes_list.append(num_nodes)
            pre_node_map_list.append(node_map)
        # ==========
        # Get ground-truth
        edges = edge_seq[tau]
        # ==========
        # For L3
        num_nodes_L3 = num_nodes_seq_gbl[tau] # Number of nodes for L3
        node_map_L3 = node_map_seq_gbl[tau]
        gnd_L3 = get_adj_wei_map(edges, node_map_L3, num_nodes_L3, max_thres) # Ground-truth
        feat_lcl = feat_lcl_seq[tau]
        feat_tnr = torch.FloatTensor(feat_lcl).to(device)
        feat_list.append(feat_tnr)
        num_nodes_list.append(num_nodes_L3)
        # ==========
        # For L2
        node_idxs_L2 = get_node_idxs_L2(pre_node_map_list, node_map_L3)
        num_nodes_L2 = len(node_idxs_L2) # Number of nodes for L2
        gnd_L2 = gnd_L3[node_idxs_L2, :]
        gnd_L2 = gnd_L2[:, node_idxs_L2]

        # ====================
        # Get prediction result
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=True)
        adj_est_L3 = adj_est_list[-1]
        adj_est_L2 = adj_est_L3[node_idxs_L2, :]
        adj_est_L2 = adj_est_L2[:, node_idxs_L2]
        if torch.cuda.is_available():
            adj_est_L2 = adj_est_L2.cpu().data.numpy()
            adj_est_L3 = adj_est_L3.cpu().data.numpy()
        else:
            adj_est_L2 = adj_est_L2.data.numpy()
            adj_est_L3 = adj_est_L3.data.numpy()
        # Rescale the edge weights to the original value range
        adj_est_L2 *= max_thres
        adj_est_L3 *= max_thres
        # ==========
        # Refine the prediction result
        for r in range(num_nodes_L3):
            if r<num_nodes_L2:
                adj_est_L2[r, r] = 0
            adj_est_L3[r, r] = 0
        for r in range(num_nodes_L3):
            for c in range(num_nodes_L3):
                if r<num_nodes_L2 and c<num_nodes_L2:
                    if adj_est_L2[r, c]<=epsilon:
                        adj_est_L2[r, c] = 0
                if adj_est_L3[r, c]<=epsilon:
                    adj_est_L3[r, c] = 0

        # ====================
        # Evaluate the prediction result
        RMSE_L2 = get_RMSE(adj_est_L2, gnd_L2, num_nodes_L2)
        MAE_L2 = get_MAE(adj_est_L2, gnd_L2, num_nodes_L2)
        MLSD_L2 = get_MLSD(adj_est_L2, gnd_L2, num_nodes_L2)
        MR_L2 = get_MR(adj_est_L2, gnd_L2, num_nodes_L2)
        # ==========
        RMSE_list_L2.append(RMSE_L2)
        MAE_list_L2.append(MAE_L2)
        MLSD_list_L2.append(MLSD_L2)
        MR_list_L2.append(MR_L2)
        # ==========
        RMSE_L3 = get_RMSE(adj_est_L3, gnd_L3, num_nodes_L3)
        MAE_L3 = get_MAE(adj_est_L3, gnd_L3, num_nodes_L3)
        MLSD_L3 = get_MLSD(adj_est_L3, gnd_L3, num_nodes_L3)
        MR_L3 = get_MR(adj_est_L3, gnd_L3, num_nodes_L3)
        # ==========
        RMSE_list_L3.append(RMSE_L3)
        MAE_list_L3.append(MAE_L3)
        MLSD_list_L3.append(MLSD_L3)
        MR_list_L3.append(MR_L3)

    # ====================
    RMSE_mean_L2_val = np.mean(RMSE_list_L2)
    RMSE_std_L2_val = np.std(RMSE_list_L2, ddof=1)
    MAE_mean_L2_val = np.mean(MAE_list_L2)
    MAE_std_L2_val = np.std(MAE_list_L2, ddof=1)
    MLSD_mean_L2_val = np.mean(MLSD_list_L2)
    MLSD_std_L2_val = np.std(MLSD_list_L2, ddof=1)
    MR_mean_L2_val = np.mean(MR_list_L2)
    MR_std_L2_val = np.std(MR_list_L2, ddof=1)
    print('(L2) Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
          % (epoch, RMSE_mean_L2_val, RMSE_std_L2_val, MAE_mean_L2_val, MAE_std_L2_val,
             MLSD_mean_L2_val, MLSD_std_L2_val, MR_mean_L2_val, MR_std_L2_val))
    writer.add_scalar(f'Val/RMSE_mean_L2', RMSE_mean_L2_val, epoch+num_pre_epochs)
    writer.add_scalar(f'Val/RMSE_std_L2', RMSE_std_L2_val, epoch+num_pre_epochs)
    writer.add_scalar(f'Val/MAE_mean_L2', MAE_mean_L2_val, epoch+num_pre_epochs)
    writer.add_scalar(f'Val/MAE_std_L2', MAE_std_L2_val, epoch+num_pre_epochs)
    

    RMSE_mean_L3_val = np.mean(RMSE_list_L3)
    RMSE_std_L3_val = np.std(RMSE_list_L3, ddof=1)
    MAE_mean_L3_val = np.mean(MAE_list_L3)
    MAE_std_L3_val = np.std(MAE_list_L3, ddof=1)
    MLSD_mean_L3_val = np.mean(MLSD_list_L3)
    MLSD_std_L3_val = np.std(MLSD_list_L3, ddof=1)
    MR_mean_L3_val = np.mean(MR_list_L3)
    MR_std_L3_val = np.std(MR_list_L3, ddof=1)
    print('(L3) Val #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
          % (epoch, RMSE_mean_L3_val, RMSE_std_L3_val, MAE_mean_L3_val, MAE_std_L3_val,
             MLSD_mean_L3_val, MLSD_std_L3_val, MR_mean_L3_val, MR_std_L3_val))
    writer.add_scalar(f'Val/RMSE_mean_L3', RMSE_mean_L3_val, epoch+num_pre_epochs)
    writer.add_scalar(f'Val/RMSE_std_L3', RMSE_std_L3_val, epoch+num_pre_epochs)
    writer.add_scalar(f'Val/MAE_mean_L3', MAE_mean_L3_val, epoch+num_pre_epochs)
    writer.add_scalar(f'Val/MAE_std_L3', MAE_std_L3_val, epoch+num_pre_epochs)


    # ====================
    # Test the model
    gen_net.eval()
    disc_net.eval()
    # ==========
    RMSE_list_L2 = []
    MAE_list_L2 = []
    MLSD_list_L2 = []
    MR_list_L2 = []
    # ==========
    RMSE_list_L3 = []
    MAE_list_L3 = []
    MLSD_list_L3 = []
    MR_list_L3 = []
    for tau in range(num_snaps-num_test_snaps, num_snaps):
        # ====================
        sup_list = [] # List of GNN support (tensor)
        noise_list = [] # List of random noise inputs
        align_list = [] # List of aligning matrices
        feat_list = [] # List of node feature inputs
        num_nodes_list = [] # List of #nodes in each snapshot
        pre_node_map_list = []
        for t in range(tau-win_size, tau):
            # ==========
            edges = edge_seq[t]
            num_nodes = num_nodes_seq_gbl[t]
            node_map = node_map_seq_gbl[t]
            adj = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
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
            # Generate the random noise via random projection
            mod_tnr = torch.FloatTensor(mod_seq[t]).to(device)
            rand_mat = rand_proj(num_nodes, noise_dim)
            rand_tnr = torch.FloatTensor(rand_mat).to(device)
            noise_tnr = torch.mm(mod_tnr, rand_tnr)
            noise_list.append(noise_tnr)
            # ==========
            align_tnr = torch.FloatTensor(align_seq_gbl[t]).to(device)
            align_list.append(align_tnr)
            feat_lcl = feat_lcl_seq[t]
            feat_tnr = torch.FloatTensor(feat_lcl).to(device)
            feat_list.append(feat_tnr)
            num_nodes_list.append(num_nodes)
            pre_node_map_list.append(node_map)
        # ==========
        # Get the ground-truth
        edges = edge_seq[tau]
        # ==========
        # For L3
        num_nodes_L3 = num_nodes_seq_gbl[tau] # Number of nodes for L3
        node_map_L3 = node_map_seq_gbl[tau]
        gnd_L3 = get_adj_wei_map(edges, node_map_L3, num_nodes_L3, max_thres) # Ground-truth
        feat_lcl = feat_lcl_seq[tau]
        feat_tnr = torch.FloatTensor(feat_lcl).to(device)
        feat_list.append(feat_tnr)
        num_nodes_list.append(num_nodes_L3)
        # ==========
        # For L2
        node_idxs_L2 = get_node_idxs_L2(pre_node_map_list, node_map_L3)
        num_nodes_L2 = len(node_idxs_L2) # Number of nodes for L2
        gnd_L2 = gnd_L3[node_idxs_L2, :]
        gnd_L2 = gnd_L2[:, node_idxs_L2]

        # ==========
        # Get the prediction result
        adj_est_list = gen_net(sup_list, feat_list, noise_list, align_list, num_nodes_list, lambd, pred_flag=True)
        adj_est_L3 = adj_est_list[-1]
        adj_est_L2 = adj_est_L3[node_idxs_L2, :]
        adj_est_L2 = adj_est_L2[:, node_idxs_L2]
        if torch.cuda.is_available():
            adj_est_L2 = adj_est_L2.cpu().data.numpy()
            adj_est_L3 = adj_est_L3.cpu().data.numpy()
        else:
            adj_est_L2 = adj_est_L2.data.numpy()
            adj_est_L3 = adj_est_L3.data.numpy()
        # Rescale the edge weights to the original value range
        adj_est_L2 *= max_thres
        adj_est_L3 *= max_thres
        # ==========
        # Refine the prediction result
        for r in range(num_nodes_L3):
            if r<num_nodes_L2:
                adj_est_L2[r, r] = 0
            adj_est_L3[r, r] = 0
        for r in range(num_nodes_L3):
            for c in range(num_nodes_L3):
                if r<num_nodes_L2 and c<num_nodes_L2:
                    if adj_est_L2[r, c]<=epsilon:
                        adj_est_L2[r, c] = 0
                if adj_est_L3[r, c]<=epsilon:
                    adj_est_L3[r, c] = 0

        # ====================
        # Evaluate the prediction result
        RMSE_L2 = get_RMSE(adj_est_L2, gnd_L2, num_nodes_L2)
        MAE_L2 = get_MAE(adj_est_L2, gnd_L2, num_nodes_L2)
        MLSD_L2 = get_MLSD(adj_est_L2, gnd_L2, num_nodes_L2)
        MR_L2 = get_MR(adj_est_L2, gnd_L2, num_nodes_L2)
        # ==========
        RMSE_list_L2.append(RMSE_L2)
        MAE_list_L2.append(MAE_L2)
        MLSD_list_L2.append(MLSD_L2)
        MR_list_L2.append(MR_L2)
        # ==========
        RMSE_L3 = get_RMSE(adj_est_L3, gnd_L3, num_nodes_L3)
        MAE_L3 = get_MAE(adj_est_L3, gnd_L3, num_nodes_L3)
        MLSD_L3 = get_MLSD(adj_est_L3, gnd_L3, num_nodes_L3)
        MR_L3 = get_MR(adj_est_L3, gnd_L3, num_nodes_L3)
        # ==========
        RMSE_list_L3.append(RMSE_L3)
        MAE_list_L3.append(MAE_L3)
        MLSD_list_L3.append(MLSD_L3)
        MR_list_L3.append(MR_L3)

    # ====================
    RMSE_mean_L2_test = np.mean(RMSE_list_L2)
    RMSE_std_L2_test = np.std(RMSE_list_L2, ddof=1)
    MAE_mean_L2_test = np.mean(MAE_list_L2)
    MAE_std_L2_test = np.std(MAE_list_L2, ddof=1)
    MLSD_mean_L2_test = np.mean(MLSD_list_L2)
    MLSD_std_L2_test = np.std(MLSD_list_L2, ddof=1)
    MR_mean_L2_test = np.mean(MR_list_L2)
    MR_std_L2_test = np.std(MR_list_L2, ddof=1)
    print('(L2) Test #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f'
          % (epoch, RMSE_mean_L2_test, RMSE_std_L2_test, MAE_mean_L2_test, MAE_std_L2_test,
             MLSD_mean_L2_test, MLSD_std_L2_test, MR_mean_L2_test, MR_std_L2_test))
    writer.add_scalar(f'Test/RMSE_mean_L2', RMSE_mean_L2_test, epoch+num_pre_epochs)
    writer.add_scalar(f'Test/RMSE_std_L2', RMSE_std_L2_test, epoch+num_pre_epochs)
    writer.add_scalar(f'Test/MAE_mean_L2', MAE_mean_L2_test, epoch+num_pre_epochs)
    writer.add_scalar(f'Test/MAE_std_L2', MAE_std_L2_test, epoch+num_pre_epochs)
    

    RMSE_mean_L3_test = np.mean(RMSE_list_L3)
    RMSE_std_L3_test= np.std(RMSE_list_L3, ddof=1)
    MAE_mean_L3_test = np.mean(MAE_list_L3)
    MAE_std_L3_test = np.std(MAE_list_L3, ddof=1)
    MLSD_mean_L3_test = np.mean(MLSD_list_L3)
    MLSD_std_L3_test = np.std(MLSD_list_L3, ddof=1)
    MR_mean_L3_test = np.mean(MR_list_L3)
    MR_std_L3_test = np.std(MR_list_L3, ddof=1)
    print('(L3) Test #%d RMSE %f %f MAE %f %f MLSD %f %f MR %f %f\n'
          % (epoch, RMSE_mean_L3_test, RMSE_std_L3_test, MAE_mean_L3_test, MAE_std_L3_test,
             MLSD_mean_L3_test, MLSD_std_L3_test, MR_mean_L3_test, MR_std_L3_test))
    
    writer.add_scalar(f'Test/RMSE_mean_L3', RMSE_mean_L3_test, epoch+num_pre_epochs)
    writer.add_scalar(f'Test/RMSE_std_L3', RMSE_std_L3_test, epoch+num_pre_epochs)
    writer.add_scalar(f'Test/MAE_mean_L3', MAE_mean_L3_test, epoch+num_pre_epochs)
    writer.add_scalar(f'Test/MAE_std_L3', MAE_std_L3_test, epoch+num_pre_epochs)
    
    mrx_mean = RMSE_mean_L3_val + MAE_mean_L3_val
    if mrx_mean < val_max_rmse:
        val_max_rmse = mrx_mean
        save_model(gen_net, gen_opt, pt_dir, epoch+num_pre_epochs, RMSE_mean_L3_test, MAE_mean_L3_test)
