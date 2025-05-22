import torch
import numpy as np
import scipy as sp


class Dataset():
    def __init__(self, params, base_dataset_path):
        self.params = params
        self.data_name = params.data_name
        self.win_size = params.win_size
        self.pred_size = params.pred_size
        self.device = params.device
        if hasattr(params, 'is_baseline'):
            self.is_baseline = params.is_baseline
        else:
            self.is_baseline = False

        self.load_info()
        self.num_val_snaps = 10
        self.num_test_snaps = 50
        self.num_train_snaps = self.num_snaps - self.num_test_snaps - self.num_val_snaps
        
        self.base_dataset_path = base_dataset_path
        L2_dataset = ['IoT', 'LyonSchool', 'INVIS13', 'INVIS15', 'Thiers13']
        
        if self.data_name in L2_dataset :
            self.edge_seq = np.load(f'{self.base_dataset_path}/{self.data_name}/{self.data_name}_edge_seq.npy', allow_pickle=True)
            self.mod_seq = np.load(f'{self.base_dataset_path}/{self.data_name}/{self.data_name}_mod_seq_all.npy', allow_pickle=True) 
            
        else:
            self.edge_seq = np.load(f'{self.base_dataset_path}/{self.data_name}/{self.data_name}_edge_seq.npy', allow_pickle=True)
            self.mod_seq = np.load(f'{self.base_dataset_path}/{self.data_name}/{self.data_name}_mod_seq.npy', allow_pickle=True)
        
        self.align_list = []
        self.num_nodes_list = []
        self.node_map_seq_gbl = [] 
        self.num_nodes_seq_gbl = []
        self.feat_lcl_seq = []

        if self.data_name == 'DC':
            self.load_attributed_graph(is_baseline=self.is_baseline)
        elif self.data_name == 'HMob' or self.data_name == "T-Drive":
            self.load_topology()
        else:
            self.load_l2_dataset(is_baseline=self.is_baseline)


    def load_info(self):
        if self.data_name == 'T-Drive':
            self.data_level = 1
            self.num_nodes = 1279
            self.num_snaps = 300
            self.max_thres = 5000
            self.noise_dim = 512
            self.feat_dim = 32
            self.pos_dim = 256

        elif self.data_name == 'DC':
            self.data_level = 1
            self.num_nodes = 128
            self.num_snaps = 700
            self.max_thres = 5000
            self.noise_dim = 100
            self.feat_dim = 32
            self.pos_dim = 32
 
        elif self.data_name == 'HMob':
            self.data_level = 1
            self.num_nodes = 92
            self.num_snaps = 500
            self.max_thres = 250
            self.noise_dim = 64
            self.pos_dim = 32

        elif self.data_name == 'IoT':
            self.data_level = 2
            self.num_nodes = 668
            self.num_snaps = 144
            self.max_thres = 1024
            self.noise_dim =  48
            self.feat_dim =  32
            self.pos_dim =  32

        elif self.data_name == 'LyonSchool':
            self.data_level = 2
            self.num_nodes = 242
            self.num_snaps = 157 
            self.max_thres = 20
            self.noise_dim =  32 
            self.feat_dim =  11 
            self.pos_dim =  32 
        
        elif self.data_name == 'INVIS13':
            self.data_level = 2
            self.num_nodes = 100
            self.num_snaps = 403
            self.max_thres = 50
            self.noise_dim =  5 
            self.feat_dim =  5 
            self.pos_dim =  5 
        
        elif self.data_name == 'INVIS15':
            self.data_level = 2
            self.num_nodes = 232
            self.num_snaps = 431
            self.max_thres = 50
            self.noise_dim =  5 
            self.feat_dim =  12 
            self.pos_dim =  12 
        
        elif self.data_name == 'Thiers13':
            self.data_level = 2
            self.num_nodes = 329
            self.num_snaps = 179
            self.max_thres = 50
            self.noise_dim =  16 
            self.feat_dim =  9 
            self.pos_dim =  16 

        else:
            raise ValueError('Invalid data name')
        return self


    def load_attributed_graph(self, is_baseline=False):
        feat = np.load(f'{self.base_dataset_path}/{self.data_name}/{self.data_name}_feat.npy', allow_pickle=True)
        feat_tnr = torch.FloatTensor(feat)
        pos_tnr = self.load_topology(return_pos=True)
        if is_baseline:
            self.feat_tnr = feat_tnr
        else:
            self.feat_tnr = torch.cat((feat_tnr, pos_tnr), dim=1)


    def load_topology(self, return_pos = False):
        pos_embs = [get_pos_emb(p, self.pos_dim) for p in range(self.num_nodes)]
        pos_emb = np.concatenate(pos_embs, axis=0)
        if return_pos:
            return torch.FloatTensor(pos_emb)
        else:
            self.feat_tnr = torch.FloatTensor(pos_emb)


    def load_l2_dataset(self, is_baseline=False):
        feat_gbl = np.load(f'{self.base_dataset_path}/{self.data_name}/{self.data_name}_feat.npy', allow_pickle=True)
        self.node_set_seq = np.load(f'{self.base_dataset_path}/{self.data_name}/{self.data_name}_node_seq.npy', allow_pickle=True)
        self.align_seq_gbl = np.load(f'{self.base_dataset_path}/{self.data_name}/{self.data_name}_align_seq.npy', allow_pickle=True)

        for t in range(self.num_snaps):
            node_set = self.node_set_seq[t]
            node_map = get_node_map(node_set)
            self.node_map_seq_gbl.append(node_map)
            self.num_nodes_seq_gbl.append(len(node_set))

        pos_emb = self.load_topology(return_pos=True)
        
        for t in range(self.num_snaps):
            node_idxs = sorted(list(self.node_set_seq[t]))
            feat_lcl = feat_gbl[node_idxs, :]
            pos_feat_lcl = pos_emb[node_idxs, :]
            if is_baseline:
                feat_cat = feat_lcl[:32] if self.data_name == 'IoT' else feat_lcl
            else:
                feat_cat = feat_lcl if self.data_name == 'IoT' else np.concatenate([feat_lcl, pos_feat_lcl], axis=-1)
            self.feat_lcl_seq.append(torch.FloatTensor(feat_cat))

    
    def get_win_data(self, start, end):
        sup_list = [] #
        noise_list = []
        feat_list = []
        
        for t in range(start, end):
            edges = self.edge_seq[t]
            if self.data_level == 2:  
                num_nodes = self.num_nodes
                node_map = self.node_map_seq_gbl[t]
                adj = get_adj_wei(edges, num_nodes, self.max_thres)
                re_map = {soft: hard for hard, soft in node_map.items()}
                feat_tnr = np.zeros((num_nodes, self.feat_lcl_seq[t].shape[1]))
                for i, feat in enumerate(self.feat_lcl_seq[t]):
                    feat_tnr[re_map[i]] = feat
                feat_tnr = torch.FloatTensor(feat_tnr).to(self.device)
                mod_tnr = torch.FloatTensor(self.mod_seq[t]).to(self.device)
            else:
                num_nodes = self.num_nodes
                feat_tnr = torch.FloatTensor(self.feat_tnr).to(self.device)
                adj = get_adj_wei(edges, num_nodes, self.max_thres)
                if self.data_name == "T-Drive":
                    mod_tnr = torch.FloatTensor(self.mod_seq[t][0]).to(self.device)
                else:
                    mod_tnr = torch.FloatTensor(self.mod_seq[t]).to(self.device)
                    
            adj_norm = adj/self.max_thres
            feat_list.append(feat_tnr)
            sup = get_gnn_sup(adj_norm)
            sup_sp = sp.sparse.coo_matrix(sup)
            sup_sp = sparse_to_tuple(sup_sp)
            idxs = torch.LongTensor(sup_sp[0].astype(float)).to(self.device)
            vals = torch.FloatTensor(sup_sp[1]).to(self.device)
            sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(self.device)
            sup_list.append(sup_tnr)
            if self.data_name == "T-Drive":
                noise_tnr = mod_tnr
            else:
                rand_mat = rand_proj(num_nodes, self.noise_dim)
                rand_tnr = torch.FloatTensor(rand_mat).to(self.device)
                noise_tnr = torch.mm(mod_tnr, rand_tnr)
            noise_list.append(noise_tnr)
        return sup_list, noise_list, feat_list


    def get_win_gnd_data(self, start, end):
        gnd_list = []
        for t in range(start, end):
            edges = self.edge_seq[t]
            num_nodes = self.num_nodes
            gnd = get_adj_wei(edges, num_nodes, self.max_thres)
            gnd_norm = gnd/self.max_thres
            gnd_norm += np.eye(num_nodes)
            gnd_tnr = torch.FloatTensor(gnd_norm).to(self.device)
            gnd_list.append(gnd_tnr)
        return gnd_list


def get_pos_emb(pos, hid_dim):
    pos_emb = np.zeros((1, hid_dim))
    for i in range(hid_dim):
        if i%2==0:
            pos_emb[0, i] = np.sin(pos/(10000**(i/hid_dim)))
        else:
            pos_emb[0, i] = np.cos(pos/(10000**((i-1)/hid_dim)))
    return pos_emb


def get_node_map(node_set):
    node_idxs = sorted(list(node_set))
    node_map = {}
    node_cnt = 0
    for node_idx in node_idxs:
        node_map[node_idx] = node_cnt
        node_cnt += 1
    return node_map


def get_adj_wei(edges, num_nodes, max_wei):
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        wei = float(edges[i][2])
        if wei>max_wei:
            wei = max_wei
        adj[src, dst] = wei
        adj[dst, src] = wei
    for i in range(num_nodes):
        adj[i, i] = 0
    return adj

def get_gnn_sup(adj):
    num_nodes, _ = adj.shape
    adj = adj + np.eye(num_nodes)
    degs = np.sqrt(np.sum(adj, axis=1))
    sup = adj
    for i in range(num_nodes):
        sup[i, :] /= degs[i]
    for j in range(num_nodes):
        sup[:, j] /= degs[j]
    return sup


def rand_proj(num_nodes, hid_dim):
    rand_mat = np.random.normal(0, 1.0/np.sqrt(hid_dim), (num_nodes, hid_dim))
    temp_l = np.linalg.norm(rand_mat, axis=1)
    for i in range(hid_dim):
        temp_row = rand_mat[:, i]
        for j in range(i-1):
            temp_j = rand_mat[:, j]
            temp_product = temp_row.T.dot(temp_j)/(temp_l[j]**2)
            temp_row -= temp_product*temp_j
        temp_row *= temp_l[i]/np.sqrt(temp_row.T.dot(temp_row))
        rand_mat[:, i] = temp_row

    return rand_mat


def get_adj_wei_map(edges, node_map, num_nodes, max_thres):
    adj = np.zeros((num_nodes, num_nodes))
    num_edges = len(edges)
    for i in range(num_edges):
        src = int(edges[i][0])
        dst = int(edges[i][1])
        if (src not in node_map) or (dst not in node_map):
            continue
        src_idx = node_map[src]
        dst_idx = node_map[dst]
        wei = float(edges[i][2])
        if wei>max_thres:
            wei = max_thres
        adj[src_idx, dst_idx] = wei
        adj[dst_idx, src_idx] = wei
    for i in range(num_nodes):
        adj[i, i] = 0
    return adj


def get_node_map(node_set):
    node_idxs = sorted(list(node_set))
    node_map = {}
    node_cnt = 0
    for node_idx in node_idxs:
        node_map[node_idx] = node_cnt
        node_cnt += 1
    return node_map


def gen_noise(m, n):
    return np.random.uniform(0, 1., size=[m, n])


def get_gnn_sup(adj):
    num_nodes, _ = adj.shape
    adj = adj + np.eye(num_nodes)
    degs = np.sqrt(np.sum(adj, axis=1))
    sup = adj # GNN support
    for i in range(num_nodes):
        sup[i, :] /= degs[i]
    for j in range(num_nodes):
        sup[:, j] /= degs[j]
    return sup


def get_gnn_sup_woSE(adj):
    num_nodes, _ = adj.shape
    degs = np.sqrt(np.sum(adj, axis=1))
    sup = adj # GNN support
    for i in range(num_nodes):
        sup[i, :] /= degs[i]
    for j in range(num_nodes):
        sup[:, j] /= degs[j]
    return sup


def sparse_to_tuple(sparse_mx):
    def to_tuple(mx):
        if not sp.sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


def rand_proj(num_nodes, hid_dim):
    rand_mat = np.random.normal(0, 1.0/np.sqrt(hid_dim), (num_nodes, hid_dim))
    temp_l = np.linalg.norm(rand_mat, axis=1)
    for i in range(hid_dim):
        temp_row = rand_mat[:, i]
        for j in range(i-1):
            temp_j = rand_mat[:, j]
            temp_product = temp_row.T.dot(temp_j)/(temp_l[j]**2)
            temp_row -= temp_product*temp_j
        temp_row *= temp_l[i]/np.sqrt(temp_row.T.dot(temp_row))
        rand_mat[:, i] = temp_row
    return rand_mat

