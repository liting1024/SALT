
from torch_sparse import SparseTensor
import importlib
import torch.nn as nn
import torch
import copy
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def dynamic_import(lib_path, class_name, **arg):
    module = importlib.import_module(lib_path)
    class_instances = getattr(module, class_name, None)
    return class_instances(**arg)


class SALT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.pred_size = args.pred_size
        
        self.feat_proj_dims = args.feat_proj_dims 
        self.num_feat_proj = len(self.feat_proj_dims)-1
        self.feat_proj = nn.ModuleList()
        for l in range(self.num_feat_proj):
            self.feat_proj.append(nn.Linear(in_features=self.feat_proj_dims[l], out_features=self.feat_proj_dims[l+1]))
            
        self.WLElayer = dynamic_import(
            lib_path=args.WLE['model_class'],
            class_name="WeightedLinkEncoder", 
            in_channels=args.WLE['in_channels'],
            hidden_channels=args.WLE['hidden_channels'],
            out_channels=args.WLE['out_channels'], 
            dropout=0.05,
            edrop=0.4,
            ln=True,
            use_xlin=True,
            tailact=True,
            twolayerlin=True,
            beta=1)

        self.tlssm = dynamic_import(
            lib_path=args.TLSSM['model_class'],
            class_name="TLSSM", 
            d_model=args.TLSSM['d_model'],
            d_ssm=args.TLSSM['d_ssm'],
            headdim=args.TLSSM['headdim'],
            d_state=args.TLSSM['d_state'],
            d_conv=args.TLSSM['d_conv'],
        )


    def forward(self, sup_list, feat_list, noise_list, pred_flag=True):
        N = feat_list[0].shape[0]
        window_size = len(sup_list)
        features = copy.deepcopy(feat_list)
        new_features = None
        for l in range(self.num_feat_proj):
            layer = self.feat_proj[l]
            new_features = []
            for x in features[:window_size + 1]:
                out = F.relu(layer(x))
                new_features.append(out)
            features = new_features

        embedding_inputs = []
        for t in range(window_size):
            concatenated = torch.cat((features[t], noise_list[t]), dim=1)
            embedding_inputs.append(concatenated)
            
        mpnn_outputs = []
        for t in range(window_size):
            sup = sup_list[t].coalesce()
            edge = sup.indices()
            adj = SparseTensor(row=edge[0], col=edge[1], value=sup.values())
            mpnn_out = self.WLElayer(embedding_inputs[t], adj, edge)
            mpnn_out = mpnn_out.squeeze(-1).contiguous()
            mpnn_out = SparseTensor(row=edge[0], col=edge[1], value=mpnn_out).to_dense().clone()
            mpnn_out = mpnn_out.reshape(N*N, -1).contiguous()
            mpnn_outputs.append(mpnn_out)

        # TLSSM
        tlssm_outputs = []
        conv_state, ssm_state = self.tlssm.init_local_and_global_hidden(batch_size=N*N,
                                                                        max_seqlen=window_size,
                                                                        dtype=torch.float32)
        for t in range(window_size):
            tlssm_out, conv_state, ssm_state = self.tlssm(mpnn_outputs[t].unsqueeze(1), conv_state, ssm_state)
            tlssm_outputs.append(tlssm_out.reshape(N, N))
            
        if pred_flag:
            return tlssm_outputs[-self.pred_size:]
        else:
            return tlssm_outputs


    def casual_training_loss(self, adj_est_list, gnd_list,alpha,theta):
        mae_loss = mae(adj_est_list, gnd_list, theta, alpha)
        mse_loss = mse(adj_est_list, gnd_list, theta, alpha)
        loss = mae_loss + mse_loss
        return loss


def mae(adj_est_list, gnd_list, theta, alpha):
    loss = 0.0
    win_size = len(adj_est_list)
    for i in range(win_size):
        adj_est = adj_est_list[i]
        gnd = gnd_list[i]
        decay = (1-theta)**(win_size-i-1)
        loss += decay*alpha*torch.norm(adj_est - gnd, p=1)
    return loss


def mse(adj_est_list, gnd_list, theta, alpha):
    loss = 0.0
    win_size = len(adj_est_list)
    for i in range(win_size):
        adj_est = adj_est_list[i]
        gnd = gnd_list[i]
        decay = (1-theta)**(win_size-i-1)
        loss += decay*alpha*torch.norm((adj_est - gnd), p='fro')**2
    return loss
