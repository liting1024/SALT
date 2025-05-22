import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter


class EvolveGCN_O(nn.Module):
    def __init__(self, params):
        super(EvolveGCN_O, self).__init__()

        enc_dims = params.enc_dims
        dec_dims = params.dec_dims

        cell_args = Namespace({})
        cell_args.rows = enc_dims[0]
        cell_args.cols = enc_dims[1]

        self.evolve_weights = mat_GRU_cell(cell_args)
        self.activation = torch.nn.ReLU()
        self.GCN_init_weights = Parameter(torch.Tensor(enc_dims[0], enc_dims[1]))
        self.reset_param(self.GCN_init_weights)
        self.decoder = nn.Sequential(
            nn.Linear(dec_dims[0], dec_dims[1]),
            nn.ReLU(),
            nn.Linear(dec_dims[1], dec_dims[2])
        )

    def reset_param(self,t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)


    def forward(self, sup_list, feat_list, noise_list, pred_flag):
        win_size = len(sup_list)
        output_list = []

        GCN_weights = self.GCN_init_weights
        for t in range(win_size):
            sup = sup_list[t]
            sup = sup.coalesce()
            edge_index = sup.indices()
            edge_weight = sup.values()
    
            GCN_weights = self.evolve_weights(GCN_weights)
            node_embs = self.activation(sup_list[t].matmul(feat_list[t].matmul(GCN_weights)))

            output_list.append(node_embs)
        
        if pred_flag:
            return [self.decoder(node_embs)]  # 预测t+1 
        else:
            return [self.decoder(output) for output in output_list]
    
    def casual_training_loss(self, adj_est_list, gnd_list, alpha, theta):
        loss = 0.0
        win_size = len(adj_est_list)
        for i in range(win_size):
            adj_est = adj_est_list[i]
            gnd = gnd_list[i]
            decay = (1-theta)**(win_size-i-1)
            loss += decay*alpha*torch.norm((adj_est - gnd), p='fro')**2
            loss += decay*alpha*torch.norm(adj_est - gnd, p=1)
        return loss

class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

def pad_with_last_val(vect,k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect

class EGCN(torch.nn.Module):
    def __init__(self, enc_dims, activation, device='cpu', skipfeats=False):
        super().__init__()
        GRCU_args = Namespace({})

        feats = enc_dims
        self.device = device
        self.skipfeats = skipfeats
        self.GRCU_layers = nn.ModuleList()  # 
        for i in range(1,len(feats)):
            GRCU_args = Namespace({'in_feats' : feats[i-1],
                                     'out_feats': feats[i],
                                     'activation': activation})

            grcu_i = GRCU(GRCU_args)
            self.GRCU_layers.append(grcu_i.to(self.device))

    def parameters(self):
        # return self._parameters
        for module in self.GRCU_layers:
            yield from module.parameters()

    def forward(self, A_list, Nodes_list,nodes_mask_list):
        node_feats= Nodes_list[-1]

        for unit in self.GRCU_layers:
            Nodes_list = unit(A_list, Nodes_list)#,nodes_mask_list)

        out = Nodes_list[-1]
        if self.skipfeats:
            out = torch.cat((out,node_feats), dim=1)   # use node_feats.to_dense() if 2hot encoded input 
        return out


class GRCU(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        cell_args = Namespace({})
        cell_args.rows = args.in_feats
        cell_args.cols = args.out_feats

        self.evolve_weights = mat_GRU_cell(cell_args)

        self.activation = self.args.activation
        self.GCN_init_weights = Parameter(torch.Tensor(self.args.in_feats,self.args.out_feats))
        self.reset_param(self.GCN_init_weights)

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,A_list,node_embs_list):#,mask_list):
        GCN_weights = self.GCN_init_weights
        out_seq = []
        for t,Ahat in enumerate(A_list):
            node_embs = node_embs_list[t]
            #first evolve the weights from the initial and use the new weights with the node_embs
            GCN_weights = self.evolve_weights(GCN_weights)#,node_embs,mask_list[t])
            node_embs = self.activation(Ahat.matmul(node_embs.matmul(GCN_weights)))

            out_seq.append(node_embs)

        return out_seq

class mat_GRU_cell(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.reset = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Sigmoid())

        self.htilda = mat_GRU_gate(args.rows,
                                   args.cols,
                                   torch.nn.Tanh())
        
        self.choose_topk = TopK(feats = args.rows,
                                k = args.cols)

    def forward(self,prev_Q):#,prev_Z,mask):
        # z_topk = self.choose_topk(prev_Z,mask)
        z_topk = prev_Q

        update = self.update(z_topk,prev_Q)
        reset = self.reset(z_topk,prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q

        

class mat_GRU_gate(torch.nn.Module):
    def __init__(self,rows,cols,activation):
        super().__init__()
        self.activation = activation
        #the k here should be in_feats which is actually the rows
        self.W = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.W)

        self.U = Parameter(torch.Tensor(rows,rows))
        self.reset_param(self.U)

        self.bias = Parameter(torch.zeros(rows,cols))

    def reset_param(self,t):
        #Initialize based on the number of columns
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv,stdv)

    def forward(self,x,hidden):
        out = self.activation(self.W.matmul(x) + \
                              self.U.matmul(hidden) + \
                              self.bias)

        return out

class TopK(torch.nn.Module):
    def __init__(self,feats,k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats,1))
        self.reset_param(self.scorer)
        
        self.k = k

    def reset_param(self,t):
        #Initialize based on the number of rows
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv,stdv)

    def forward(self,node_embs,mask):
        scores = node_embs.matmul(self.scorer) / self.scorer.norm()
        scores = scores + mask

        vals, topk_indices = scores.view(-1).topk(self.k)
        topk_indices = topk_indices[vals > -float("Inf")]

        if topk_indices.size(0) < self.k:
            topk_indices = pad_with_last_val(topk_indices,self.k)
            
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        out = node_embs[topk_indices] * tanh(scores[topk_indices].view(-1,1))

        return out.t()