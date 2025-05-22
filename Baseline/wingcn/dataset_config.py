def load_info(self):
    if self.data_name == 'T-Drive':
        self.num_nodes = 1279 
        self.num_snaps = 300 
        self.max_thres = 5000
        self.noise_dim = 512
        self.feat_dim = 32 
        self.pos_dim = 256 
        self.GNN_feat_dim = self.pos_dim
        
    elif self.data_name == 'DC':
        self.num_nodes = 128
        self.num_snaps = 700  
        self.max_thres = 5000 
        self.noise_dim = 100 
        self.feat_dim = 32 
        self.pos_dim = 32 
        self.GNN_feat_dim = self.feat_dim + self.pos_dim 

    elif self.data_name == 'HMob':
        self.num_nodes = 92 
        self.num_snaps = 500  
        self.max_thres = 250 
        self.noise_dim = 64 
        self.pos_dim = 32 
        self.GNN_feat_dim = self.pos_dim 

    elif self.data_name == 'IoT':
        self.num_nodes = 668 
        self.num_snaps = 144  
        self.max_thres = 1024 
        self.noise_dim =  48 
        self.feat_dim =  32 
        self.pos_dim =  32 

    elif self.data_name == 'LyonSchool':
        self.num_nodes = 242
        self.num_snaps = 157 
        self.max_thres = 20
        self.noise_dim =  32 
        self.feat_dim =  11 
        self.pos_dim =  32 
    
    elif self.data_name == 'INVIS13':  # 
        self.num_nodes = 100
        self.num_snaps = 403
        self.max_thres = 50
        self.noise_dim =  5 
        self.feat_dim =  5 
        self.pos_dim =  5 
    
    elif self.data_name == 'INVIS15':
        self.num_nodes = 232
        self.num_snaps = 431
        self.max_thres = 50
        self.noise_dim =  5 
        self.feat_dim =  12 
        self.pos_dim =  12 
    

    elif self.data_name == 'Thiers13': 
        self.num_nodes = 329
        self.num_snaps = 179
        self.max_thres = 50
        self.noise_dim =  16 
        self.feat_dim =  9 
        self.pos_dim =  16 
    
    else:
        raise ValueError('Invalid data name')
    return self