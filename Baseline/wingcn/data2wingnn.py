import os
import numpy as np

old_root_dir = './Dataset/'
new_root_dir = './WinDataset/'

def print_datasetname_check_dirs():
    for root, dirs, _ in os.walk(old_root_dir):
        for dataset_name in dirs:
            dir_path = os.path.join(root, dataset_name)
            print(f"Found folder: {dir_path}")

def create_edge_index():
    dataset_name_list = [
    # 'HMob', 
    # 'IoT', 
    # 'DC', 
    # 'INVIS13', 
    # 'LyonSchool', 
    'INVIS15', 
    'LH10', 
    # 'Thiers13'
    # 'T-Drive',
    # 'Mesh', 
    ] 

    dataset_name = dataset_name_list[0]
    
    for dataset_name in dataset_name_list:
        edge_seq_path = os.path.join(old_root_dir, dataset_name, f'{dataset_name}_edge_seq.npy')
        edge_seq = np.load(edge_seq_path, allow_pickle=True)

        for index, edge_list in enumerate(edge_seq): 

            new_edge_list, edge_weight = [], []
            for edge in edge_list:
                new_edge_list.append([edge[0], edge[1]])
                edge_weight.append(edge[2])

            new_edge_list = np.array(new_edge_list).T
            edge_weight = np.array(edge_weight)

            new_edge_index_path = os.path.join(new_root_dir, dataset_name, 'edge_index', f'{index}.npy')
            edge_weight_path = os.path.join(new_root_dir, dataset_name, 'edge_weight', f'{index}.npy')

            os.makedirs(os.path.dirname(new_edge_index_path), exist_ok=True)
            os.makedirs(os.path.dirname(edge_weight_path), exist_ok=True)

            np.save(new_edge_index_path, new_edge_list)
            np.save(edge_weight_path, edge_weight)

            print(f"Saved {new_edge_index_path}")

def create_node_feature():
    dataset_name_list = [
        # 'HMob',
        # 'IoT', 
        # 'DC', 
        # 'INVIS13', 
        # 'LyonSchool', 
        'INVIS15', 
        'LH10', 
        # 'Thiers13'
        # 'T-Drive', 
        # 'Mesh', 
        ] 

    dataset_name = dataset_name_list[0]

    for dataset_name in dataset_name_list:
        node_feat_path = os.path.join(old_root_dir, dataset_name, f'{dataset_name}_feat.npy')
        node_feat_seq = np.load(node_feat_path, allow_pickle=True)
        if dataset_name in ['IoT', 'T-Drive']:
            node_feat_seq = node_feat_seq[:,:32]
        new_node_feature = np.array(node_feat_seq)
        new_node_feat_path = os.path.join(new_root_dir, dataset_name, 'node_feature', f'0.npy')
        if not os.path.exists(os.path.dirname(new_node_feat_path)):
            os.makedirs(os.path.dirname(new_node_feat_path))
        np.save(new_node_feat_path, new_node_feature)
        print(f"Saved {new_node_feat_path}")

def create_edge_time():
    pass

def create_edge_feature():
    pass

if __name__ == "__main__":
    create_edge_index()
    create_node_feature()
