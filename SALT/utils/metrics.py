import numpy as np

def get_RMSE(adj_est, gnd, num_nodes):
    f_norm = np.linalg.norm(gnd-adj_est, ord='fro')**2
    RMSE = np.sqrt(f_norm/(num_nodes*num_nodes))
    return RMSE


def get_MAE(adj_est, gnd, num_nodes):
    MAE = np.sum(np.abs(gnd-adj_est))/(num_nodes*num_nodes)
    return MAE