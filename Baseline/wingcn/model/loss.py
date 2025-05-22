import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchmetrics
from model.config import cfg
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score

def prediction(pred_score, true_l):
    pred = pred_score.clone()
    pred = torch.where(pred > 0.5, 1, 0)
    pred = pred.detach().cpu().numpy()
    pred_score = pred_score.detach().cpu().numpy()

    true = true_l
    true = true.cpu().numpy()
    acc = accuracy_score(true, pred)
    ap = average_precision_score(true, pred_score)
    f1 = f1_score(true, pred, average='macro')
    macro_auc = roc_auc_score(true, pred_score, average='macro')
    micro_auc = roc_auc_score(true, pred_score, average='micro')

    return acc, ap, f1, macro_auc, micro_auc


def get_MAE(pred, y):
    if pred.device != 'cpu':
        pred = pred.clone()
        pred = pred.detach().cpu()
    if y.device != 'cpu':
        y = y.clone()
        y = y.detach().cpu()
    return torch.sum(torch.abs(pred - y))/(pred.shape[0]) #torch.mean(torch.abs(pred - y))

def get_RMSE(pred, y):
    if pred.device != 'cpu':
        pred = pred.clone()
        pred = pred.detach().cpu()
    if y.device != 'cpu':
        y = y.clone()
        y = y.detach().cpu()
    return torch.sqrt(torch.sum((pred - y) ** 2)/(pred.shape[0])) #torch.sqrt(torch.mean((pred - y) ** 2))  np.sqrt(np.linalg.norm(y-pred, ord='fro')**2/pred.shape[0])

def mse_mae_loss(pred, y):
    mse = nn.MSELoss()
    mae = nn.L1Loss()
    pred = pred.float()
    y = y.to(pred)
    loss = mse(pred, y) + mae(pred, y)

    return loss

def Link_loss_meta(pred, y):
    L = nn.BCELoss()
    pred = pred.float()
    y = y.to(pred)
    loss = L(pred, y)

    return loss

