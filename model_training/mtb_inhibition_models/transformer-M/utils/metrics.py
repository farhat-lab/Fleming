from torchmetrics.classification import BinaryAUROC, MulticlassAUROC
import torch, torch.nn as nn

def binary_classification_metric_return(y_true, y_pred, state):

    # assert state in ['train','validation','test'], "Wrong state information, should be in train, validation, and test."

    to_return = {}
    
    auroc = BinaryAUROC()(y_pred, y_true)

    bce_loss = nn.BCELoss()(y_pred, y_true)

    to_return[f'{state}auroc'] = auroc
    to_return[f'{state}BCE_loss'] = bce_loss

    return to_return

def regression_metric_return(y_true, y_pred, state):

    to_return = {}

    l1_loss = nn.L1Loss()(y_pred, y_true)
    l2_loss = nn.MSELoss()(y_pred, y_true)
    pc = pearson_correlation(y_pred, y_true)

    to_return[f'{state} L1_Loss'] = l1_loss
    to_return[f'{state} L2_Loss'] = l2_loss
    to_return[f'{state} Pearson_Correlation'] = pc

    return to_return

def pearson_correlation(y_pred, y_true):
    
    vx = y_pred - torch.mean(y_pred)
    vy = y_true - torch.mean(y_true)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)))