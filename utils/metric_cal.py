import numpy as np
from skimage.metrics import structural_similarity as ssim
def inv_rmse(estimate, target):
    return np.sqrt(np.mean((1.0/estimate - 1.0/target) ** 2))

def inv_mae(estimate, target):
    return np.mean(np.abs(1.0/estimate - 1.0/target))

def calmetrics(pred, target, mask):
    metrics = np.zeros((1, 9), dtype=float)
    # convert (b, c, h, w) to (c, h,w)
    pred = pred.squeeze()
    target = target.squeeze()
    mask = mask.squeeze()

    # MSE
    metrics[0, 0] = np.mean((target[mask] - pred[mask]) ** 2)
    
    # RMSE
    metrics[0, 1] = np.sqrt(metrics[0, 0])

    # log RMS
    rmse_log = (np.log(target[mask]) - np.log(pred[mask])) ** 2
    metrics[0, 2] = np.sqrt(rmse_log.mean())
    
    # absolute relative
    metrics[0, 3] = np.mean(np.abs(target[mask] - pred[mask]) / target[mask])

    # square relative
    metrics[0, 4] = np.mean(((target[mask] - pred[mask]) ** 2) / target[mask])
    
    # SSIMs
    metrics[0, 5] = 0
    # MAE
    # metrics[0, 5] = np.mean(np.abs(target[mask] - pred[mask]))
    
    # accuracies
    thresh = np.maximum((target[mask] / pred[mask]), (pred[mask] / target[mask]))
    metrics[0,6] = (thresh < 1.25).mean()
    metrics[0,7] = (thresh < 1.25 ** 2).mean()
    metrics[0,8] = (thresh < 1.25 ** 3).mean()
    
    
    return metrics