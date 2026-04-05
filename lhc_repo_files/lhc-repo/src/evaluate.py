import torch
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def reconstruction_error(model, X, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        X_hat = model(X_t)
    return ((X_t - X_hat) ** 2).mean(dim=1).cpu().numpy()


def anomaly_roc(err_bkg, err_sig):
    y_true  = np.concatenate([np.zeros(len(err_bkg)), np.ones(len(err_sig))])
    y_score = np.concatenate([err_bkg, err_sig])
    auc     = roc_auc_score(y_true, y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    return auc, fpr, tpr, thresholds


def per_type_errors(model, X_clean, jet_type, scaler, type_map, n=3000, device=None):
    # returns dict of {type_name: error_array}
    results = {}
    for jet_id, name in type_map.items():
        mask   = jet_type == jet_id
        X_type = scaler.transform(X_clean[mask][:n])
        results[name] = reconstruction_error(model, X_type, device)
    return results
