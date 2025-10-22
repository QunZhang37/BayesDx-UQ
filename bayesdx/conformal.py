import torch

def split_conformal_threshold(scores_calib, alpha=0.1):
    # scores: 1 - p_true (lower is better)
    q = torch.quantile(scores_calib, 1 - alpha, interpolation='higher')
    return q.item()

def prediction_set(probs, q):
    # return set of labels where 1 - p_y <= q  -> p_y >= 1 - q
    return probs >= (1 - q)
