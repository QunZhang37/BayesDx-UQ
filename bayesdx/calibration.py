import torch, numpy as np

def ece(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15):
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece_val = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i+1])
        prop = in_bin.float().mean().item()
        if prop > 0:
            acc = accuracies[in_bin].float().mean().item()
            conf = confidences[in_bin].float().mean().item()
            ece_val += abs(acc - conf) * prop
    return ece_val
