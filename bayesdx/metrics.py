import torch, torch.nn.functional as F

@torch.no_grad()
def brier_score(probs: torch.Tensor, labels: torch.Tensor, n_classes=2):
    y_onehot = F.one_hot(labels, num_classes=n_classes).float()
    return ((probs - y_onehot)**2).sum(dim=1).mean().item()

def nll(logits: torch.Tensor, labels: torch.Tensor):
    return F.cross_entropy(logits, labels).item()
