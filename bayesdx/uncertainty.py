import torch, torch.nn.functional as F
from .models import enable_mc_dropout

@torch.no_grad()
def mc_dropout_predict(model, x, T=30):
    model.eval()
    enable_mc_dropout(model)
    probs = []
    for _ in range(T):
        logits = model(x)
        probs.append(F.softmax(logits, dim=-1))
    probs = torch.stack(probs, dim=0)  # [T,B,C]
    mean = probs.mean(0)
    var = probs.var(0)  # predictive variance per class
    entropy = -(mean * (mean+1e-12).log()).sum(-1)
    return mean, var, entropy

def predictive_entropy(p):
    return -(p * (p+1e-12).log()).sum(-1)

def mutual_information(mc_probs):
    mean = mc_probs.mean(0)
    return predictive_entropy(mean) - (-(mc_probs*(mc_probs+1e-12).log()).sum(-1)).mean(0)
