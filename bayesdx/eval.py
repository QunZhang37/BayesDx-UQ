import torch
from torch.utils.data import DataLoader
from .data import SyntheticDermDataset
from .models import SimpleCNN
from .uncertainty import mc_dropout_predict
from .calibration import ece
from .metrics import brier_score, nll
from .utils import device
import json, os

def evaluate(ckpt, batch_size=128, T=30, outdir='outputs/eval'):
    os.makedirs(outdir, exist_ok=True)
    ds = SyntheticDermDataset(n=1000, split='test')
    loader = DataLoader(ds, batch_size=batch_size)
    net = SimpleCNN(); net.load_state_dict(torch.load(ckpt, map_location=device())); net.to(device()).eval()
    all_probs=[]; all_labels=[]; all_logits=[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device())
            mean, var, ent = mc_dropout_predict(net, x, T=T)
            all_probs.append(mean.cpu()); all_labels.append(y); all_logits.append((mean+1e-12).log())
    probs = torch.cat(all_probs); labels = torch.cat(all_labels)
    ece_val = ece(probs, labels)
    brier = brier_score(probs, labels, n_classes=2)
    # pseudo NLL using prob->logits approximation already done
    metrics = {"ece": float(ece_val), "brier": float(brier)}
    json.dump(metrics, open(os.path.join(outdir,"metrics.json"),"w"))
    return metrics

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    args = p.parse_args()
    print(evaluate(args.ckpt))
