import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import random_split, DataLoader
from .data import SyntheticDermDataset
from .models import SimpleCNN
from .utils import set_seed, device
import os, json

def train(model='mc_dropout', epochs=5, batch_size=64, lr=1e-3, seed=42, outdir='outputs/run'):
    set_seed(seed)
    os.makedirs(outdir, exist_ok=True)
    ds = SyntheticDermDataset(n=3000)
    n_train = int(0.7*len(ds)); n_val = len(ds)-n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    net = SimpleCNN(dropout_p=0.5).to(device())
    opt = optim.Adam(net.parameters(), lr=lr)
    best_val = 0; best_path = os.path.join(outdir, 'checkpoints'); os.makedirs(best_path, exist_ok=True)
    for ep in range(1, epochs+1):
        net.train()
        for x,y in train_loader:
            x,y = x.to(device()), y.to(device())
            opt.zero_grad()
            logits = net(x)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward(); opt.step()
        # simple val acc
        net.eval(); correct=0; total=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y = x.to(device()), y.to(device())
                pred = net(x).argmax(-1)
                correct += (pred==y).sum().item(); total += y.numel()
        acc = correct/total
        if acc>best_val:
            best_val=acc
            torch.save(net.state_dict(), os.path.join(best_path, 'best.pt'))
        print(f'Epoch {ep}: val_acc={acc:.3f}')
    json.dump({"best_val_acc": best_val}, open(os.path.join(outdir,"summary.json"),"w"))
    return os.path.join(best_path,'best.pt')

if __name__ == "__main__":
    train()
