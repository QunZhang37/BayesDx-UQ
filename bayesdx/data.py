import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SyntheticDermDataset(Dataset):
    """Synthetic medical-like patches with circular 'lesions' and noise."""
    def __init__(self, n=2000, size=32, split='train', seed=123):
        rng = np.random.default_rng(seed if split=='train' else seed+1)
        self.X, self.y = [], []
        for i in range(n):
            img = rng.normal(0, 0.1, (size, size)).astype(np.float32)
            # inject a lesion: random circle radius/intensity
            radius = rng.integers(size//8, size//5)
            cx, cy = rng.integers(radius, size-radius, size=2)
            Y, X = np.ogrid[:size, :size]
            mask = (X - cx)**2 + (Y - cy)**2 <= radius**2
            intensity = rng.uniform(0.6, 1.2)
            img[mask] += intensity
            label = 1 if intensity > 0.9 else 0
            self.X.append(img[None, ...])
            self.y.append(label)
        self.X = np.stack(self.X)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y

def loader(batch_size=64, split='train'):
    ds = SyntheticDermDataset(split=split)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split=='train'))
