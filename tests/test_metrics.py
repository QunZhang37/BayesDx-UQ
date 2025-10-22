import torch
from bayesdx.calibration import ece
from bayesdx.metrics import brier_score

def test_metrics_shapes():
    probs = torch.tensor([[0.7,0.3],[0.2,0.8]])
    labels = torch.tensor([0,1])
    assert 0 <= ece(probs, labels) <= 1
    assert 0 <= brier_score(probs, labels, 2) <= 1
