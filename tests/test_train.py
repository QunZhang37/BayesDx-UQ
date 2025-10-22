import os, json
from bayesdx.train import train

def test_train_runs(tmp_path):
    ckpt = train(epochs=1, outdir=tmp_path.as_posix())
    assert os.path.exists(ckpt)
    js = os.path.join(tmp_path, "summary.json")
    assert os.path.exists(js)
