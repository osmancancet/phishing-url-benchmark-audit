"""E4: a character-level CNN (the character branch of URLNet, Le et al. 2018) on raw and canonical strings,
random and registered-domain-grouped splits, seeds 0-2. Same caps and splits as exp_within."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA, load
from urlfeat import canonical, reg_domain
from models import scores
from exp_within import CAP
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

SEEDS = [0, 1, 2]
MAXLEN, EMB, FILT, KS, EPOCHS, BATCH, LR = 200, 32, 128, (3, 4, 5, 6), 5, 256, 1e-3
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "cnn")
os.makedirs(OUT, exist_ok=True)


def encode(urls):
    X = np.zeros((len(urls), MAXLEN), dtype=np.int64)
    for i, u in enumerate(urls):
        b = [min(ord(ch), 127) + 1 for ch in u[:MAXLEN]]  # 0 = padding; non-ASCII mapped to 128
        X[i, :len(b)] = b
    return X


def run(job):
    corpus, seed, view, split = job
    fn = os.path.join(OUT, f"{corpus}_s{seed}_{view}_{split}.json")
    if os.path.exists(fn):
        return fn
    import torch, torch.nn as nn
    torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)

    class CharCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(130, EMB, padding_idx=0)
            self.convs = nn.ModuleList([nn.Conv1d(EMB, FILT, k) for k in KS])
            self.drop = nn.Dropout(0.5)
            self.fc = nn.Linear(FILT * len(KS), 1)

        def forward(self, x):
            e = self.emb(x).transpose(1, 2)
            h = torch.cat([torch.relu(c(e)).max(dim=2).values for c in self.convs], 1)
            return self.fc(self.drop(h)).squeeze(1)

    d = load(corpus, cap_per_class=CAP, seed=seed)
    urls, y = d["url"].values, d["label"].values
    if split == "random":
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    else:
        groups = d["url"].map(reg_domain).values
        tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(urls, y, groups))
    text = urls if view == "raw" else np.array([canonical(u) for u in urls])
    X = torch.from_numpy(encode(text)); Y = torch.from_numpy(y.astype(np.float32))
    model = CharCNN(); opt = torch.optim.Adam(model.parameters(), lr=LR); lossf = nn.BCEWithLogitsLoss()
    t0 = time.time()
    rng = np.random.RandomState(seed)
    for ep in range(EPOCHS):
        model.train(); perm = rng.permutation(tr)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad(); loss = lossf(model(X[b]), Y[b]); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(model(X[te[i:i + 2048]])) for i in range(0, len(te), 2048)]).numpy()
    s = scores(y[te], p); s.update(corpus=corpus, seed=seed, view=view, split=split, sec=round(time.time() - t0, 1))
    json.dump(s, open(fn, "w"), indent=1)
    return fn


if __name__ == "__main__":
    jobs = [(c, s, v, sp) for s in SEEDS for c in CORPORA for v in ("raw", "canon") for sp in ("random", "domain")]
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for f in pool.imap_unordered(run, jobs):
            print("done", os.path.basename(f), flush=True)
