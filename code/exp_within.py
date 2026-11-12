"""E1/E2: within-corpus evaluation, random vs registered-domain-grouped split, 10 repeated splits."""
import json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA, load
from urlfeat import reg_domain
from models import MODELS, features, make, scores
from sklearn.model_selection import train_test_split, StratifiedGroupKFold

CAP = 25_000
SEEDS = list(range(10))
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "within")
os.makedirs(OUT, exist_ok=True)


def run(job):
    corpus, seed = job
    fn = os.path.join(OUT, f"{corpus}_s{seed}.json")
    if os.path.exists(fn):
        return fn
    d = load(corpus, cap_per_class=CAP, seed=seed)
    urls, y = d["url"].values, d["label"].values
    groups = d["url"].map(reg_domain).values
    splits = {}
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    splits["random"] = (tr, te)
    # stratified, group-disjoint 80/20 split: first fold of a shuffled 5-fold StratifiedGroupKFold
    tr, te = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(urls, y, groups))
    splits["domain"] = (tr, te)
    rec = {"corpus": corpus, "seed": seed, "n": int(len(y)), "runs": []}
    # share of test rows whose registered domain also occurs in train
    for sname, (tr, te) in splits.items():
        trg = set(groups[tr])
        rec[f"{sname}_test_domain_seen"] = float(np.mean([g in trg for g in groups[te]]))
        rec[f"{sname}_test_phish_share"] = float(y[te].mean())
        rec[f"{sname}_n_test"] = int(len(te))
    feats = {m: features(m, urls) for m in MODELS}
    for sname, (tr, te) in splits.items():
        for m in MODELS:
            X = feats[m]
            Xtr = [X[i] for i in tr] if isinstance(X, list) else X[tr]
            Xte = [X[i] for i in te] if isinstance(X, list) else X[te]
            t0 = time.time()
            clf = make(m, seed).fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            s = scores(y[te], p); s.update(split=sname, model=m, sec=round(time.time() - t0, 2))
            rec["runs"].append(s)
    json.dump(rec, open(fn, "w"), indent=1)
    return fn


if __name__ == "__main__":
    jobs = [(c, s) for s in SEEDS for c in CORPORA]
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for f in pool.imap_unordered(run, jobs):
            print("done", os.path.basename(f), flush=True)
