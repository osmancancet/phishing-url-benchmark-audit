"""E1c: presentation-probe ablations (drop one flag at a time, add https) and the probe restricted to URLs that
have a path in both classes, which separates presentation from homepage sampling. Random 80/20 split, 10 seeds."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA, load
from urlfeat import PRESENTATION, presentation_feats, scheme_feats, structure_feats, matrix
from models import make, scores
from exp_within import CAP, SEEDS
from sklearn.model_selection import train_test_split

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "probe.json")
res = {}
for c in CORPORA:
    res[c] = {}
    for seed in SEEDS:
        d = load(c, cap_per_class=CAP, seed=seed)
        y = d["label"].values
        P = matrix(d["url"], presentation_feats); H = matrix(d["url"], scheme_feats)
        has_path = matrix(d["url"], structure_feats)[:, 0].astype(bool)
        variants = {"all": P, "all+https": np.hstack([P, H])}
        for i, k in enumerate(PRESENTATION):
            variants[f"drop_{k}"] = np.delete(P, i, axis=1)
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        for k, X in variants.items():
            p = make("presentation", seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            res[c].setdefault(k, []).append(scores(y[te], p)["mcc"])
        # path-only subset
        idx = np.where(has_path)[0]
        n0, n1 = int((y[idx] == 0).sum()), int((y[idx] == 1).sum())
        res[c].setdefault("path_subset_n", [n0, n1])
        if min(n0, n1) >= 500:
            tr, te = train_test_split(idx, test_size=0.2, stratify=y[idx], random_state=seed)
            p = make("presentation", seed).fit(P[tr], y[tr]).predict_proba(P[te])[:, 1]
            res[c].setdefault("path_subset", []).append(scores(y[te], p)["mcc"])
    print(c, {k: round(float(np.mean(v)), 3) for k, v in res[c].items() if k != "path_subset_n"}, res[c]["path_subset_n"], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
