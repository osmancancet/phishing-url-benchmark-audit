"""E3: cross-corpus transfer (train on one corpus, test on each other) and leave-one-corpus-out (LOCO).
Two test-set hygiene levels: 'url' removes test URLs whose canonical form occurs anywhere in the full training
corpus (or corpora); 'domain' additionally removes test URLs whose registered domain occurs there."""
import json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA, load
from urlfeat import canonical, reg_domain
from models import features, make, scores

CAP_TR, CAP_TE, CAP_LOCO = 25_000, 5_000, 8_000
SEEDS = list(range(3))
XMODELS = ["presentation", "presentation_https", "lexical", "char_raw", "char_canon", "char_host"]
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "cross")
os.makedirs(OUT, exist_ok=True)
_FULL = {}


def full_keys(c):
    if c not in _FULL:
        d = load(c)
        _FULL[c] = (set(d["url"].map(canonical)), set(d["url"].map(reg_domain)))
    return _FULL[c]


def run(job):
    kind, corpus, seed = job
    fn = os.path.join(OUT, f"{kind}_{corpus}_s{seed}.json")
    if os.path.exists(fn):
        return fn
    if kind == "pair":
        train = load(corpus, cap_per_class=CAP_TR, seed=seed)
        src = [corpus]
        tests = {c: load(c, cap_per_class=CAP_TE, seed=100 + seed) for c in CORPORA if c != corpus}
    else:
        src = [c for c in CORPORA if c != corpus]
        train = pd.concat([load(c, cap_per_class=CAP_LOCO, seed=seed) for c in src], ignore_index=True)
        tests = {corpus: load(corpus, cap_per_class=CAP_TE, seed=100 + seed)}
    ck = set().union(*[full_keys(c)[0] for c in src]); dk = set().union(*[full_keys(c)[1] for c in src])
    views = {}
    for k, te in tests.items():
        cu = te["url"].map(canonical); rd = te["url"].map(reg_domain)
        u = te[~cu.isin(ck)]
        views[(k, "url")] = u
        views[(k, "domain")] = u[~rd[~cu.isin(ck)].isin(dk)]
    rec = {"kind": kind, "corpus": corpus, "seed": seed, "n_train": int(len(train)), "runs": []}
    for m in XMODELS:
        clf = make(m, seed).fit(features(m, train["url"].values), train["label"].values)
        for (k, purge), te in views.items():
            if len(te) == 0 or te["label"].nunique() < 2:
                continue
            p = clf.predict_proba(features(m, te["url"].values))[:, 1]
            s = scores(te["label"].values, p)
            s.update(model=m, test=k, purge=purge, n_test=int(len(te)), test_phish_share=float(te["label"].mean()))
            rec["runs"].append(s)
    json.dump(rec, open(fn, "w"), indent=1)
    return fn


if __name__ == "__main__":
    jobs = [(k, c, s) for s in SEEDS for k in ("pair", "loco") for c in CORPORA]
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for f in pool.imap_unordered(run, jobs):
            print("done", os.path.basename(f), flush=True)
