"""E5: controls added in the second internal review round. Random 80/20 split, 10 seeds, same caps as E1.
(a) Kaggle-MU-clean: Kaggle-MU without registered domains that carry both labels inside Kaggle-MU and without
    phishing rows whose canonical URL or registered domain is legitimate in another corpus.
(b) Homepage subset: only URLs without a path in both classes (the complement of the path subset of E1c).
(c) Canonical string plus the https flag (char_canon_https).
(d) Share of explicit https among PhishTank submissions per year in the PT-Alexa source file."""
import json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA, FILES, load
from urlfeat import canonical, reg_domain, matrix, structure_feats
from models import features, make, scores
from exp_within import CAP, SEEDS
from sklearn.model_selection import train_test_split

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(R, "extra"); os.makedirs(OUT, exist_ok=True)
SUB_MODELS = ["presentation", "presentation_https", "char_raw", "char_canon"]


def canon_https(urls):
    return [("S " if u.lower().startswith("https://") else "N ") + canonical(u) for u in urls]


def kaggle_clean():
    d = load("Kaggle-MU")
    rd = d["url"].map(reg_domain); cu = d["url"].map(canonical)
    mixed = set(rd[d.label == 0]) & set(rd[d.label == 1])
    leg_c, leg_d = set(), set()
    for c in CORPORA:
        if c == "Kaggle-MU":
            continue
        o = load(c); o = o[o.label == 0]
        leg_c |= set(o["url"].map(canonical)); leg_d |= set(o["url"].map(reg_domain))
    drop = rd.isin(mixed) | ((d.label == 1) & (cu.isin(leg_c) | rd.isin(leg_d)))
    info = {"n_before": int(len(d)), "n_mixed_rows": int(rd.isin(mixed).sum()), "n_dropped": int(drop.sum()),
            "n_after_legit": int(((~drop) & (d.label == 0)).sum()), "n_after_phish": int(((~drop) & (d.label == 1)).sum())}
    return d[~drop].reset_index(drop=True), info


def cap(d, seed):
    rng = np.random.RandomState(seed)
    return pd.concat([g.sample(n=min(len(g), CAP), random_state=rng) for _, g in d.groupby("label")]).sort_index().reset_index(drop=True)


def fit_eval(urls, y, tr, te, models, seed):
    out = {}
    for m in models:
        X = canon_https(urls) if m == "char_canon_https" else features(m, urls)
        Xtr = [X[i] for i in tr] if isinstance(X, list) else X[tr]
        Xte = [X[i] for i in te] if isinstance(X, list) else X[te]
        p = make("char_raw" if m == "char_canon_https" else m, seed).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        out[m] = scores(y[te], p)
    return out


def run(job):
    kind, corpus, seed = job
    fn = os.path.join(OUT, f"{kind}_{corpus}_s{seed}.json")
    if os.path.exists(fn):
        return fn
    rec = {"kind": kind, "corpus": corpus, "seed": seed}
    if kind == "kaggle_clean":
        full, info = kaggle_clean(); rec["info"] = info
        d = cap(full, seed); urls, y = d["url"].values, d["label"].values
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        rec["runs"] = fit_eval(urls, y, tr, te, SUB_MODELS, seed)
    elif kind == "homepage":
        d = load(corpus, cap_per_class=CAP, seed=seed)
        nopath = ~matrix(d["url"], structure_feats)[:, 0].astype(bool)
        d = d[nopath].reset_index(drop=True); y = d["label"].values
        rec["n"] = [int((y == 0).sum()), int((y == 1).sum())]
        if min(rec["n"]) >= 500:
            tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
            rec["runs"] = fit_eval(d["url"].values, y, tr, te, SUB_MODELS, seed)
    elif kind == "canon_https":
        d = load(corpus, cap_per_class=CAP, seed=seed); y = d["label"].values
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        rec["runs"] = fit_eval(d["url"].values, y, tr, te, ["char_canon_https"], seed)
    json.dump(rec, open(fn, "w"), indent=1)
    return fn


def phishtank_https():
    d = pd.read_csv(FILES["PT-Alexa"][1], usecols=["url", "submission_time"])
    d["year"] = d["submission_time"].str[:4]
    d["https"] = d["url"].astype(str).str.lower().str.startswith("https://")
    g = d.groupby("year")["https"].agg(["mean", "size"])
    return {y: {"https_share": float(r["mean"]), "n": int(r["size"])} for y, r in g.iterrows()}


if __name__ == "__main__":
    json.dump(phishtank_https(), open(os.path.join(R, "phishtank_https.json"), "w"), indent=1)
    jobs = [("kaggle_clean", "Kaggle-MU", s) for s in SEEDS] + [(k, c, s) for s in SEEDS for k in ("homepage", "canon_https") for c in CORPORA]
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for f in pool.imap_unordered(run, jobs):
            print("done", os.path.basename(f), flush=True)
