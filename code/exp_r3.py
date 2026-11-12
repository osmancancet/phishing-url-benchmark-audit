"""E6: controls added in the third internal review round.
(a) valthr: within-corpus thresholds chosen on a validation split. The random 80/20 split of E1 is kept; 20% of the
    training part is held out as validation, the character models are fitted on the remaining 64%, and the threshold
    with the highest validation TPR at validation FPR <= a is applied to the test part. 10 seeds.
(b) xthr: the same validation thresholds for the canonical and raw character models trained on one corpus and applied
    to each other corpus after the URL purge of E3. 3 seeds.
(c) kaggle_clean2: Kaggle-MU-clean (E5a) without phishing rows that start with a bare 'www.' or whose registered
    domain is in the Alexa top 1,000. 10 seeds.
(d) temporal: PT-Alexa with phishing URLs submitted up to 2019 for training and in 2020 for testing; a matched random
    control draws the same numbers of phishing rows from all years. Legitimate rows are split at random. 3 seeds."""
import json, os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA, FILES, load
from urlfeat import canonical, reg_domain
from models import features, make, scores
from exp_within import CAP, SEEDS
from exp_cross import CAP_TR, CAP_TE, SEEDS as XSEEDS, full_keys
from exp_extra import kaggle_clean, cap, fit_eval, SUB_MODELS
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

R = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(R, "r3"); os.makedirs(OUT, exist_ok=True)
ALPHAS = (0.001, 0.01)
CHAR = ["char_raw", "char_canon"]


def val_thresholds(yv, pv):
    f, t, thr = roc_curve(yv, pv)
    out = {}
    for a in ALPHAS:
        ok = f <= a
        out[a] = float(thr[ok][np.argmax(t[ok])])
    return out


def at_thr(y, p, thr):
    yhat = p >= thr
    fp = int((yhat & (y == 0)).sum()); tp = int((yhat & (y == 1)).sum())
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    return {"thr": thr, "fp": fp, "tp": tp, "n0": n0, "n1": n1, "fpr": fp / max(1, n0), "tpr": tp / max(1, n1)}


def fit_val(urls, y, tr, seed, m):
    fit, val = train_test_split(tr, test_size=0.2, stratify=y[tr], random_state=seed)
    X = features(m, urls)
    clf = make(m, seed).fit([X[i] for i in fit], y[fit])
    th = val_thresholds(y[val], clf.predict_proba([X[i] for i in val])[:, 1])
    return clf, th


def run(job):
    kind, corpus, seed = job
    fn = os.path.join(OUT, f"{kind}_{corpus}_s{seed}.json")
    if os.path.exists(fn):
        return fn
    rec = {"kind": kind, "corpus": corpus, "seed": seed}
    if kind == "valthr":
        d = load(corpus, cap_per_class=CAP, seed=seed); urls, y = d["url"].values, d["label"].values
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        rec["runs"] = {}
        for m in CHAR:
            clf, th = fit_val(urls, y, tr, seed, m)
            p = clf.predict_proba(features(m, urls[te]))[:, 1]
            rec["runs"][m] = {str(a): at_thr(y[te], p, th[a]) for a in ALPHAS}
            rec["runs"][m]["0.5"] = at_thr(y[te], p, 0.5)
    elif kind == "xthr":
        tr_d = load(corpus, cap_per_class=CAP_TR, seed=seed); urls, y = tr_d["url"].values, tr_d["label"].values
        ck = full_keys(corpus)[0]
        rec["runs"] = []
        for m in CHAR:
            clf, th = fit_val(urls, y, np.arange(len(y)), seed, m)
            for c in CORPORA:
                if c == corpus:
                    continue
                te = load(c, cap_per_class=CAP_TE, seed=100 + seed)
                te = te[~te["url"].map(canonical).isin(ck)]
                if te["label"].nunique() < 2:
                    continue
                yt = te["label"].values
                p = clf.predict_proba(features(m, te["url"].values))[:, 1]
                r = {"model": m, "test": c}
                for a in ALPHAS:
                    r[str(a)] = at_thr(yt, p, th[a])
                r["0.5"] = at_thr(yt, p, 0.5)
                rec["runs"].append(r)
    elif kind == "kaggle_clean2":
        full, info = kaggle_clean()
        alexa = pd.read_csv(FILES["PT-Alexa"][0], header=None, names=["rank", "domain"]).sort_values("rank")
        TOP1K = set(alexa["domain"].astype(str).str.lower().head(1000).map(reg_domain))  # as in audit.py
        rd = full["url"].map(reg_domain)
        bare_www = full["url"].str.lower().str.startswith("www.")
        drop = (full.label == 1) & (bare_www | rd.isin(TOP1K))
        info2 = {"n_bare_www_phish": int(((full.label == 1) & bare_www).sum()),
                 "n_top1k_phish": int(((full.label == 1) & rd.isin(TOP1K)).sum()),
                 "n_dropped": int(drop.sum()), "n_after_legit": int(((~drop) & (full.label == 0)).sum()),
                 "n_after_phish": int(((~drop) & (full.label == 1)).sum())}
        rec["info"] = info2
        d = cap(full[~drop].reset_index(drop=True), seed); urls, y = d["url"].values, d["label"].values
        tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
        rec["runs"] = fit_eval(urls, y, tr, te, SUB_MODELS, seed)
    elif kind in ("temporal", "temporal_random"):
        d = load("PT-Alexa")
        sub = pd.read_csv(FILES["PT-Alexa"][1], usecols=["url", "submission_time"])
        sub["url"] = sub["url"].astype(str).str.strip()
        yr = sub.drop_duplicates("url").set_index("url")["submission_time"].str[:4].astype(int)
        ph = d[d.label == 1].copy(); ph["year"] = ph["url"].map(yr)
        ph = ph.dropna(subset=["year"])
        early, late = ph[ph.year <= 2019], ph[ph.year == 2020]
        rng = np.random.RandomState(seed)
        n_tr, n_te = len(early), min(len(late), CAP_TE)
        if kind == "temporal":
            ph_tr = early; ph_te = late.sample(n=n_te, random_state=rng)
        else:
            pool = ph[ph.year <= 2020].sample(frac=1.0, random_state=rng)
            ph_tr, ph_te = pool.iloc[:n_tr], pool.iloc[n_tr:n_tr + n_te]
        leg = d[d.label == 0].sample(n=n_tr + n_te, random_state=rng)
        tr_d = pd.concat([ph_tr, leg.iloc[:n_tr]]); te_d = pd.concat([ph_te, leg.iloc[n_tr:]])
        all_d = pd.concat([tr_d, te_d], ignore_index=True)
        urls, y = all_d["url"].values, all_d["label"].values.astype(int)
        tr = np.arange(len(tr_d)); te = np.arange(len(tr_d), len(all_d))
        rec["n"] = {"train_phish": int(n_tr), "test_phish": int(n_te),
                    "train_https": float(pd.Series(ph_tr["url"]).str.lower().str.startswith("https://").mean()),
                    "test_https": float(pd.Series(ph_te["url"]).str.lower().str.startswith("https://").mean())}
        rec["runs"] = fit_eval(urls, y, tr, te, SUB_MODELS + ["char_canon_https"], seed)
    json.dump(rec, open(fn, "w"), indent=1)
    return fn


if __name__ == "__main__":
    jobs = ([("temporal", "PT-Alexa", s) for s in XSEEDS] + [("temporal_random", "PT-Alexa", s) for s in XSEEDS]
            + [("kaggle_clean2", "Kaggle-MU", s) for s in SEEDS]
            + [("xthr", c, s) for s in XSEEDS for c in CORPORA]
            + [("valthr", c, s) for s in SEEDS for c in CORPORA])
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for f in pool.imap_unordered(run, jobs):
            print("done", os.path.basename(f), flush=True)
