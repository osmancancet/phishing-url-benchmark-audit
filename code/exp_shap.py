"""E7: can feature attribution reveal the formatting shortcut? Random 80/20 split of E1, seeds 0-2.
(a) Tree model on all 30 hand-crafted features (5 presentation flags, the https flag, 5 structure and 19 lexical
    features), explained with TreeSHAP on 2,000 test URLs: share of mean |SHAP| per feature group.
(b) Raw-string character n-gram model, explained with exact linear SHAP under feature independence
    (phi_j = w_j (x_j - mean_j)) on 2,000 test URLs: share of |phi| carried by n-grams that never occur in any
    canonical string of the corpus ('formatting-only' n-grams), and by n-grams present in a URL's raw string but
    absent from its own canonical string (n-grams created by that URL's formatting), and a symmetric share that also
    counts absent n-grams: |phi| on 'formatting-heavy' n-grams, whose document frequency in the raw training strings
    exceeds twice that in the canonical training strings plus 5, whether present in the URL or not.
(c) Same-model intervention: the fitted tree and the fitted n-gram model scored on the canonical form of the same
    test URLs, so that the drop refers to the explained model and not to a retrained one."""
import json, os, sys
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA, load
from urlfeat import PRESENTATION, SCHEME_TYPE, STRUCTURE, LEXICAL, canonical
from models import features, make, scores
from exp_within import CAP
from sklearn.model_selection import train_test_split
import shap

SEEDS = [0, 1, 2]
N_EXPLAIN = 2000
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "shap"); os.makedirs(OUT, exist_ok=True)
GROUPS = {"presentation": PRESENTATION, "https": SCHEME_TYPE, "structure": STRUCTURE, "lexical": LEXICAL}


def run(job):
    corpus, seed = job
    fn = os.path.join(OUT, f"{corpus}_s{seed}.json")
    if os.path.exists(fn):
        return fn
    d = load(corpus, cap_per_class=CAP, seed=seed); urls, y = d["url"].values, d["label"].values
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=seed)
    ex = np.random.RandomState(seed).choice(te, size=min(N_EXPLAIN, len(te)), replace=False)
    rec = {"corpus": corpus, "seed": seed}
    # (a) tree model on all hand-crafted features
    X = np.hstack([features("presentation_https", urls), features("lexical", urls)])  # 6 + (19 lexical + 5 structure)
    names = PRESENTATION + SCHEME_TYPE + LEXICAL + STRUCTURE
    clf = make("tree", seed).fit(X[tr], y[tr])
    rec["tree_scores"] = scores(y[te], clf.predict_proba(X[te])[:, 1])
    sv = np.abs(np.asarray(shap.TreeExplainer(clf).shap_values(X[ex]))).mean(0)
    rec["tree_feature_share"] = {n: float(v / sv.sum()) for n, v in zip(names, sv)}
    rec["tree_group_share"] = {g: float(sum(rec["tree_feature_share"][n] for n in fs)) for g, fs in GROUPS.items()}
    ute_c = [canonical(u) for u in urls[te]]
    Xc_te = np.hstack([features("presentation_https", ute_c), features("lexical", ute_c)])
    rec["tree_scores_canon_input"] = scores(y[te], clf.predict_proba(Xc_te)[:, 1])
    # (b) raw-string n-gram model, linear SHAP, formatting-only n-grams
    pipe = make("char_raw", seed).fit(list(urls[tr]), y[tr])
    vec, lr = pipe.steps[0][1], pipe.steps[1][1]
    canon_df = np.asarray((vec.transform([canonical(u) for u in urls]) > 0).sum(0)).ravel()
    fmt_only = canon_df == 0
    rec["char_scores"] = scores(y[te], pipe.predict_proba(list(urls[te]))[:, 1])
    rec["char_scores_canon_input"] = scores(y[te], pipe.predict_proba(ute_c)[:, 1])
    Xtr = vec.transform(list(urls[tr]))
    raw_df_tr = np.asarray((Xtr > 0).sum(0)).ravel()
    can_df_tr = np.asarray((vec.transform([canonical(u) for u in urls[tr]]) > 0).sum(0)).ravel()
    fmt_heavy = raw_df_tr > 2 * can_df_tr + 5
    mu = np.asarray(Xtr.mean(0)).ravel(); w = lr.coef_.ravel()
    Xe = vec.transform(list(urls[ex])); Xc = vec.transform([canonical(u) for u in urls[ex]])
    share = np.empty(len(ex)); share_own = np.empty(len(ex)); share_sym = np.empty(len(ex)); share_abs = np.empty(len(ex))
    mu, w = mu.astype(np.float32), w.astype(np.float32)
    for k in range(0, len(ex), 100):  # dense in chunks to bound memory
        xe = Xe[k:k + 100].toarray(); phi = np.abs((xe - mu) * w)
        share[k:k + 100] = phi[:, fmt_only].sum(1) / phi.sum(1)
        own = (xe > 0) & (Xc[k:k + 100].toarray() == 0)  # n-grams present only because of this URL's formatting
        share_own[k:k + 100] = (phi * own).sum(1) / phi.sum(1)
        share_sym[k:k + 100] = phi[:, fmt_heavy].sum(1) / phi.sum(1)  # present or absent formatting-heavy n-grams
        share_abs[k:k + 100] = (phi * (xe == 0)).sum(1) / phi.sum(1)
    rec["char_own_fmt_share"] = float(share_own.mean())
    rec["char_own_fmt_share_phish"] = float(share_own[y[ex] == 1].mean()); rec["char_own_fmt_share_legit"] = float(share_own[y[ex] == 0].mean())
    for nm, v in (("char_sym_fmt_share", share_sym), ("char_absent_share", share_abs)):
        rec[nm] = float(v.mean()); rec[nm + "_phish"] = float(v[y[ex] == 1].mean()); rec[nm + "_legit"] = float(v[y[ex] == 0].mean())
    rec["char_fmt_heavy_ngrams"] = int(fmt_heavy.sum())
    rec["char_fmt_only_ngrams"] = int(fmt_only.sum()); rec["char_vocab"] = int(len(w))
    rec["char_fmt_share"] = float(share.mean())
    rec["char_fmt_share_phish"] = float(share[y[ex] == 1].mean()); rec["char_fmt_share_legit"] = float(share[y[ex] == 0].mean())
    json.dump(rec, open(fn, "w"), indent=1)
    return fn


if __name__ == "__main__":
    jobs = [(c, s) for s in SEEDS for c in CORPORA]
    with Pool(int(os.environ.get("NPROC", 4))) as pool:
        for f in pool.imap_unordered(run, jobs):
            print("done", os.path.basename(f), flush=True)
