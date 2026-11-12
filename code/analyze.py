"""Aggregate per-run JSON into numbers.json. No number in the paper is typed by hand."""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA

R = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "results"))
TEST_FRAC = 0.2


def ci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i]); adj[i] = min(1.0, run)
    return adj


def corrected_t(d, test_frac=TEST_FRAC):
    """Nadeau and Bengio (2003) corrected resampled t-test for repeated random train/test splits."""
    d = np.asarray(d, float); k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 1.0 if d.mean() == 0 else 0.0
    t = d.mean() / np.sqrt((1 / k + test_frac / (1 - test_frac)) * var)
    return float(2 * stats.t.sf(abs(t), k - 1))


def corrected_ci(d, test_frac=TEST_FRAC):
    """95% interval with the Nadeau-Bengio variance correction (half-width)."""
    d = np.asarray(d, float); k = len(d)
    return float(d.mean()), float(stats.t.ppf(0.975, k - 1) * np.sqrt((1 / k + test_frac / (1 - test_frac)) * d.var(ddof=1)))


def precision_at(tpr, fpr, prev):
    return tpr * prev / (tpr * prev + fpr * (1 - prev)) if (tpr * prev + fpr * (1 - prev)) > 0 else float("nan")


def load_within():
    rows, meta = [], []
    for f in glob.glob(os.path.join(R, "within", "*.json")):
        r = json.load(open(f))
        meta.append({k: v for k, v in r.items() if k != "runs"})
        for x in r["runs"]:
            rows.append(dict(corpus=r["corpus"], seed=r["seed"], **x))
    return pd.DataFrame(rows), pd.DataFrame(meta)


def load_cross():
    rows = []
    for f in glob.glob(os.path.join(R, "cross", "*.json")):
        r = json.load(open(f))
        for x in r["runs"]:
            rows.append(dict(kind=r["kind"], train=r["corpus"], seed=r["seed"], n_train=r["n_train"], **x))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    N = {"audit": json.load(open(os.path.join(R, "audit.json")))}
    if os.path.exists(os.path.join(R, "probe.json")):
        P = json.load(open(os.path.join(R, "probe.json")))
        N["probe"] = {c: {k: (ci(v) if k != "path_subset_n" else v) for k, v in P[c].items()} for c in P}
    W, M = load_within()
    W["recall"] = W["recall"].astype(float)
    N["within"] = {}
    for (c, s, m), g in W.groupby(["corpus", "split", "model"]):
        e = {k: ci(g[k]) for k in ["mcc", "auc", "f1", "precision", "recall", "fpr", "acc"]}
        e["n_seeds"] = int(g.seed.nunique())
        e["prec_at_1pct"] = ci([precision_at(a, b, 0.01) for a, b in zip(g["recall"], g["fpr"])])
        e["prec_at_10pct"] = ci([precision_at(a, b, 0.10) for a, b in zip(g["recall"], g["fpr"])])
        # best-case operating points read from the test ROC (FPR <= 0.1% and <= 1%)
        for a in (("0.001", "0.01") if "tpr_at_fpr0.001" in g else ()):
            e[f"tpr_at_fpr{a}"] = ci(g[f"tpr_at_fpr{a}"])
            for prev in (0.01, 0.10):
                e[f"prec_{int(prev*100)}pct_op{a}"] = ci([precision_at(t, f, prev) for t, f in zip(g[f"tpr_at_fpr{a}"], g[f"fpr_used{a}"])])
        e["fp_zero_seeds"] = int((g["fp"] == 0).sum()); e["n_legit_test"] = int((g["fp"] + g["tn"]).mean())
        N["within"].setdefault(c, {}).setdefault(s, {})[m] = e
    N["split_meta"] = {c: {s: {"seen": ci(g[f"{s}_test_domain_seen"]), "phish_share": ci(g[f"{s}_test_phish_share"]),
                              "n_test": ci(g[f"{s}_n_test"])} for s in ["random", "domain"]} for c, g in M.groupby("corpus")}
    N["n_within"] = {c: int(g["n"].iloc[0]) if g["n"].nunique() == 1 else [int(g["n"].min()), int(g["n"].max())] for c, g in M.groupby("corpus")}
    comps = {"raw_vs_canon": (("random", "char_raw"), ("random", "char_canon")),
             "raw_vs_host": (("random", "char_raw"), ("random", "char_host")),
             "canon_random_vs_domain": (("random", "char_canon"), ("domain", "char_canon")),
             "lexical_random_vs_domain": (("random", "lexical"), ("domain", "lexical")),
             "host_random_vs_domain": (("random", "char_host"), ("domain", "char_host"))}
    tests = {}
    for name, (a, b) in comps.items():
        res = {}
        for c in CORPORA:
            out = {}
            for met in ["mcc", "auc"]:
                ga = W[(W.corpus == c) & (W.split == a[0]) & (W.model == a[1])].set_index("seed")[met]
                gb = W[(W.corpus == c) & (W.split == b[0]) & (W.model == b[1])].set_index("seed")[met]
                d = (ga - gb).dropna().values
                out[met] = {"diff": ci(d), "diff_corr": corrected_ci(d), "p": corrected_t(d)}
            res[c] = out
        for met in ["mcc", "auc"]:
            adj = holm([res[c][met]["p"] for c in CORPORA])
            for c, x in zip(CORPORA, adj):
                res[c][met]["p_holm"] = float(x)
        tests[name] = res
    N["tests"] = tests
    C = load_cross()
    if len(C):
        N["cross"] = {}
        for (k, pu, tr, te, m), g in C.groupby(["kind", "purge", "train", "test", "model"]):
            N["cross"].setdefault(k, {}).setdefault(pu, {}).setdefault(m, {}).setdefault(tr, {})[te] = {
                "mcc": ci(g["mcc"]), "auc": ci(g["auc"]), "fpr": ci(g["fpr"]), "recall": ci(g["recall"]),
                "n_test": int(g["n_test"].mean()), "phish_share": float(g["test_phish_share"].mean())}
        # corpus-level tests: mean over test corpora per training corpus (7 units), Wilcoxon signed-rank
        xt = {}
        for pu in ["url", "domain"]:
            Pp = C[(C.kind == "pair") & (C.purge == pu)].groupby(["train", "test", "model"])[["mcc", "auc"]].mean().unstack("model")
            for a, b in [("char_canon", "char_raw"), ("char_host", "char_raw"), ("char_canon", "lexical")]:
                for met in ["mcc", "auc"]:
                    d_pair = (Pp[(met, a)] - Pp[(met, b)])
                    d_corpus = d_pair.groupby(level="train").mean().values
                    xt[f"{pu}_{a}_minus_{b}_{met}"] = {"diff_pairs": ci(d_pair.values), "n_better_pairs": int((d_pair.values > 0).sum()),
                                                       "n_pairs": int(len(d_pair)), "diff_corpus": ci(d_corpus),
                                                       "p_corpus": float(stats.wilcoxon(d_corpus).pvalue), "n_better_corpus": int((d_corpus > 0).sum())}
        N["cross_tests"] = xt
        # where does the canonical gain come from? pairs in which the presentation probe ranks the classes the
        # wrong way round (AUC < 0.5: formatting conventions invert between corpora) versus the other pairs
        Pu = C[(C.kind == "pair") & (C.purge == "url")].groupby(["train", "test", "model"])["auc"].mean().unstack("model")
        gain = Pu["char_canon"] - Pu["char_raw"]; inv = Pu["presentation"] < 0.5
        conc = {"inverted": {"n_pairs": int(inv.sum()), "gain": ci(gain[inv].values)},
                "other": {"n_pairs": int((~inv).sum()), "gain": ci(gain[~inv].values)}}
        g2 = gain[(gain.index.get_level_values("train") != "Kaggle-MU") & (gain.index.get_level_values("test") != "Kaggle-MU")]
        dc = g2.groupby(level="train").mean().values
        conc["without_kaggle"] = {"n_pairs": int(len(g2)), "diff_corpus": ci(dc), "p_corpus": float(stats.wilcoxon(dc).pvalue),
                                  "n_better_corpus": int((dc > 0).sum()), "n_corpus": int(len(dc))}
        # the same gain averaged per test corpus (each test corpus is one unit)
        dt = gain.groupby(level="test").mean()
        conc["by_test"] = {"diff": ci(dt.values), "p": float(stats.wilcoxon(dt.values).pvalue), "n_better": int((dt.values > 0).sum()), "n": int(len(dt)),
                           "per_test": {k: float(v) for k, v in dt.items()}}
        dt2 = g2.groupby(level="test").mean()
        conc["by_test_without_kaggle"] = {"diff": ci(dt2.values), "p": float(stats.wilcoxon(dt2.values).pvalue), "n_better": int((dt2.values > 0).sum()), "n": int(len(dt2))}
        # precision a transferred detector would reach at 1% prevalence (threshold 0.5)
        Pf = C[(C.kind == "pair") & (C.purge == "url")].groupby(["train", "test", "model"])[["fpr", "recall"]].mean()
        conc["prec1"] = {m: {"median_fpr": float(g["fpr"].median()), "median_prec1": float(np.median([precision_at(r, f, 0.01) for r, f in zip(g["recall"], g["fpr"])]))}
                         for m, g in Pf.groupby(level="model")}
        N["cross_conc"] = conc
        # direction in time: publication year of each corpus (PT-Alexa: last year in its PhishTank dump)
        YEAR = {"PhishStorm": 2014, "ISCX-URL2016": 2016, "Ebbu2017": 2019, "PT-Alexa": 2020, "Hannousse2021": 2021, "Kaggle-MU": 2021, "PhiUSIIL": 2024}
        Pp = C[(C.kind == "pair") & (C.purge == "url")].groupby(["train", "test", "model"])["auc"].mean().reset_index()
        N["corpus_year"] = YEAR
        Pp["dir"] = [("forward" if YEAR[a] < YEAR[b] else "backward" if YEAR[a] > YEAR[b] else "same") for a, b in zip(Pp.train, Pp.test)]
        N["cross_time"] = {d: {m: {"auc": float(g[g.model == m]["auc"].mean()), "n_pairs": int((g.model == m).sum())} for m in g.model.unique()} for d, g in Pp.groupby("dir")}
        C.to_csv(os.path.join(R, "cross_runs.csv"), index=False)
    cf = glob.glob(os.path.join(R, "cnn", "*.json"))
    if cf:
        K = pd.DataFrame([json.load(open(f)) for f in cf])
        N["cnn"] = {}
        for (c, sp, v), g in K.groupby(["corpus", "split", "view"]):
            N["cnn"].setdefault(c, {}).setdefault(sp, {})[v] = {"mcc": ci(g["mcc"]), "mcc_sd": float(g["mcc"].std(ddof=1)), "mcc_min": float(g["mcc"].min()),
                                                                "mcc_max": float(g["mcc"].max()), "auc": ci(g["auc"]), "n_seeds": int(len(g)), "sec": float(g["sec"].mean())}
        for c in N["cnn"]:
            for sp in N["cnn"][c]:
                a = K[(K.corpus == c) & (K.split == sp) & (K.view == "raw")].set_index("seed")["mcc"]
                b = K[(K.corpus == c) & (K.split == sp) & (K.view == "canon")].set_index("seed")["mcc"]
                N["cnn"][c][sp]["raw_minus_canon"] = ci((a - b).dropna().values)
        K.to_csv(os.path.join(R, "cnn_runs.csv"), index=False)
    ef = glob.glob(os.path.join(R, "extra", "*.json"))
    if ef:
        ex = {}
        for f in ef:
            r = json.load(open(f))
            if "info" in r:
                N["kaggle_clean_info"] = r["info"]
            if "n" in r:
                ex.setdefault(r["kind"], {}).setdefault(r["corpus"], {})["n"] = r["n"]
            for m, sc in r.get("runs", {}).items():
                ex.setdefault(r["kind"], {}).setdefault(r["corpus"], {}).setdefault(m, []).append(sc)
        N["extra"] = {}
        for kind, cs in ex.items():
            for c, ms in cs.items():
                for m, L in ms.items():
                    if m == "n":
                        N["extra"].setdefault(kind, {}).setdefault(c, {})["n"] = L; continue
                    df = pd.DataFrame(L)
                    N["extra"].setdefault(kind, {}).setdefault(c, {})[m] = {"mcc": ci(df["mcc"]), "auc": ci(df["auc"]), "n_seeds": int(len(df)),
                        "prec_at_1pct": ci([precision_at(a, b, 0.01) for a, b in zip(df["recall"], df["fpr"])])}
    rf = glob.glob(os.path.join(R, "r3", "*.json"))
    if rf:
        r3 = {}
        recs = [json.load(open(f)) for f in rf]
        # (a) validation thresholds within corpus: rates pooled over seeds, precision from pooled rates
        V = [(r["corpus"], m, a, x) for r in recs if r["kind"] == "valthr" for m, d in r["runs"].items() for a, x in d.items()]
        if V:
            va = {}
            for c in CORPORA:
                for m in ["char_raw", "char_canon"]:
                    for a in ["0.001", "0.01", "0.5"]:
                        xs = [x for (cc, mm, aa, x) in V if cc == c and mm == m and aa == a]
                        if not xs:
                            continue
                        fp = sum(x["fp"] for x in xs); n0 = sum(x["n0"] for x in xs); tp = sum(x["tp"] for x in xs); n1 = sum(x["n1"] for x in xs)
                        fpr, tpr = fp / n0, tp / n1
                        va.setdefault(c, {}).setdefault(m, {})[a] = {"fp_total": fp, "n0_total": n0, "fpr": fpr, "tpr": tpr, "n_seeds": len(xs),
                            "fpr_per_seed": ci([x["fpr"] for x in xs]), "prec_1pct": precision_at(tpr, fpr, 0.01), "prec_01pct": precision_at(tpr, fpr, 0.001)}
            r3["valthr"] = va
        # (b) validation thresholds carried to other corpora
        X = [dict(train=r["corpus"], seed=r["seed"], model=x["model"], test=x["test"], **{f"{a}_{k}": x[a][k] for a in ["0.001", "0.01", "0.5"] for k in ["fpr", "tpr"]})
             for r in recs if r["kind"] == "xthr" for x in r["runs"]]
        if X:
            X = pd.DataFrame(X).groupby(["train", "test", "model"]).mean(numeric_only=True).reset_index()
            xa = {}
            for m, g in X.groupby("model"):
                for a in ["0.001", "0.01", "0.5"]:
                    f, t = g[f"{a}_fpr"].values, g[f"{a}_tpr"].values
                    xa.setdefault(m, {})[a] = {"n_pairs": int(len(g)), "median_fpr": float(np.median(f)), "median_tpr": float(np.median(t)),
                        "n_fpr_over_10x": int((f > 10 * float(a)).sum()) if a != "0.5" else None,
                        "median_prec1": float(np.nanmedian([precision_at(b, e, 0.01) for b, e in zip(t, f)])),
                        "median_prec01": float(np.nanmedian([precision_at(b, e, 0.001) for b, e in zip(t, f)]))}
            r3["xthr"] = xa
            X.to_csv(os.path.join(R, "xthr_runs.csv"), index=False)
        # (c, d) Kaggle-MU second clean variant and PT-Alexa time split
        for kind in ["kaggle_clean2", "temporal", "temporal_random"]:
            rs = [r for r in recs if r["kind"] == kind]
            if not rs:
                continue
            e = {m: {"mcc": ci([r["runs"][m]["mcc"] for r in rs]), "auc": ci([r["runs"][m]["auc"] for r in rs])} for m in rs[0]["runs"]}
            e["n_seeds"] = len(rs)
            if "info" in rs[0]:
                e["info"] = rs[0]["info"]
            if "n" in rs[0]:
                e["n"] = {k: float(np.mean([r["n"][k] for r in rs])) for k in rs[0]["n"]}
            r3[kind] = e
        N["r3"] = r3
    sf = glob.glob(os.path.join(R, "shap", "*.json"))
    if sf:
        recs = [json.load(open(f)) for f in sf]; sh = {}
        for c in CORPORA:
            rs = [r for r in recs if r["corpus"] == c]
            if not rs:
                continue
            e = {"n_seeds": len(rs), "tree_mcc": ci([r["tree_scores"]["mcc"] for r in rs]),
                 "tree_group": {g: ci([r["tree_group_share"][g] for r in rs]) for g in rs[0]["tree_group_share"]},
                 "tree_fmt": ci([r["tree_group_share"]["presentation"] + r["tree_group_share"]["https"] for r in rs]),
                 "char_fmt_only": ci([r["char_fmt_share"] for r in rs]), "char_own": ci([r["char_own_fmt_share"] for r in rs]),
                 "char_own_legit": ci([r["char_own_fmt_share_legit"] for r in rs]), "char_own_phish": ci([r["char_own_fmt_share_phish"] for r in rs]),
                 "char_sym": ci([r["char_sym_fmt_share"] for r in rs]), "char_sym_legit": ci([r["char_sym_fmt_share_legit"] for r in rs]),
                 "char_sym_phish": ci([r["char_sym_fmt_share_phish"] for r in rs]), "char_absent": ci([r["char_absent_share"] for r in rs]),
                 "char_vocab": float(np.mean([r["char_vocab"] for r in rs])),
                 "tree_same_drop": ci([r["tree_scores"]["mcc"] - r["tree_scores_canon_input"]["mcc"] for r in rs]),
                 "char_same_drop": ci([r["char_scores"]["mcc"] - r["char_scores_canon_input"]["mcc"] for r in rs]),
                 "pres_only": ci([r["tree_group_share"]["presentation"] for r in rs])}
            # largest seed-to-seed range of the reported shares and MCC
            e["max_seed_range"] = float(max(np.ptp([r[k] for r in rs]) for k in ("char_own_fmt_share", "char_sym_fmt_share")) )
            e["max_seed_range"] = max(e["max_seed_range"], float(np.ptp([r["tree_group_share"]["presentation"] + r["tree_group_share"]["https"] for r in rs])),
                                      float(np.ptp([r["tree_scores"]["mcc"] for r in rs])))
            fs = {}
            for r in rs:
                for n, v in r["tree_feature_share"].items():
                    fs.setdefault(n, []).append(v)
            e["tree_top"] = sorted(((n, float(np.mean(v))) for n, v in fs.items()), key=lambda x: -x[1])[:3]
            # dependence actually measured elsewhere: tree model without formatting flags (lexical + structure) and char raw - canonical
            sd = [r["seed"] for r in rs]
            lx = W[(W.corpus == c) & (W.split == "random") & (W.model == "lexical") & (W.seed.isin(sd))]["mcc"].mean()
            cr = W[(W.corpus == c) & (W.split == "random") & (W.model == "char_raw") & (W.seed.isin(sd))]["mcc"].mean()
            cc = W[(W.corpus == c) & (W.split == "random") & (W.model == "char_canon") & (W.seed.isin(sd))]["mcc"].mean()
            e["tree_dep"] = float(e["tree_mcc"][0] - lx); e["char_dep"] = float(cr - cc)
            sh[c] = e
        cs = list(sh)
        sh["_spearman_tree"] = float(stats.spearmanr([sh[c]["tree_fmt"][0] for c in cs], [sh[c]["tree_dep"] for c in cs]).statistic)
        sh["_spearman_char"] = float(stats.spearmanr([sh[c]["char_own"][0] for c in cs], [sh[c]["char_dep"] for c in cs]).statistic)
        sp = lambda a, b, cc=cs: stats.spearmanr([sh[c][a][0] if isinstance(sh[c][a], tuple) else sh[c][a] for c in cc],
                                                 [sh[c][b][0] if isinstance(sh[c][b], tuple) else sh[c][b] for c in cc])
        no_pt = [c for c in cs if c != "PT-Alexa"]
        r_ = sp("tree_fmt", "tree_dep", no_pt); sh["_spearman_tree_no_pt"] = float(r_.statistic); sh["_spearman_tree_no_pt_p"] = float(r_.pvalue)
        sh["_spearman_tree_p"] = float(sp("tree_fmt", "tree_dep").pvalue); sh["_spearman_char_p"] = float(sp("char_own", "char_dep").pvalue)
        r_ = sp("char_sym", "char_dep"); sh["_spearman_char_sym"] = float(r_.statistic); sh["_spearman_char_sym_p"] = float(r_.pvalue)
        r_ = sp("tree_fmt", "tree_same_drop"); sh["_spearman_tree_same"] = float(r_.statistic); sh["_spearman_tree_same_p"] = float(r_.pvalue)
        r_ = sp("char_sym", "char_same_drop"); sh["_spearman_char_sym_same"] = float(r_.statistic); sh["_spearman_char_sym_same_p"] = float(r_.pvalue)
        r_ = sp("char_own", "char_same_drop"); sh["_spearman_char_same"] = float(r_.statistic); sh["_spearman_char_same_p"] = float(r_.pvalue)
        sh["_max_seed_range"] = max(sh[c]["max_seed_range"] for c in cs)
        N["shap"] = sh
    if os.path.exists(os.path.join(R, "phishtank_https.json")):
        N["phishtank_https"] = json.load(open(os.path.join(R, "phishtank_https.json")))
    json.dump(N, open(os.path.join(R, "numbers.json"), "w"), indent=1)
    W.to_csv(os.path.join(R, "within_runs.csv"), index=False)
    for c in CORPORA:
        if c in N["within"]:
            print(c, {s: {m: round(v["mcc"][0], 3) for m, v in N["within"][c][s].items()} for s in N["within"][c]})
