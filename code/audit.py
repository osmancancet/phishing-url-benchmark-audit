"""Descriptive artefact and overlap audit of the seven corpora (no model training)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA, load, load_raw
from urlfeat import PRESENTATION, STRUCTURE, presentation_feats, scheme_feats, structure_feats, canonical, reg_domain, matrix
from data import FILES

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)

alexa = pd.read_csv(FILES["PT-Alexa"][0], header=None, names=["rank", "domain"]).sort_values("rank")
TOP1K = set(alexa["domain"].astype(str).str.lower().head(1000).map(lambda x: reg_domain(x)))
res, frames = {}, {}
for n in CORPORA:
    raw = load_raw(n)
    d = load(n, canonical_dedup=False)
    d["canon"] = d["url"].map(canonical)
    d["rd"] = d["url"].map(reg_domain)
    dc = load(n)  # canonical-deduplicated version used for cross-corpus comparisons
    frames[n] = pd.DataFrame({"canon": dc["url"].map(canonical), "rd": dc["url"].map(reg_domain), "label": dc["label"]})
    P = np.hstack([matrix(d["url"], presentation_feats), matrix(d["url"], scheme_feats)]); S = matrix(d["url"], structure_feats)
    r = {"n_raw": int(len(raw)), "n_raw_phish": int(raw.label.sum()),
         "exact_dup_removed": int(len(raw) - len(raw.drop_duplicates(["url", "label"]))),
         "exact_dup_removed_legit": int((raw.label == 0).sum() - (raw.drop_duplicates(["url", "label"]).label == 0).sum()),
         "exact_dup_removed_phish": int((raw.label == 1).sum() - (raw.drop_duplicates(["url", "label"]).label == 1).sum()),
         "raw_label_conflicts": int(raw.drop_duplicates(["url", "label"])["url"].duplicated(keep=False).sum()),
         "n": int(len(d)), "n_phish": int(d.label.sum()), "n_legit": int((d.label == 0).sum())}
    # canonical collisions
    g = d.groupby("canon")["label"].agg(["nunique", "size"])
    r["canon_unique"] = int(len(g))
    r["canon_dup_rows"] = int(len(d) - len(g))
    r["canon_label_conflict_keys"] = int((g["nunique"] > 1).sum())
    # registered domain stats
    gd = d.groupby("rd")["label"].agg(["nunique", "size", "mean"])
    r["reg_domains"] = int(len(gd))
    r["reg_domains_legit"] = int(d[d.label == 0]["rd"].nunique()); r["reg_domains_phish"] = int(d[d.label == 1]["rd"].nunique())
    r["reg_domains_both_labels"] = int((gd["nunique"] > 1).sum())
    r["rows_in_mixed_domains"] = int(gd.loc[gd["nunique"] > 1, "size"].sum())
    r["urls_per_domain_phish"] = float(d[d.label == 1].groupby("rd").size().mean())
    r["urls_per_domain_legit"] = float(d[d.label == 0].groupby("rd").size().mean())
    for y, nm in [(0, "legit"), (1, "phish")]:
        m = (d.label == y).values
        r[f"pres_{nm}"] = {k: float(P[m, i].mean()) for i, k in enumerate(PRESENTATION + ["is_https"])}
        r[f"struct_{nm}"] = {k: float(np.mean(S[m, i])) if k not in ("url_len",) else float(np.median(S[m, i])) for i, k in enumerate(STRUCTURE)}
    # largest registered-domain group per class (private PSL suffixes on), share of class rows
    for y, nm in [(0, "legit"), (1, "phish")]:
        vc = d[d.label == y]["rd"].value_counts()
        r[f"largest_group_{nm}"] = [str(vc.index[0]), int(vc.iloc[0]), float(vc.iloc[0] / vc.sum())]
    # popularity proxy for label noise: share of phishing-labelled rows whose registered domain is an Alexa top-1k domain
    r["phish_on_alexa_top1k"] = float(d[d.label == 1]["rd"].isin(TOP1K).mean())
    # homepages (no path): share whose string ends in a slash, per class
    home = S[:, 0] == 0
    for y, nm in [(0, "legit"), (1, "phish")]:
        m = home & (d.label == y).values
        r[f"home_slash_{nm}"] = float(np.mean([str(u).endswith("/") for u in d["url"].values[m]])) if m.any() else None
    res[n] = r
    print(n, {k: r[k] for k in ["n", "exact_dup_removed", "canon_dup_rows", "canon_label_conflict_keys", "reg_domains_both_labels"]})
    print("  legit", {k: round(v, 3) for k, v in r["pres_legit"].items()}, {k: round(v, 2) for k, v in r["struct_legit"].items()})
    print("  phish", {k: round(v, 3) for k, v in r["pres_phish"].items()}, {k: round(v, 2) for k, v in r["struct_phish"].items()})

# cross-corpus overlap of registered domains and of canonical URLs, with label agreement
allf = pd.concat(frames.values(), ignore_index=True)
cross = {}
for a in CORPORA:
    for b in CORPORA:
        if a >= b:
            continue
        A, B = frames[a], frames[b]
        ca = A.drop_duplicates("canon").set_index("canon")["label"]; cb = B.drop_duplicates("canon").set_index("canon")["label"]
        common = ca.index.intersection(cb.index)
        da = A.groupby("rd")["label"].mean(); db = B.groupby("rd")["label"].mean()
        cd = da.index.intersection(db.index)
        # domain is "pure legit" in one corpus and "pure phish" in the other
        opp = int((((da[cd] == 0) & (db[cd] == 1)) | ((da[cd] == 1) & (db[cd] == 0))).sum())
        cross[f"{a}|{b}"] = {"shared_canonical_urls": int(len(common)),
                             "shared_canonical_conflicting_label": int((ca[common] != cb[common]).sum()),
                             "conflict_a_phish_b_legit": int(((ca[common] == 1) & (cb[common] == 0)).sum()),
                             "conflict_a_legit_b_phish": int(((ca[common] == 0) & (cb[common] == 1)).sum()),
                             "shared_reg_domains": int(len(cd)), "reg_domains_opposite_pure_label": opp}
res["_cross"] = cross
json.dump(res, open(os.path.join(OUT, "audit.json"), "w"), indent=1)
for k, v in cross.items():
    print(k, v)
