"""Load seven public phishing-URL corpora into a common (url, label) frame.

label: 1 = phishing, 0 = legitimate. Raw strings are kept unchanged.
Sources are git mirrors; commit hashes and file SHA-256 are written to results/provenance.json.
"""
import hashlib, json, os, subprocess
import numpy as np
import pandas as pd

DS = os.environ.get("URLDS_ROOT", "/home/user/ds")
P = os.path.join(DS, "probe")
G = os.path.join(P, "GramBeddings", "data")

SOURCES = {
    "pdd": ("https://github.com/ebubekirbbr/pdd", os.path.join(DS, "pdd")),
    "GramBeddings": ("https://github.com/fcdalgic/GramBeddings", os.path.join(P, "GramBeddings")),
    "PhiUSIIL-UCI-zip": ("https://github.com/AbdulHamidNoubaa/Phishing-Detection-Using-PhiUSIIL-Phishing-URL-Dataset-Website-",
                         os.path.join(P, "Phishing-Detection-Using-PhiUSIIL-Phishing-URL-Dataset-Website-")),
    "Hannousse-mirror": ("https://github.com/TetianaYaskova/Web-page-Phishing-Detection-Dataset",
                         os.path.join(P, "Web-page-Phishing-Detection-Dataset")),
}

FILES = {
    "Ebbu2017": [os.path.join(DS, "pdd/input/data_legitimate_36400.json"), os.path.join(DS, "pdd/input/data_phishing_37175.json")],
    "PhishStorm": [os.path.join(G, "PhishStorm/valid-urls.csv"), os.path.join(G, "PhishStorm/phishing-urls.csv")],
    "ISCX-URL2016": [os.path.join(G, "ISCXURL2016/Benign_list_big_final.csv"), os.path.join(G, "ISCXURL2016/phishing_dataset.csv")],
    "PT-Alexa": [os.path.join(G, "PDRCNN/alexa1m_dataset.csv"), os.path.join(G, "PDRCNN/Phishtank_dataset.csv")],
    "Hannousse2021": [os.path.join(P, "Web-page-Phishing-Detection-Dataset/dataset_phishing.csv")],
    "PhiUSIIL": [os.path.join(P, "Phishing-Detection-Using-PhiUSIIL-Phishing-URL-Dataset-Website-/uci/PhiUSIIL_Phishing_URL_Dataset.csv")],
    "Kaggle-MU": [os.path.join(G, "kaggle/01-sid321axn/malicious_phish.csv")],
}

CORPORA = list(FILES)


def _lines(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return [l.rstrip("\r\n") for l in f if l.strip()]


def load_raw(name):
    f = FILES[name]
    if name == "Ebbu2017":
        leg = json.load(open(f[0], encoding="utf-8")); ph = json.load(open(f[1], encoding="utf-8"))
        return pd.DataFrame({"url": leg + ph, "label": [0] * len(leg) + [1] * len(ph)})
    if name in ("PhishStorm", "ISCX-URL2016"):
        leg, ph = _lines(f[0]), _lines(f[1])
        return pd.DataFrame({"url": leg + ph, "label": [0] * len(leg) + [1] * len(ph)})
    if name == "PT-Alexa":
        # PDRCNN-style construction (Wang et al., 2019): PhishTank verified phish + Alexa top-1M domains.
        ph = pd.read_csv(f[1], usecols=["url"])["url"].astype(str).tolist()
        alexa = pd.read_csv(f[0], header=None, names=["rank", "domain"])
        alexa = alexa.sort_values("rank")
        leg = alexa["domain"].astype(str).head(len(ph)).tolist()  # top-ranked domains, class-balanced
        return pd.DataFrame({"url": leg + ph, "label": [0] * len(leg) + [1] * len(ph)})
    if name == "Hannousse2021":
        d = pd.read_csv(f[0], usecols=["url", "status"])
        return pd.DataFrame({"url": d["url"].astype(str), "label": (d["status"] == "phishing").astype(int)})
    if name == "PhiUSIIL":
        d = pd.read_csv(f[0], usecols=["URL", "label"], encoding="utf-8-sig")
        return pd.DataFrame({"url": d["URL"].astype(str), "label": (d["label"] == 0).astype(int)})  # UCI: 1 = legitimate
    if name == "Kaggle-MU":
        d = pd.read_csv(f[0])
        d = d[d["type"].isin(["benign", "phishing"])]
        return pd.DataFrame({"url": d["url"].astype(str), "label": (d["type"] == "phishing").astype(int)})
    raise KeyError(name)


def load(name, cap_per_class=None, seed=0, canonical_dedup=True):
    """Return a de-duplicated corpus. Exact duplicates are removed and strings carrying both labels dropped.
    With canonical_dedup (default), URLs that collapse to the same canonical form are reduced to the first
    occurrence, and canonical forms carrying both labels are dropped, so that no canonical string can occur on
    both sides of a split."""
    from urlfeat import canonical
    d = load_raw(name)
    d["url"] = d["url"].str.strip()
    d = d[d["url"].str.len() > 0]
    d = d.drop_duplicates(["url", "label"])
    conflict = d["url"].duplicated(keep=False)
    d = d[~conflict].reset_index(drop=True)
    if canonical_dedup:
        d["_c"] = d["url"].map(canonical)
        d = d.drop_duplicates(["_c", "label"])
        d = d[~d["_c"].duplicated(keep=False)].drop(columns="_c").reset_index(drop=True)
    if cap_per_class:
        rng = np.random.RandomState(seed)
        parts = []
        for y, g in d.groupby("label"):
            parts.append(g.sample(n=min(len(g), cap_per_class), random_state=rng))
        d = pd.concat(parts).sort_index().reset_index(drop=True)
    d["corpus"] = name
    return d


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def provenance(out):
    rec = {"repos": {}, "files": {}}
    for k, (url, path) in SOURCES.items():
        try:
            c = subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            c = None
        rec["repos"][k] = {"url": url, "commit": c}
    for n, fs in FILES.items():
        rec["files"][n] = [{"path": os.path.relpath(p, DS), "sha256": sha256(p), "bytes": os.path.getsize(p)} for p in fs]
    json.dump(rec, open(out, "w"), indent=2)
    return rec


if __name__ == "__main__":
    provenance(os.path.join(os.path.dirname(__file__), "..", "results", "provenance.json"))
    for n in CORPORA:
        r = load_raw(n); d = load(n, canonical_dedup=False)
        print(f"{n:14s} raw={len(r):7d} (phish={int(r.label.sum())}) dedup={len(d):7d} phish={int(d.label.sum())} legit={int((d.label==0).sum())}")
