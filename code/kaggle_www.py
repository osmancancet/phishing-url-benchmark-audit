"""Describe the phishing rows of Kaggle-MU-clean that start with a bare 'www.': top registered domains and the share
that contains one of the 14 brand and credential keywords of the lexical group, against the other phishing rows."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from exp_extra import kaggle_clean
from urlfeat import reg_domain, SUSP

d, _ = kaggle_clean()
ph = d[d.label == 1]; low = ph["url"].str.lower(); www = low.str.startswith("www.")
kw = low.apply(lambda u: any(k in u for k in SUSP))
out = {"n_bare_www": int(www.sum()), "n_phish": int(len(ph)), "kw_share_bare_www": float(kw[www].mean()),
       "kw_share_other": float(kw[~www].mean()), "top_domains": ph[www]["url"].map(reg_domain).value_counts().head(10).to_dict()}
json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "kaggle_www.json"), "w"), indent=1)
print(out)
