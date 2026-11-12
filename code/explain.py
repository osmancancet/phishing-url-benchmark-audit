"""E5: which character n-grams carry the decision of the raw-string model (seed 0, full training cap)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from data import CORPORA, load
from models import make
from exp_within import CAP

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "explain.json")
res = {}
for c in CORPORA:
    d = load(c, cap_per_class=CAP, seed=0)
    clf = make("char_raw", 0).fit(list(d["url"]), d["label"].values)
    vec, lr = clf.steps[0][1], clf.steps[1][1]
    names = np.array(vec.get_feature_names_out()); w = lr.coef_[0]
    o = np.argsort(w)
    res[c] = {"phish": [(names[i], float(w[i])) for i in o[::-1][:12]], "legit": [(names[i], float(w[i])) for i in o[:12]]}
    print(c, "PHISH:", [n for n, _ in res[c]["phish"]][:8], "LEGIT:", [n for n, _ in res[c]["legit"]][:8], flush=True)
json.dump(res, open(OUT, "w"), indent=1)
