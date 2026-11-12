"""Model zoo. All hyper-parameters are fixed here before any experiment runs."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, confusion_matrix
from urlfeat import presentation_feats, scheme_feats, structure_feats, lexical_feats, canonical, host_view, matrix

HGB = dict(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, early_stopping=False)
TFIDF = dict(analyzer="char", ngram_range=(2, 5), min_df=2, max_features=300_000, sublinear_tf=True, lowercase=False, dtype=np.float32)
LR = dict(C=1.0, solver="liblinear", max_iter=2000)

MODELS = ["presentation", "presentation_https", "structure", "lexical", "char_raw", "char_canon", "char_host"]


def features(name, urls):
    if name == "presentation":
        return matrix(urls, presentation_feats)
    if name == "presentation_https":
        return np.hstack([matrix(urls, presentation_feats), matrix(urls, scheme_feats)])
    if name == "structure":
        return matrix(urls, structure_feats)
    if name == "lexical":
        return np.hstack([matrix(urls, lexical_feats), matrix(urls, structure_feats)])
    if name == "char_raw":
        return list(urls)
    if name == "char_canon":
        return [canonical(u) for u in urls]
    if name == "char_host":
        return [host_view(u) for u in urls]
    raise KeyError(name)


def make(name, seed):
    if name.startswith("char"):
        return make_pipeline(TfidfVectorizer(**TFIDF), LogisticRegression(random_state=seed, **LR))
    return HistGradientBoostingClassifier(random_state=seed, **HGB)


def scores(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out = {"mcc": float(matthews_corrcoef(y, yhat)), "f1": float(f1_score(y, yhat, zero_division=0)),
           "precision": float(precision_score(y, yhat, zero_division=0)), "recall": float(recall_score(y, yhat, zero_division=0)),
           "fpr": float(fp / max(1, fp + tn)), "acc": float((tp + tn) / len(y)),
           "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    out["auc"] = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    # best-case operating points read off the test ROC: highest TPR with FPR at or below the target
    if len(set(y)) == 2:
        f, t, _ = roc_curve(y, p)
        for a in (0.001, 0.01):
            out[f"tpr_at_fpr{a}"] = float(t[f <= a].max()) if (f <= a).any() else 0.0
            out[f"fpr_used{a}"] = float(f[f <= a][np.argmax(t[f <= a])]) if (f <= a).any() else 0.0
    return out
