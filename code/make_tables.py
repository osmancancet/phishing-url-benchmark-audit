"""Write LaTeX tables and number macros from results/*.json. Run after analyze.py."""
import json, os, re
import numpy as np
from data import CORPORA, FILES

R = os.environ.get("RESULTS_DIR", os.path.join(os.path.dirname(__file__), "..", "results"))
M = os.path.join(os.path.dirname(__file__), "..", "manuscript")
T = os.path.join(M, "tables"); os.makedirs(T, exist_ok=True)
N = json.load(open(os.path.join(R, "numbers.json")))
A = N["audit"]; P = json.load(open(os.path.join(R, "provenance.json")))
V = json.load(open(os.path.join(R, "versions.json"))) if os.path.exists(os.path.join(R, "versions.json")) else {}
UF = os.path.join(R, "unique_counts.json")
if not os.path.exists(UF):
    from data import load
    json.dump({c: int(len(load(c))) for c in CORPORA}, open(UF, "w"))
UNIQ = json.load(open(UF))
META = {
    "Ebbu2017": ("2019", "Yandex search results", "PhishTank", "sahingoz2019"),
    "PhishStorm": ("2014", "--", "--", "marchal2014"),
    "ISCX-URL2016": ("2016", "crawl of top domains", "--", "mamun2016"),
    "PT-Alexa": ("constructed", "Alexa top domains", "PhishTank dump", "bozkir2023"),
    "Hannousse2021": ("2021", "--", "--", "hannousse2021"),
    "PhiUSIIL": ("2024", "--", "--", "prasad2024"),
    "Kaggle-MU": ("2021", "--", "--", "siddhartha2021"),
}
f3 = lambda x: f"{x:.3f}".replace("-", "$-$")
pct = lambda x: f"{100*x:.1f}"
fp_ = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
macros = {}
cname = lambda c: re.sub(r"[^A-Za-z]", "", c)
mname = lambda m: re.sub(r"[^A-Za-z]", "", m.title())


def mac(k, v):
    # a leading hyphen in a signed number becomes a true minus sign
    if isinstance(v, str) and re.fullmatch(r"-\d[\d.,]*", v):
        v = "$-$" + v[1:]
    macros[k] = v


def w(name, s):
    open(os.path.join(T, name + ".tex"), "w").write(s)


def table(name, caption, label, colspec, header, rows, resize=True, extra=""):
    tab = "\\begin{tabular}{" + colspec + "}\n\\toprule\n" + header + " \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}"
    if resize:
        tab = "\\begin{adjustbox}{max width=\\linewidth}\n" + tab + "\n\\end{adjustbox}"
    w(name, "\\begin{table}[tp]\n\\centering\n\\caption{" + caption + "}\n\\label{" + label + "}\n\\footnotesize\n" + extra + tab + "\n\\end{table}\n")


# corpora
rows = []
for c in CORPORA:
    a = A[c]; y, ls, ps, ref = META[c]
    lab = "PT-Alexa (this study)" if c == "PT-Alexa" else f"{c} \\citep{{{ref}}}"
    rows.append(f"{lab} & {y} & {ls} & {ps} & {a['n_raw']:,} & {a['exact_dup_removed']+a['raw_label_conflicts']:,} & {UNIQ[c]:,} & {pct(a['n_phish']/a['n'])} \\\\")
table("corpora", "The seven corpora. Year: publication year of the corpus paper or release. Sources are listed only where a checked publication or the construction script documents them (--: not documented). Removed: exact duplicate strings plus strings carrying both labels. Unique: canonical strings with a single label, the unit used in all experiments; Raw $-$ Removed $-$ Unique is the number of canonical duplicates and canonical conflicts (Table~\\ref{tab:overlap}). Phish: share of phishing URLs after exact de-duplication (\\%).",
      "tab:corpora", "@{}lllp{2.4cm}rrrr@{}", "Corpus & Year & Legitimate source & Phishing source & Raw & Removed & Unique & Phish", rows)

# provenance
USES = {"pdd": "Ebbu2017", "GramBeddings": "PhishStorm, ISCX-URL2016, PT-Alexa, Kaggle-MU", "PhiUSIIL-UCI-zip": "PhiUSIIL", "Hannousse-mirror": "Hannousse2021"}
rows = [f"\\url{{{v['url'].replace('https://github.com/', '')}}} & \\texttt{{{v['commit'][:12]}}} & {USES.get(k, '')} \\\\" for k, v in P["repos"].items()]
nfiles = sum(len(v) for v in P["files"].values())
mac("NSourceFiles", str(nfiles))
table("provenance", "Source mirrors, commits and the corpora taken from each. The PhiUSIIL file is a copy of the UCI archive stored in the listed repository; the PhishStorm and Kaggle-MU files are copies inside GramBeddings, whose reference files point to a PhishStorm re-implementation and to the Kaggle page; SHA-256 digests of all source files are in \\texttt{results/provenance.json}.",
      "tab:prov", "@{}>{\\raggedright\\arraybackslash}p{6.4cm}lp{3.6cm}@{}", "Repository (github.com/) & Commit & Corpora", rows, resize=False)

# artefacts by class
cols = [("has_scheme", "pres"), ("has_www", "pres"), ("trailing_slash", "pres"), ("html_entity", "pres"), ("is_https", "pres"), ("has_path", "struct")]
rows = []
for c in CORPORA:
    a = A[c]; cells = []
    for k, g in cols:
        cells += [pct(a[f"{g}_legit"][k]), pct(a[f"{g}_phish"][k])]
    cells += [f"{a['struct_legit']['url_len']:.0f}", f"{a['struct_phish']['url_len']:.0f}"]
    rows.append(c + " & " + " & ".join(cells) + " \\\\")
hdr = (" & \\multicolumn{2}{c}{Scheme} & \\multicolumn{2}{c}{www.} & \\multicolumn{2}{c}{Trailing /} & \\multicolumn{2}{c}{\\texttt{\\&amp;}} & \\multicolumn{2}{c}{https} & \\multicolumn{2}{c}{Has path} & \\multicolumn{2}{c}{Length} \\\\\n"
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}\\cmidrule(lr){14-15}\n"
       "Corpus & L & P & L & P & L & P & L & P & L & P & L & P & L & P")
table("artefacts", "Presentation, scheme-type and structure properties by class (L: legitimate, P: phishing), computed on the exact-de-duplicated files. Columns give the percentage of URLs with the property, except the last pair, which gives the median canonical length in characters.",
      "tab:artefacts", "@{}l*{14}{r}@{}", hdr, rows, extra="\\setlength{\\tabcolsep}{3pt}\n")

# overlap
rows = []
for c in CORPORA:
    a = A[c]; ds = N["split_meta"][c]["random"]["seen"][0]
    rows.append(f"{c} & {a['exact_dup_removed']:,} & {a['raw_label_conflicts']:,} & {a['canon_dup_rows']:,} & {a['canon_label_conflict_keys']:,} & {a['reg_domains_legit']:,} & {a['reg_domains_phish']:,} & {a['reg_domains_both_labels']:,} & {pct(ds)} \\\\")
table("overlap", "Duplicates, collisions and domain structure inside each corpus. Exact dup.: repeated raw strings with the same label. Raw conflict: rows whose raw string carries both labels. Canon.\\ dup.: further rows that collapse after canonicalisation; Canon.\\ conflict: canonical strings with both labels. Reg.\\ L/P: registered domains in the legitimate and phishing class (PSL private suffixes treated as public). Mixed: registered domains with URLs of both classes. Seen: share of test URLs in a random 80/20 split whose registered domain occurs in training (mean over 10 seeds, \\%).",
      "tab:overlap", "@{}lrrrrrrrr@{}", "Corpus & Exact dup. & Raw conflict & Canon.\\ dup. & Canon.\\ conflict & Reg.\\ L & Reg.\\ P & Mixed & Seen", rows)

# within tables
models = ["presentation", "presentation_https", "structure", "lexical", "char_raw", "char_canon", "char_host"]
mh = ["Pres.", "Pres.+https", "Structure", "Lexical", "Char raw", "Char canon.", "Char host"]
for split in ["random", "domain"]:
    rows = []
    for c in CORPORA:
        cells = []
        for m in models:
            v = N["within"][c][split].get(m)
            cells.append(f"{v['mcc'][0]:.3f}\\,$\\pm$\\,{v['mcc'][1]:.3f}" if v else "--")
        rows.append(c + " & " + " & ".join(cells) + " \\\\")
    cap = ("Within-corpus MCC (mean\\,$\\pm$\\,95\\% CI half-width over 10 repeated splits), " +
           ("stratified random 80/20 split." if split == "random" else "stratified 80/20 split with disjoint registered domains (first fold of a shuffled stratified group 5-fold partition).") +
           " Pres.: the five presentation flags; Pres.+https adds the scheme-type flag.")
    table(f"within_{split}", cap, f"tab:within_{split}", "@{}l" + "r" * len(models) + "@{}", "Corpus & " + " & ".join(mh), rows, extra="\\setlength{\\tabcolsep}{3pt}\n")

# paired tests
names = {"raw_vs_canon": "Char raw $-$ canonical (random split)", "raw_vs_host": "Char raw $-$ host view (random split)",
         "canon_random_vs_domain": "Random $-$ grouped split, char canonical", "lexical_random_vs_domain": "Random $-$ grouped split, lexical",
         "host_random_vs_domain": "Random $-$ grouped split, char host view"}
rows = []
for k, lab in names.items():
    rows.append(r"\multicolumn{5}{@{}l}{\emph{" + lab + r"}} \\")
    for c in CORPORA:
        t = N["tests"][k][c]
        dm, hm = t['mcc']['diff_corr']; da, ha = t['auc']['diff_corr']
        sg = lambda x: f"{x:+.3f}".replace("-", "$-$")
        rows.append(f"\\quad {c} & {sg(dm)} [{sg(dm-hm)}, {sg(dm+hm)}] & {fp_(t['mcc']['p_holm'])} & {sg(da)} [{sg(da-ha)}, {sg(da+ha)}] & {fp_(t['auc']['p_holm'])} \\\\")
        mac(f"DLo{mname(k.replace('_', ' '))}{cname(c)}", f"{dm-hm:+.3f}"); mac(f"DHi{mname(k.replace('_', ' '))}{cname(c)}", f"{dm+hm:+.3f}")
table("tests", "Differences paired by seed over 10 repeated splits: mean MCC and ROC-AUC difference with a 95\\% interval that uses the variance correction of \\citet{nadeau2003}, and a Holm-adjusted $p$-value from the corrected resampled $t$-test across the seven corpora of each comparison. For the random-versus-grouped comparisons the two settings share the sample of a seed but not the split. Differences smaller than 0.0005 in absolute value print as $+$0.000; small $p$-values for such rows reflect very small seed variance, not a material effect.",
      "tab:tests", "@{}lrrrr@{}", "Comparison and corpus & $\\Delta$MCC [95\\% CI] & $p_\\mathrm{Holm}$ & $\\Delta$AUC [95\\% CI] & $p_\\mathrm{Holm}$", rows, resize=True, extra="\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n")

# probe ablation
if "probe" in N:
    PR = N["probe"]; keys = ["all", "all+https", "drop_has_scheme", "drop_has_www", "drop_trailing_slash", "drop_html_entity", "drop_upper_in_host", "path_subset"]
    hh = ["All five", "+https", "$-$scheme", "$-$www.", "$-$slash", "$-$\\texttt{\\&amp;}", "$-$case", "Path subset"]
    rows = []
    for c in CORPORA:
        cells = [f"{PR[c][k][0]:.3f}" if k in PR[c] else "--" for k in keys]
        n0, n1 = PR[c]["path_subset_n"]
        rows.append(c + " & " + " & ".join(cells) + f" & {n0:,}/{n1:,} \\\\")
        for k in keys:
            if k in PR[c]:
                mac(f"Probe{cname(c)}{mname(k.replace('+', ' plus ').replace('_', ' '))}", f3(PR[c][k][0]))
    table("probe", "Presentation-probe ablation (MCC, random split, mean over 10 seeds). $-$X: probe without flag X (scheme: explicit scheme; slash: string ends in a slash; case: upper-case letter in the host). Path subset: probe trained and tested only on URLs that have a path, which removes the homepage-sampling contrast; --: fewer than 500 such URLs in one class. Last column: legitimate/phishing URLs with a path.",
          "tab:probe", "@{}l" + "r" * (len(keys) + 1) + "@{}", "Corpus & " + " & ".join(hh) + " & $n$ with path (L/P)", rows, extra="\\setlength{\\tabcolsep}{3pt}\n")

# CNN
if "cnn" in N:
    rows = []
    for c in CORPORA:
        K = N["cnn"].get(c, {}); cells = []
        for sp in ["random", "domain"]:
            for v in ["raw", "canon"]:
                e = K.get(sp, {}).get(v)
                cells.append(f"{e['mcc'][0]:.3f} ({e['mcc_sd']:.3f})" if e else "--")
            lr = N["within"][c][sp]
            cells.append(f"{lr['char_raw']['mcc'][0]:.3f} / {lr['char_canon']['mcc'][0]:.3f}")
        rows.append(c + " & " + " & ".join(cells) + " \\\\")
        for sp in ["random", "domain"]:
            for v in ["raw", "canon"]:
                e = K.get(sp, {}).get(v)
                if e:
                    mac(f"CNN{cname(c)}{sp.title()}{v.title()}", f3(e["mcc"][0])); mac(f"CNNSD{cname(c)}{sp.title()}{v.title()}", f3(e["mcc_sd"]))
                    mac(f"CNNMin{cname(c)}{sp.title()}{v.title()}", f3(e["mcc_min"])); mac(f"CNNMax{cname(c)}{sp.title()}{v.title()}", f3(e["mcc_max"]))
            if "raw_minus_canon" in K.get(sp, {}):
                mac(f"CNNDrop{cname(c)}{sp.title()}", f3(K[sp]["raw_minus_canon"][0]))
        if K.get("random", {}).get("canon") and K.get("domain", {}).get("canon"):
            mac(f"CNNGroupCost{cname(c)}", f3(K["random"]["canon"]["mcc"][0] - K["domain"]["canon"]["mcc"][0]))
    hdr = (" & \\multicolumn{3}{c}{Random split} & \\multicolumn{3}{c}{Grouped split} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
           "Corpus & CNN raw & CNN canon. & LR raw / canon. & CNN raw & CNN canon. & LR raw / canon.")
    secs = [N["cnn"][c][sp][v]["sec"] for c in N["cnn"] for sp in N["cnn"][c] for v in ("raw", "canon") if v in N["cnn"][c][sp]]
    mac("CNNSecMean", f"{np.mean(secs):.0f}")
    table("cnn", "Character-level CNN (URLNet character branch) against the character n-gram logistic regression (LR): MCC, mean (standard deviation) over seeds 0--2 for the CNN and mean over seeds 0--9 for LR, same caps and split procedure. The CNN is trained for a fixed five epochs without a validation set.",
          "tab:cnn", "@{}lrrrrrr@{}", hdr, rows)

# cross tables (purge level 'url' primary, 'domain' secondary)
if "cross" in N:
    xm = ["presentation", "presentation_https", "lexical", "char_raw", "char_canon", "char_host"]
    xh = ["Pres.", "Pres.+https", "Lexical", "Char raw", "Char canon.", "Char host"]
    LROWS = {}
    for pu in ["url", "domain"]:
        X = N["cross"]["pair"][pu]
        rows = []
        for tr in CORPORA:
            cells = []
            for m in xm:
                d = X[m][tr]
                cells.append(f"{np.mean([d[te]['auc'][0] for te in d]):.3f} ({np.mean([d[te]['mcc'][0] for te in d]):.3f})")
            rows.append(tr + " & " + " & ".join(cells) + " \\\\")
        cells = []
        for m in xm:
            d = X[m]
            au = [d[a][b]["auc"][0] for a in d for b in d[a]]; mc = [d[a][b]["mcc"][0] for a in d for b in d[a]]
            cells.append(f"{np.mean(au):.3f} ({np.mean(mc):.3f})")
            mac(f"X{pu.title()}NPairs", str(len(au)))
            mac(f"X{pu.title()}{mname(m)}MeanAUC", f3(np.mean(au))); mac(f"X{pu.title()}{mname(m)}MeanMCC", f3(np.mean(mc)))
            mac(f"X{pu.title()}{mname(m)}MinMCC", f3(np.min(mc))); mac(f"X{pu.title()}{mname(m)}MaxMCC", f3(np.max(mc)))
            mac(f"X{pu.title()}{mname(m)}MinAUC", f3(np.min(au))); mac(f"X{pu.title()}{mname(m)}MaxAUC", f3(np.max(au)))
        rows.append(f"\\midrule\nAll {len(au)} pairs & " + " & ".join(cells) + " \\\\")
        cap = ("Cross-corpus transfer: ROC-AUC (MCC at threshold 0.5) averaged over the test corpora of each training corpus (five for Kaggle-MU, whose ISCX-URL2016 test set is empty after purging), and over all ordered pairs that keep both classes after purging; three seeds. Purging changes the size and class prior of a test set, which affects MCC but not ROC-AUC; per-pair sizes and priors are stored with the results. " +
               ("Test URLs whose canonical form occurs anywhere in the full training corpus are removed." if pu == "url" else "Test URLs whose canonical form or registered domain occurs anywhere in the full training corpus are removed."))
        table(f"pair_{pu}", cap, f"tab:pair_{pu}", "@{}l" + "r" * len(xm) + "@{}", "Training corpus & " + " & ".join(xh), rows)
        L = N["cross"]["loco"][pu]
        rows = []
        for c in CORPORA:
            cells = []
            for m in xm:
                v = L[m].get(c, {}).get(c)
                if v is None and m == xm[0]:
                    break
                cells.append(f"{v['auc'][0]:.3f} ({v['mcc'][0]:.3f})" if v else "--")
                if v:
                    mac(f"L{pu.title()}{cname(c)}{mname(m)}", f3(v["mcc"][0])); mac(f"LA{pu.title()}{cname(c)}{mname(m)}", f3(v["auc"][0]))
            if cells:
                rows.append(c + " & " + " & ".join(cells) + " \\\\")
        LROWS[pu] = rows
        if pu == "domain":
            rows = [r"\multicolumn{7}{@{}l}{\emph{URL purge}} \\"] + LROWS["url"] + [r"\midrule", r"\multicolumn{7}{@{}l}{\emph{Domain purge}} \\"] + LROWS["domain"]
            table("loco", "Leave-one-corpus-out: ROC-AUC (MCC at 0.5) on the held-out corpus for a detector trained on the other six; three seeds. URL purge: test URLs whose canonical form occurs in any training corpus are removed; domain purge: also those whose registered domain occurs there. ISCX-URL2016 has no row because every legitimate ISCX-URL2016 URL also occurs in Kaggle-MU, so the purged test set has one class.",
                  "tab:loco", "@{}l" + "r" * len(xm) + "@{}", "Held-out corpus & " + " & ".join(xh), rows)
    for k, v in N.get("cross_tests", {}).items():
        nm = mname(k.replace("_", " "))
        mac(f"XT{nm}", f"{v['diff_pairs'][0]:+.3f}"); mac(f"XT{nm}Better", str(v["n_better_pairs"]))
        mac(f"XT{nm}Corpus", f"{v['diff_corpus'][0]:+.3f}"); mac(f"XT{nm}CorpusLo", f"{v['diff_corpus'][0]-v['diff_corpus'][1]:+.3f}")
        mac(f"XT{nm}CorpusHi", f"{v['diff_corpus'][0]+v['diff_corpus'][1]:+.3f}"); mac(f"XT{nm}P", f"{v['p_corpus']:.3f}")
        mac(f"XT{nm}CorpusBetter", str(v["n_better_corpus"]))
    # specific cells used in text
    for pu in ["url", "domain"]:
        for m in ["char_raw", "char_canon", "char_host"]:
            for tr, te in [("Kaggle-MU", "ISCX-URL2016"), ("PhiUSIIL", "Kaggle-MU"), ("PT-Alexa", "PhiUSIIL"), ("Ebbu2017", "PhiUSIIL"), ("Kaggle-MU", "PhiUSIIL")]:
                v = N["cross"]["pair"][pu][m][tr].get(te)
                if v is None:
                    mac(f"XN{pu.title()}{cname(tr)}{cname(te)}", "0"); continue
                mac(f"XC{pu.title()}{mname(m)}{cname(tr)}{cname(te)}", f3(v["auc"][0])); mac(f"XN{pu.title()}{cname(tr)}{cname(te)}", f"{v['n_test']:,}")

for d, g in N.get("cross_time", {}).items():
    for m, v in g.items():
        mac(f"XTime{d.title()}{mname(m)}", f3(v["auc"])); mac(f"XTime{d.title()}NPairs", str(v["n_pairs"]))

# n-grams
if os.path.exists(os.path.join(R, "explain.json")):
    E = json.load(open(os.path.join(R, "explain.json")))
    def tt(s):
        for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("~", "\\~{}"), ("$", "\\$"), ("{", "\\{"), ("}", "\\}"), ("^", "\\^{}")]:
            s = s.replace(a, b)
        return "\\texttt{" + s.replace(" ", "\\textvisiblespace{}") + "}"
    rows = [c + " & " + ", ".join(tt(n) for n, _ in E[c]["phish"][:6]) + " & " + ", ".join(tt(n) for n, _ in E[c]["legit"][:6]) + " \\\\" for c in CORPORA]
    w("ngrams", "\\begin{table}[tp]\n\\centering\n\\caption{Character n-grams with the largest positive (phishing) and negative (legitimate) weights in the raw-string model (seed 0, full training cap).}\n\\label{tab:ngrams}\n\\footnotesize\n\\begin{tabularx}{\\linewidth}{@{}lXX@{}}\n\\toprule\nCorpus & Top phishing n-grams & Top legitimate n-grams \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabularx}\n\\end{table}\n")


# operating points and base rates: thresholds chosen on a validation split (round 3), char raw / canonical, random split
for c in CORPORA:
    for sp in ["random", "domain"]:
        for m in ["char_raw", "char_canon"]:
            e = N["within"][c][sp][m]
            mac(f"PrecTen{cname(c)}{sp.title()}{mname(m)}", f3(e["prec_at_10pct"][0]))
            mac(f"FPZero{cname(c)}{sp.title()}{mname(m)}", str(e["fp_zero_seeds"])); mac(f"NLegTest{cname(c)}", f"{e['n_legit_test']:,}")
VA = N.get("r3", {}).get("valthr")
if VA:
    rows = []
    for c in CORPORA:
        cells = []
        for m in ["char_raw", "char_canon"]:
            e = VA[c][m]["0.001"]; e5 = VA[c][m]["0.5"]
            cells += [f"{100*e['fpr']:.2f}", f3(e["tpr"]), f3(e["prec_1pct"]), f3(e["prec_01pct"]), f3(e5["prec_1pct"])]
            k = f"{cname(c)}{mname(m)}"
            mac(f"VFpr{k}", f"{100*e['fpr']:.2f}"); mac(f"VTpr{k}", f3(e["tpr"])); mac(f"VPrecOne{k}", f3(e["prec_1pct"]))
            mac(f"VPrecTenth{k}", f3(e["prec_01pct"])); mac(f"VFP{k}", f"{e['fp_total']:,}"); mac(f"VNZero{k}", f"{e['n0_total']:,}")
            mac(f"VFprOnePct{k}", f"{100*VA[c][m]['0.01']['fpr']:.2f}"); mac(f"VPrecOneAtOnePct{k}", f3(VA[c][m]["0.01"]["prec_1pct"]))
        rows.append(c + " & " + " & ".join(cells) + " \\\\")
    hdr = (r" & \multicolumn{5}{c}{Char raw} & \multicolumn{5}{c}{Char canon.} \\" "\n" r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}" "\n"
           r" & \multicolumn{4}{c}{Validation threshold} & @0.5 & \multicolumn{4}{c}{Validation threshold} & @0.5 \\" "\n"
           r"\cmidrule(lr){2-5}\cmidrule(lr){6-6}\cmidrule(lr){7-10}\cmidrule(lr){11-11}" "\n"
           r"Corpus & FPR (\%) & TPR & Prec.\ 1\% & Prec.\ 0.1\% & Prec.\ 1\% & FPR (\%) & TPR & Prec.\ 1\% & Prec.\ 0.1\% & Prec.\ 1\%")
    table("precision", "Test-set error rates and the precision a within-corpus detector would reach at 1\\% and 0.1\\% phishing prevalence (random split, 10 seeds). Validation threshold: 20\\% of the training part is held out, the model is fitted on the rest, and the threshold with the highest validation true-positive rate at a validation false-positive rate of at most 0.1\\% is applied to the test part; FPR and TPR are the realised test rates. @0.5: the model fitted on the full training part at threshold 0.5. Rates are pooled over seeds before precision is computed.",
          "tab:precision", "@{}l" + "r" * 10 + "@{}", hdr, rows, extra="\\setlength{\\tabcolsep}{3pt}\n")
XA = N.get("r3", {}).get("xthr")
if XA:
    for m, d in XA.items():
        k = mname(m)
        mac(f"XVFpr{k}", f"{100*d['0.001']['median_fpr']:.1f}"); mac(f"XVTpr{k}", f3(d["0.001"]["median_tpr"]))
        mac(f"XVPrecOne{k}", f3(d["0.001"]["median_prec1"])); mac(f"XVPrecTenth{k}", f3(d["0.001"]["median_prec01"]))
        mac(f"XVOverTen{k}", str(d["0.001"]["n_fpr_over_10x"])); mac(f"XVNPairs{k}", str(d["0.001"]["n_pairs"]))
        mac(f"XVFprOnePct{k}", f"{100*d['0.01']['median_fpr']:.1f}"); mac(f"XVOverTenOnePct{k}", str(d["0.01"]["n_fpr_over_10x"]))
        mac(f"XVFprHalf{k}", f3(d["0.5"]["median_fpr"])); mac(f"XVPrecOneHalf{k}", f3(d["0.5"]["median_prec1"]))

# controls added in review round 2: homepage subset and canonical + https token
EX = N.get("extra", {})
if EX:
    rows = []
    for c in CORPORA:
        h = EX.get("homepage", {}).get(c, {}); n = h.get("n", [0, 0])
        cells = [f"{n[0]:,}/{n[1]:,}"] + [f3(h[m]["mcc"][0]) if m in h else "--" for m in ["presentation_https", "char_raw", "char_canon"]]
        ch = EX.get("canon_https", {}).get(c, {}).get("char_canon_https")
        cells += [f3(N["within"][c]["random"]["char_canon"]["mcc"][0]), f3(ch["mcc"][0]) if ch else "--"]
        rows.append(c + " & " + " & ".join(cells) + " \\\\")
        for m in ["presentation_https", "char_raw", "char_canon"]:
            if m in h:
                mac(f"Home{cname(c)}{mname(m)}", f3(h[m]["mcc"][0]))
        mac(f"HomeN{cname(c)}", f"{n[0]:,}/{n[1]:,}")
        if ch:
            mac(f"CanonHttps{cname(c)}", f3(ch["mcc"][0]))
            mac(f"CanonHttpsGain{cname(c)}", f"{ch['mcc'][0] - N['within'][c]['random']['char_canon']['mcc'][0]:+.3f}".replace("-0.000", "+0.000"))
    hdr = (r" & \multicolumn{4}{c}{Homepage subset (no path in either class)} & \multicolumn{2}{c}{Full corpus} \\" "\n" r"\cmidrule(lr){2-5}\cmidrule(lr){6-7}" "\n"
           r"Corpus & $n$ (L/P) & Pres.+https & Char raw & Char canon. & Char canon. & Canon.+https")
    table("controls", "Controls (MCC, random split, mean over 10 seeds). Homepage subset: only URLs without a path, so that both classes are sampled at the same granularity; $n$: legitimate and phishing homepages in the capped sample; --: fewer than 500 such URLs in one class. Full corpus: the canonical model of Table~\\ref{tab:within_random} and the same model with a token for an explicit \\texttt{https} scheme prepended to the canonical string.",
          "tab:controls", "@{}lrrrrrr@{}", hdr, rows)
    # Kaggle-MU subsets and the PT-Alexa time split (rounds 2 and 3)
    rows = []
    kc = EX.get("kaggle_clean", {}).get("Kaggle-MU", {}); R3 = N.get("r3", {})
    wk = N["within"]["Kaggle-MU"]["random"]
    rows.append(f"Kaggle-MU, full & {UNIQ['Kaggle-MU']:,} in all & " + " & ".join(f3(wk[m]["mcc"][0]) for m in ["presentation_https", "char_raw", "char_canon"]) + " & -- \\\\")
    if kc:
        info = N["kaggle_clean_info"]
        rows.append(f"Kaggle-MU-clean & {info['n_after_legit']:,}/{info['n_after_phish']:,} & " + " & ".join(f3(kc[m]["mcc"][0]) for m in ["presentation_https", "char_raw", "char_canon"]) + " & -- \\\\")
        for m in ["presentation", "presentation_https", "char_raw", "char_canon"]:
            mac(f"KClean{mname(m)}", f3(kc[m]["mcc"][0]))
        mac("KCleanDrop", f3(kc["char_raw"]["mcc"][0] - kc["char_canon"]["mcc"][0]))
        for k, v in info.items():
            mac(f"KClean{mname(k.replace('_', ' '))}", f"{v:,}")
    k2 = R3.get("kaggle_clean2")
    if k2:
        i2 = k2["info"]
        rows.append(f"Kaggle-MU-clean without bare-\\texttt{{www.}} phishing & {i2['n_after_legit']:,}/{i2['n_after_phish']:,} & " + " & ".join(f3(k2[m]["mcc"][0]) for m in ["presentation_https", "char_raw", "char_canon"]) + " & -- \\\\")
        for m in ["presentation", "presentation_https", "char_raw", "char_canon"]:
            mac(f"KCleanTwo{mname(m)}", f3(k2[m]["mcc"][0]))
        mac("KCleanTwoDrop", f3(k2["char_raw"]["mcc"][0] - k2["char_canon"]["mcc"][0]))
        for k, v in i2.items():
            mac(f"KCleanTwo{mname(k.replace('_', ' '))}", f"{v:,}")
        kcl = N["kaggle_clean_info"]["n_after_phish"]
        mac("KCleanBareWwwPct", pct(i2["n_bare_www_phish"] / kcl))
    for kind, lab in [("temporal_random", "PT-Alexa, random years (matched)"), ("temporal", "PT-Alexa, train $\\le$2019, test 2020")]:
        t = R3.get(kind)
        if t:
            n = t["n"]
            rows.append(f"{lab} & {int(n['train_phish']):,}/{int(n['test_phish']):,} & " + " & ".join(f3(t[m]["mcc"][0]) for m in ["presentation_https", "char_raw", "char_canon", "char_canon_https"]) + " \\\\")
            kk = "Time" if kind == "temporal" else "TimeRand"
            for m in ["presentation", "presentation_https", "char_raw", "char_canon", "char_canon_https"]:
                mac(f"{kk}{mname(m)}", f3(t[m]["mcc"][0]))
            mac(f"{kk}TrainHttps", pct(n["train_https"])); mac(f"{kk}TestHttps", pct(n["test_https"]))
            mac(f"{kk}NTrain", f"{int(n['train_phish']):,}"); mac(f"{kk}NTest", f"{int(n['test_phish']):,}")
    if R3.get("temporal") and R3.get("temporal_random"):
        for m in ["char_canon", "char_canon_https"]:
            mac(f"TimeGap{mname(m)}", f3(R3["temporal_random"][m]["mcc"][0] - R3["temporal"][m]["mcc"][0]))
    if len(rows) > 1:
        rows.insert(3 if k2 else 2, "\\midrule")
        hdr = r"Control & $n$ & Pres.+https & Char raw & Char canon. & Canon.+https"
        table("controls2", "Kaggle-MU subsets and a time split (MCC, mean over seeds). Kaggle-MU rows: random split, 10 seeds, at most 25,000 URLs per class; $n$: legitimate/phishing rows before the cap. Kaggle-MU-clean drops registered domains with both labels and phishing URLs that another corpus labels legitimate; the next row also drops phishing URLs that start with a bare \\texttt{www.} (the rule also removes phishing URLs on Alexa top-1,000 domains, but none remain after the first rule). PT-Alexa rows: 3 seeds; $n$: phishing URLs for training/testing, with as many legitimate URLs drawn at random; the time split trains on phishing URLs submitted up to 2019 and tests on 2020 submissions, the matched control draws the same numbers from all years.",
              "tab:controls2", "@{}>{\\raggedright\\arraybackslash}p{4.2cm}rrrrr@{}", hdr, rows, resize=True, extra="\\setlength{\\tabcolsep}{3pt}\n")

# feature attribution (round 4, XAI)
SH = N.get("shap")
if SH:
    rows = []
    for c in CORPORA:
        e = SH[c]; g = e["tree_group"]
        rows.append(f"{c} & {f3(e['tree_mcc'][0])} & {f3(g['presentation'][0])} & {f3(g['https'][0])} & {f3(g['structure'][0])} & {f3(g['lexical'][0])} & {f3(e['tree_same_drop'][0])} & {f3(e['tree_dep'])} & {f3(e['char_own'][0])} & {f3(e['char_sym'][0])} & {f3(e['char_same_drop'][0])} & {f3(e['char_dep'])} \\\\")
        k = cname(c)
        mac(f"ShTreeFmt{k}", f3(e["tree_fmt"][0])); mac(f"ShTreeDep{k}", f3(e["tree_dep"])); mac(f"ShTreeMCC{k}", f3(e["tree_mcc"][0]))
        mac(f"ShPres{k}", f3(e["pres_only"][0])); mac(f"ShHttps{k}", f3(g["https"][0]))
        mac(f"ShCharOwn{k}", f3(e["char_own"][0])); mac(f"ShCharDep{k}", f3(e["char_dep"])); mac(f"ShCharFmtOnly{k}", f3(e["char_fmt_only"][0]))
        mac(f"ShCharOwnLegit{k}", f3(e["char_own_legit"][0])); mac(f"ShCharSymLegit{k}", f3(e["char_sym_legit"][0]))
        mac(f"ShCharSym{k}", f3(e["char_sym"][0])); mac(f"ShCharAbsent{k}", f3(e["char_absent"][0]))
        mac(f"ShTreeSame{k}", f3(e["tree_same_drop"][0])); mac(f"ShCharSame{k}", f3(e["char_same_drop"][0]))
        mac(f"ShVocab{k}", f"{int(round(e['char_vocab'])):,}")
        mac(f"ShTop{k}", e["tree_top"][0][0].replace("_", "\\_")); mac(f"ShTopShare{k}", f3(e["tree_top"][0][1]))
    for nm, key in (("Tree", "_spearman_tree"), ("Char", "_spearman_char"), ("TreeNoPT", "_spearman_tree_no_pt"), ("CharSym", "_spearman_char_sym"),
                    ("TreeSame", "_spearman_tree_same"), ("CharSymSame", "_spearman_char_sym_same"), ("CharSame", "_spearman_char_same")):
        mac(f"ShRho{nm}", f"{SH[key]:.2f}"); mac(f"ShRho{nm}P", ("$<$0.001" if SH[key + "_p"] < 0.001 else f"{SH[key + '_p']:.3f}") if key + "_p" in SH else "")
    WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven"]
    mac("ShWwwTopThreeN", WORDS[sum(any(n == "has_www" for n, _ in SH[c]["tree_top"]) for c in CORPORA)])
    mac("ShMaxSeedRange", f"{SH['_max_seed_range']:.3f}")
    ab = [SH[c]["char_absent"][0] for c in CORPORA]; mac("ShAbsentMin", f3(min(ab))); mac("ShAbsentMax", f3(max(ab)))
    vb = [SH[c]["char_vocab"] for c in CORPORA]; mac("ShVocabMin", f"{int(round(min(vb))):,}"); mac("ShVocabMax", f"{int(round(max(vb))):,}")
    oth = [c for c in CORPORA if c not in ("PhiUSIIL", "PT-Alexa", "Kaggle-MU")]
    mac("ShTreeFmtMinOther", f3(min(SH[c]["tree_fmt"][0] for c in oth))); mac("ShTreeFmtMaxOther", f3(max(SH[c]["tree_fmt"][0] for c in oth)))
    mac("ShCharOwnMinOther", f3(min(SH[c]["char_own"][0] for c in oth))); mac("ShCharOwnMaxOther", f3(max(SH[c]["char_own"][0] for c in oth)))
    mac("ShCharFmtOnlyMax", f3(max(SH[c]["char_fmt_only"][0] for c in CORPORA)))
    hdr = (r" & \multicolumn{7}{c}{Tree, 30 features (TreeSHAP)} & \multicolumn{4}{c}{Char raw (linear SHAP)} \\" "\n"
           r"\cmidrule(lr){2-8}\cmidrule(lr){9-12}" "\n"
           r"Corpus & MCC & Pres. & https & Str. & Lex. & Drop & Loss & Own & Sym. & Drop & Loss")
    table("shap", "Feature attribution against two measures of dependence (random split, means over seeds 0--2, 2,000 explained test URLs per seed; MCC: Matthews correlation coefficient). Tree: gradient-boosted trees on the five presentation flags, the https flag, five structure and 19 lexical features, explained with TreeSHAP; Pres., https, Str., Lex.: shares of mean absolute SHAP value per feature group, where the formatting share used in the text is Pres.\\ plus https. Char raw: the raw-string n-gram model, explained with linear SHAP (training mean as baseline, features treated as independent, log-odds scale). Own: share of a URL's $|\\phi|$ on n-grams present in its raw string but absent from its own canonical string; Sym.: share on n-grams whose training document frequency in raw strings exceeds twice that in canonical strings plus five, whether present in the URL or not. Drop: MCC of the same fitted model on raw minus canonical test input. Loss: MCC of a detector retrained without formatting (the lexical model and the canonical n-gram model of Table~\\ref{tab:within_random}, same seeds) subtracted from that of the explained model. Shares and MCC vary by at most " + f"{SH['_max_seed_range']:.3f}" + " between seeds.",
          "tab:shap", "@{}l" + "r" * 11 + "@{}", hdr, rows, extra="\\setlength{\\tabcolsep}{2.5pt}\n")

CC = N.get("cross_conc")
if CC:
    mac("XInvN", str(CC["inverted"]["n_pairs"])); mac("XInvGain", f"{CC['inverted']['gain'][0]:+.3f}")
    mac("XOthN", str(CC["other"]["n_pairs"])); mac("XOthGain", f"{CC['other']['gain'][0]:+.3f}")
    wk = CC["without_kaggle"]
    mac("XNoKGain", f"{wk['diff_corpus'][0]:+.3f}"); mac("XNoKP", f"{wk['p_corpus']:.3f}"); mac("XNoKBetter", str(wk["n_better_corpus"])); mac("XNoKN", str(wk["n_corpus"]))
    mac("XNoKLo", f"{wk['diff_corpus'][0]-wk['diff_corpus'][1]:+.3f}"); mac("XNoKHi", f"{wk['diff_corpus'][0]+wk['diff_corpus'][1]:+.3f}")
    bt = CC.get("by_test")
    if bt:
        mac("XByTestGain", f"{bt['diff'][0]:+.3f}"); mac("XByTestP", f"{bt['p']:.3f}"); mac("XByTestBetter", str(bt["n_better"])); mac("XByTestN", str(bt["n"]))
        b2 = CC["by_test_without_kaggle"]
        mac("XByTestNoKGain", f"{b2['diff'][0]:+.3f}"); mac("XByTestNoKP", f"{b2['p']:.3f}"); mac("XByTestNoKBetter", str(b2["n_better"])); mac("XByTestNoKN", str(b2["n"]))
    for m, v in CC["prec1"].items():
        mac(f"XMedFPR{mname(m)}", f3(v["median_fpr"])); mac(f"XMedPrec{mname(m)}", f3(v["median_prec1"]))
PH = N.get("phishtank_https")
if PH:
    for y in ["2016", "2017", "2018", "2019", "2020"]:
        if y in PH:
            mac(f"PTHttps{['Sixteen','Seventeen','Eighteen','Nineteen','Twenty'][int(y)-2016]}", pct(PH[y]["https_share"]))
    early = [v for y, v in PH.items() if int(y) <= 2017]
    mac("PTHttpsEarly", pct(sum(v["https_share"] * v["n"] for v in early) / sum(v["n"] for v in early))); mac("PTHttpsEarlyN", f"{sum(v['n'] for v in early):,}")
    mac("PTNTwenty", f"{PH['2020']['n']:,}")
if "corpus_year" in N and "cross_time" in N:
    _ns = N["cross_time"].get("same", {}).get("char_raw", {}).get("n_pairs", 0)
    mac("XTimeSameNPairs", {1: "one", 2: "two", 3: "three", 4: "four"}.get(_ns, str(_ns)))
sh = A.get("_cross", {}).get("ISCX-URL2016|Kaggle-MU", {}).get("shared_canonical_urls")
if sh:
    mac("ISCXInKagglePct", f"{100*sh/UNIQ['ISCX-URL2016']:.0f}")
# hand-typed claims replaced by macros
up = max(max(A[c]["pres_legit"]["upper_in_host"], A[c]["pres_phish"]["upper_in_host"]) for c in CORPORA)
mac("UpperMaxPct", f"{100*up:.1f}")
amp = {c: A[c]["pres_legit"]["html_entity"] for c in CORPORA}
mac("AmpLegitMaxOther", f"{100*max(v for c, v in amp.items() if c != 'Kaggle-MU'):.3f}")

# scalar macros
for c in CORPORA:
    for split in ["random", "domain"]:
        for m in models:
            v = N["within"][c][split].get(m)
            if v:
                mac(f"W{cname(c)}{split.title()}{mname(m)}", f3(v["mcc"][0])); mac(f"A{cname(c)}{split.title()}{mname(m)}", f3(v["auc"][0]))
                mac(f"PrecOne{cname(c)}{split.title()}{mname(m)}", f3(v["prec_at_1pct"][0]))
    sm = N["split_meta"][c]
    mac(f"Seen{cname(c)}", pct(sm["random"]["seen"][0]))
    a = A[c]
    mac(f"Raw{cname(c)}", f"{a['n_raw']:,}"); mac(f"Uniq{cname(c)}", f"{UNIQ[c]:,}")
    mac(f"Dup{cname(c)}", f"{a['exact_dup_removed']:,}"); mac(f"DupPct{cname(c)}", pct(a['exact_dup_removed'] / a['n_raw']))
    mac(f"CanonDup{cname(c)}", f"{a['canon_dup_rows']:,}"); mac(f"CanonConf{cname(c)}", f"{a['canon_label_conflict_keys']:,}")
    mac(f"Mixed{cname(c)}", f"{a['reg_domains_both_labels']:,}")
    mac(f"RegLegit{cname(c)}", f"{a['reg_domains_legit']:,}"); mac(f"RegPhish{cname(c)}", f"{a['reg_domains_phish']:,}")
    mac(f"TopK{cname(c)}", pct(a["phish_on_alexa_top1k"]))
    if a.get("home_slash_legit") is not None and a.get("home_slash_phish") is not None:
        mac(f"HomeSlashLegit{cname(c)}", pct(a["home_slash_legit"])); mac(f"HomeSlashPhish{cname(c)}", pct(a["home_slash_phish"]))
    mac(f"DupLegit{cname(c)}", f"{a.get('exact_dup_removed_legit', 0):,}"); mac(f"DupPhish{cname(c)}", f"{a.get('exact_dup_removed_phish', 0):,}")
    mac(f"LargestLegit{cname(c)}", a["largest_group_legit"][0].replace("_", "\\_")); mac(f"LargestLegitPct{cname(c)}", pct(a["largest_group_legit"][2]))
    mac(f"LargestPhish{cname(c)}", a["largest_group_phish"][0].replace("_", "\\_")); mac(f"LargestPhishPct{cname(c)}", pct(a["largest_group_phish"][2]))
    for k in N["tests"]:
        mac(f"D{mname(k.replace('_', ' '))}{cname(c)}", f3(N["tests"][k][c]["mcc"]["diff"][0]))
        mac(f"DA{mname(k.replace('_', ' '))}{cname(c)}", f3(N["tests"][k][c]["auc"]["diff"][0]))
tk = {c: A[c]["phish_on_alexa_top1k"] for c in CORPORA if c != "Kaggle-MU"}
mac("TopKMinOther", pct(min(tk.values()))); mac("TopKMaxOther", pct(max(tk.values()))); mac("TopKMaxOtherCorpus", max(tk, key=tk.get))
pres = {c: N["within"][c]["random"]["presentation"]["mcc"][0] for c in CORPORA}
mac("PresMinMCC", f3(min(pres.values()))); mac("PresMinCorpus", min(pres, key=pres.get))
for k, v in A.get("_cross", {}).items():
    a, b = k.split("|")
    mac(f"XShared{cname(a)}{cname(b)}", f"{v['shared_canonical_urls']:,}"); mac(f"XConf{cname(a)}{cname(b)}", f"{v['shared_canonical_conflicting_label']:,}")
    mac(f"XConfAP{cname(a)}{cname(b)}", f"{v['conflict_a_phish_b_legit']:,}"); mac(f"XConfAL{cname(a)}{cname(b)}", f"{v['conflict_a_legit_b_phish']:,}")
    mac(f"XOpp{cname(a)}{cname(b)}", f"{v['reg_domains_opposite_pure_label']:,}")
RL = os.path.join(R, "run_all.log")
if os.path.exists(RL):
    m = re.search(r"real\s+(\d+)m", open(RL).read())
    if m:
        mac("RunMinutes", m.group(1))
KW = os.path.join(R, "kaggle_www.json")
if os.path.exists(KW):
    kw = json.load(open(KW))
    mac("KWwwKwShare", pct(kw["kw_share_bare_www"])); mac("KOtherKwShare", pct(kw["kw_share_other"]))
    mac("KWwwTopDomains", ", ".join("\\texttt{" + k + "}" for k in list(kw["top_domains"])[:4]))
R3L = os.path.join(R, "r3.log")
if os.path.exists(R3L):
    m = re.search(r"real\s+(\d+)m", open(R3L).read())
    if m:
        mac("RThreeMinutes", m.group(1))
SHL = os.path.join(R, "shap.log")
if os.path.exists(SHL):
    m = re.search(r"real\s+(\d+)m", open(SHL).read())
    if m:
        mac("ShapMinutes", m.group(1))
RT = os.path.join(R, "runtime.json")
if os.path.exists(RT):
    mac("CNNHours", f"{json.load(open(RT))['cnn_hours']:.1f}")
for k, v in V.items():
    mac(f"Ver{mname(k)}", v)
open(os.path.join(M, "macros.tex"), "w").write("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
json.dump(macros, open(os.path.join(R, "macros.json"), "w"), indent=0)
print(len(macros), "macros")
