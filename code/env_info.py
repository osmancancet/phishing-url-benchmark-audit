"""Record software versions, CPU and a SHA-256 of every code file, so that results can be tied to the code."""
import glob, hashlib, json, os, platform, subprocess
import numpy, pandas, scipy, sklearn, tldextract
R = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(R, exist_ok=True)
v = {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__,
     "scipy": scipy.__version__, "tldextract": tldextract.__version__}
try:
    import shap; v["shap"] = shap.__version__
except ImportError:
    v["shap"] = None
try:
    import torch; v["torch"] = torch.__version__.split("+")[0]
except ImportError:
    v["torch"] = None
try:
    v["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
except Exception:
    v["cpu"] = platform.processor()
v["n_cpu"] = os.cpu_count()
json.dump(v, open(os.path.join(R, "versions.json"), "w"), indent=1)
here = os.path.dirname(os.path.abspath(__file__))
code = {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(os.path.join(here, "*.py")) + glob.glob(os.path.join(here, "*.sh")))}
json.dump(code, open(os.path.join(R, "code_sha256.json"), "w"), indent=1)
print(v); print(len(code), "code files hashed")
