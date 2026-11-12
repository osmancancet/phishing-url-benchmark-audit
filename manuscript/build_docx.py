"""Build main.docx from the LaTeX source with pandoc + citeproc, then tidy styles (black text, TNR 12, double spacing)."""
import os, re, subprocess
from docx import Document
from docx.shared import Pt, RGBColor
from docx.oxml.ns import qn

H = os.path.dirname(os.path.abspath(__file__))
os.chdir(H)

def expand(path):
    s = open(path).read()
    s = re.sub(r"\\input\{([^}]+)\}", lambda m: expand(m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex"), s)
    return s

src = expand("main.tex")
# pre-pandoc rewrites
src = re.sub(r"\\cmidrule\([^)]*\)\{[^}]*\}", "", src)
src = re.sub(r"\\paragraph\{([^}]*)\}", r"\\textbf{\1.} ", src)
src = re.sub(r"\\resizebox\{[^}]*\}\{!\}\{(\\begin\{tabular\})", r"\1", src)
src = src.replace(r"\end{tabular}}", r"\end{tabular}")
src = re.sub(r"\\begin\{adjustbox\}\{[^}]*\}\n?", "", src).replace("\\end{adjustbox}", "")
src = src.replace(r">{\raggedright\arraybackslash}", "").replace(r"\scriptsize", "")
src = re.sub(r"\\multicolumn\{(\d+)\}\{@\{\}l\}", r"\\multicolumn{\1}{l}", src)
src = src.replace(r"\linenumbers", "").replace(".pdf}", ".png}")
open("_flat.tex", "w").write(src)
# reference doc
if not os.path.exists("reference.docx"):
    subprocess.run(["pandoc", "-o", "reference.docx", "--print-default-data-file", "reference.docx"], check=True)
subprocess.run(["pandoc", "_flat.tex", "-f", "latex", "-t", "docx", "--citeproc", "--bibliography", "refs.bib", "--csl", "elsevier-harvard.csl",
                "--resource-path", ".", "-o", "main.docx"], check=True)
d = Document("main.docx")
for st in d.styles:
    try:
        st.font.name = "Times New Roman"; st.font.color.rgb = RGBColor(0, 0, 0)
        st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    except Exception:
        pass
for p in d.paragraphs:
    p.paragraph_format.line_spacing = 2.0
    for r in p.runs:
        r.font.color.rgb = RGBColor(0, 0, 0)
for t in d.tables:
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.line_spacing = 1.0
                for r in p.runs:
                    r.font.size = Pt(9); r.font.color.rgb = RGBColor(0, 0, 0)
d.core_properties.author = "Osman Can Çetlenbik"
d.save("main.docx")
os.remove("_flat.tex")
print("main.docx written")
