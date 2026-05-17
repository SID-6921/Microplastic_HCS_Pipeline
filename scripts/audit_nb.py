import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = json.load(open(os.path.join(ROOT, "notebooks", "MS2_Manuscript.ipynb"), encoding="utf-8"))
code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
md   = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
print(f"Code cells: {code}, Markdown: {md}, Total: {len(nb['cells'])}")

# Cross-check: every load() and show_fig() filename must exist on disk
import re
TABLES  = os.path.join(ROOT, "results", "tables")
FIGURES = os.path.join(ROOT, "results", "figures")
src = json.dumps(nb)
errors = []
for m in re.findall(r'load\(\\"([^"]+)\\"\)', src):
    if not os.path.exists(os.path.join(TABLES, m)):
        errors.append(f"MISSING TABLE: {m}")
for m in re.findall(r'show_fig\(\\"([^"]+)\\"\)', src):
    if not os.path.exists(os.path.join(FIGURES, m)):
        errors.append(f"MISSING FIGURE: {m}")
if errors:
    print("ERRORS:")
    for e in errors: print(" ", e)
else:
    print("All load() and show_fig() filenames found on disk. OK")
