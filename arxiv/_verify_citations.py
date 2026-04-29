"""Quick check that every \cite{} key in main.tex has a matching @entry in references.bib."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
tex = (HERE / "main.tex").read_text(encoding="utf-8")
bib = (HERE / "references.bib").read_text(encoding="utf-8")

cited = set()
for m in re.finditer(r"\\(?:cite|citep|citet|citealp)\*?\{([^}]+)\}", tex):
    for k in m.group(1).split(","):
        cited.add(k.strip())

defined = set(re.findall(r"^@\w+\{([^,]+),", bib, re.MULTILINE))

print(f"Cited keys ({len(cited)}):")
for k in sorted(cited):
    flag = " " if k in defined else "X"
    print(f"  [{flag}] {k}")

undefined = sorted(cited - defined)
uncited = sorted(defined - cited)

print()
print(f"UNDEFINED (cited but missing from bib): {undefined or 'none'}")
print(f"UNCITED  (in bib but never cited)     : {uncited or 'none'}")

sys.exit(0 if not undefined else 1)
