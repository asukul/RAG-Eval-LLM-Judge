"""Print word-count and structural stats for main.tex.

Estimates whether the paper fits the NeurIPS 2026 D&B page limit
(typically 9 pp main + 4 pp checklist + unlimited appendix).
"""
import re
from pathlib import Path

text = Path(__file__).parent.joinpath("main.tex").read_text(encoding="utf-8")

word_pattern = re.compile(r"[a-zA-Z]+")
section_pattern = re.compile(r"^\\section\{", re.MULTILINE)
subsection_pattern = re.compile(r"^\\subsection\{", re.MULTILINE)
figure_pattern = re.compile(r"\\begin\{figure\}")
table_pattern = re.compile(r"\\begin\{table\}")
cite_pattern = re.compile(r"\\cite[pt]?\{")

print("=" * 60)
print("main.tex stats")
print("=" * 60)
print(f"  characters:  {len(text):>6}")
print(f"  word-like:   {len(word_pattern.findall(text)):>6}")
print(f"  sections:    {len(section_pattern.findall(text)):>6}")
print(f"  subsections: {len(subsection_pattern.findall(text)):>6}")
print(f"  figures:     {len(figure_pattern.findall(text)):>6}")
print(f"  tables:      {len(table_pattern.findall(text)):>6}")
print(f"  citations:   {len(cite_pattern.findall(text)):>6}")
print()
print("NeurIPS D&B 2025 page limit:")
print("  9 pp main + 4 pp checklist + unlimited appendix")
print()
words = len(word_pattern.findall(text))
body_words = int(words * 0.65)
print(f"Body-word estimate (post LaTeX-command discount): ~{body_words}")
print(f"Rough page estimate at ~550 words/page: ~{body_words / 550:.1f} pages")
print()
print("Verify exactly: pdflatex main && bibtex main && pdflatex main && pdflatex main")
