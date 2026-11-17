"""Build the Discover Computing manuscript (.docx) from draft/*.md.

Rules implemented (Discover Computing submission guidelines, read 2026-11-12,
https://link.springer.com/journal/10791/submission-guidelines):
  - Word file, figures and tables placed in the body where they are referenced
  - one consistent font, at least 12 pt; Arial
  - at most three heading levels
  - citations as numbers in square brackets, numbered in order of first citation
  - references in one standard style (NLM / Vancouver, the style the guideline
    recommends), DOIs written as full https://doi.org/ links
  - table captions above the table; figure captions below, starting "Fig. N",
    no punctuation after the number and none at the end of the caption
  - Declarations: funding, competing interests, ethics, consent, data/code
    availability, author contributions

Draft conventions read here:
  # Title                         (00-title.md, first line)
  ## 1 Heading / ### 1.1 Heading  (levels 1-2; level 3 is #### and used rarely)
  [@key] or [@key1; @key2]        citations -> [n]
  **Table N** caption             then a Markdown table (optionally preceded by
                                  a <!-- results: x.csv --> binding comment)
  ![Fig. N caption](../figures/x.png)
  $$ equation text $$ (n)         display equation, plain Unicode, numbered
  - item                          bullet list (real Word numbering)
  **bold**, *italic*

Author, affiliation and e-mail fields are placeholders by design: whoever
submits fills them in.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.parser import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
DRAFT = BASE / "draft"
FIGS = BASE / "figures"
REFS = BASE / "literature" / "refs_verified.json"
OUT = DRAFT / "manuscript.docx"   # Discover Computing version (the NJES build will be manuscript_njes.docx)
FONT = "Arial"
BODY = Pt(12)
SMALL = Pt(10)
CITE_RE = re.compile(r"\[(@[\w-]+(?:\s*;\s*@[\w-]+)*)\]")
KB_META = re.compile(r"^\s*>\s*\*\*(Last updated|Purpose|Status)")


# ---------------------------------------------------------------- references
def load_refs():
    return json.loads(REFS.read_text(encoding="utf-8"))


def nlm(r: dict) -> str:
    au = r.get("authors") or []
    names = []
    for fam, giv in au[:6]:
        ini = "".join(p[0] for p in re.split(r"[\s\-.]+", giv) if p) if giv else ""
        names.append(f"{fam} {ini}".strip())
    who = ", ".join(names) + (", et al." if len(au) > 6 else "")
    title = r["title"].rstrip(".")
    t = r.get("type")
    yr = r.get("year")
    doi = r.get("doi")
    link = f" https://doi.org/{doi}" if doi else (f" {r['url']}" if r.get("url") else "")
    if t == "journal-article":
        vol = f";{r['volume']}" if r.get("volume") else ""
        iss = f"({r['issue']})" if r.get("issue") else ""
        pg = f":{r['page']}" if r.get("page") else ""
        cont = r.get("short_container") or r.get("container")
        return f"{who}. {title}. {cont}. {yr}{vol}{iss}{pg}.{link}"
    if t == "proceedings-article":
        pg = f" p. {r['page']}." if r.get("page") else ""
        return f"{who}. {title}. In: {r['container']}; {yr}.{pg}{link}"
    if t == "standard":
        return f"{who}. {r['container']}: {title}. Geneva: {r['publisher']}; {yr}.{link}"
    if t == "preprint":
        return f"{who}. {title}. arXiv:{r['arxiv']} [Preprint]. {yr}.{link}"
    return f"{who}. {title}. {r.get('container', '')}. {yr}.{link}"


# ---------------------------------------------------------------- docx helpers
def style_doc(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = BODY
    st.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    pf = st.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.5
    for s in ("List Bullet", "List Number"):
        doc.styles[s].font.name = FONT
        doc.styles[s].font.size = BODY
    sec = doc.sections[0]
    sec.left_margin = sec.right_margin = Inches(1)
    sec.top_margin = sec.bottom_margin = Inches(1)
    # page number in the footer
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for tag, text in (("begin", None), (None, "PAGE"), ("end", None)):
        r = p.add_run()
        if tag:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag); r._r.append(el)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text; r._r.append(el)


def add_runs(p, text, size=BODY, bold=False, italic=False):
    """**bold**, *italic*, ==highlight== (yellow; marks fields the submitter must complete)."""
    from docx.enum.text import WD_COLOR_INDEX
    for part in re.split(r"(==[^=]+==|\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        b, i, t = bold, italic, part
        if part.startswith("==") and part.endswith("=="):
            before = len(p.runs)
            add_runs(p, part[2:-2], size=size, bold=bold, italic=italic)
            for r in p.runs[before:]:
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
            continue
        if part.startswith("**") and part.endswith("**"):
            b, t = True, part[2:-2]
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            i, t = True, part[1:-1]
        r = p.add_run(t)
        r.bold, r.italic = b, i
        r.font.size = size
        r.font.name = FONT
    return p


def heading(doc, text, level):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.bold = True
    r.italic = level == 3
    r.font.name = FONT
    r.font.size = Pt(14 if level == 1 else 12)


def borders(table):
    tblPr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
        b.append(e)
    tblPr.append(b)


def add_table(doc, caption, rows):
    cap = doc.add_paragraph()
    cap.paragraph_format.keep_with_next = True
    add_runs(cap, caption, size=SMALL)
    hdr, body = rows[0], rows[1:]
    t = doc.add_table(rows=len(rows), cols=len(hdr))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    borders(t)
    # Column widths from content. A body cell (a number such as "−0.005" or
    # "0.618 ± 0.020") must never wrap, so body cells set a hard floor; header
    # words may wrap at spaces. If the floors do not fit, the font shrinks.
    total_in = 6.5
    floor, want = [], []
    for j in range(len(hdr)):
        body_cells = [r[j] for r in body] or [""]
        words = [w for c in [hdr[j]] + body_cells for w in (c.split() or [""])]
        if j == 0:
            floor.append(max(len(w) for w in words) + 2)
            want.append(min(max(len(c) for c in body_cells) + 2, 30))
        else:
            floor.append(max(max(len(c) for c in body_cells), max(len(w) for w in hdr[j].split() or [""])) + 2)
            want.append(max(floor[-1], min(len(hdr[j]), 16) + 2))   # room for a two-line header
    size = SMALL
    for pt, cpi in ((10, 13.0), (9, 14.5), (8, 16.5)):
        size = Pt(pt)
        if sum(floor) <= total_in * cpi:
            break
    budget = total_in * cpi
    extra = max(budget - sum(floor), 0)
    gap = [max(w - f, 0) for w, f in zip(want, floor)]
    need = [f + (extra * g / sum(gap) if sum(gap) else extra / len(floor)) for f, g in zip(floor, gap)]
    widths = [total_in * n / sum(need) for n in need]
    for j, w in enumerate(widths):
        for i in range(len(rows)):
            t.cell(i, j).width = Inches(w)
    for i, tr in enumerate(t.rows):
        trPr = tr._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)              # never break a row across pages
        if i == 0:
            th = OxmlElement("w:tblHeader"); trPr.append(th)            # repeat header on a new page
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            c = t.cell(i, j)
            c.text = ""
            p = c.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            if len(rows) <= 14 and i < len(rows) - 1:
                p.paragraph_format.keep_with_next = True       # keep a short table on one page
            add_runs(p, cell, size=size, bold=(i == 0))
            if j > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(doc, path, caption):
    im = Image.open(path)
    w_in = min(6.3, im.size[0] / 300)
    h_in = w_in * im.size[1] / im.size[0]
    if h_in > 7.5:
        w_in *= 7.5 / h_in
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(w_in))
    cap = doc.add_paragraph()
    add_runs(cap, caption.rstrip(" ."), size=SMALL)


def split_row(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


# ---------------------------------------------------------------- build
def build():
    refs = load_refs()
    order: list[str] = []

    def cite(m):
        keys = [k.strip()[1:] for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in refs:
                sys.exit(f"citation key not in refs_verified.json: {k}")
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        return "[" + ", ".join(str(n) for n in sorted(nums)) + "]"

    files = sorted(DRAFT.glob("[0-9][0-9]-*.md"))
    if not files:
        sys.exit("no draft sections")
    doc = Document()
    style_doc(doc)
    fig_n = tab_n = 0
    for f in files:
        lines = [ln for ln in f.read_text(encoding="utf-8").splitlines() if not KB_META.match(ln)]
        i = 0
        while i < len(lines):
            ln = lines[i].rstrip()
            s = ln.strip()
            if not s or s.startswith("<!--"):
                i += 1; continue
            if s.startswith("# "):
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                r = p.add_run(s[2:]); r.bold = True; r.font.size = Pt(16); r.font.name = FONT
                i += 1; continue
            if s.startswith("#### "):
                heading(doc, CITE_RE.sub(cite, s[5:]), 3); i += 1; continue
            if s.startswith("### "):
                heading(doc, CITE_RE.sub(cite, s[4:]), 2); i += 1; continue
            if s.startswith("## "):
                heading(doc, CITE_RE.sub(cite, s[3:]), 1); i += 1; continue
            if re.match(r"^\*\*Table \d+\*\*", s):
                cap = s
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("|"):
                    j += 1
                rows = []
                while j < len(lines) and lines[j].strip().startswith("|"):
                    cells = split_row(lines[j])
                    if not set("".join(cells)) <= set("-: "):
                        rows.append([CITE_RE.sub(cite, c) for c in cells])
                    j += 1
                tab_n += 1
                num = int(re.match(r"^\*\*Table (\d+)\*\*", cap).group(1))
                if num != tab_n:
                    sys.exit(f"{f.name}: caption says Table {num} but it is table {tab_n} in document order")
                add_table(doc, CITE_RE.sub(cite, cap), rows)
                i = j; continue
            m = re.match(r"^!\[(.+)\]\((.+)\)$", s)
            if m:
                fig_n += 1
                num = int(re.match(r"Fig\. (\d+)", m.group(1)).group(1))
                if num != fig_n:
                    sys.exit(f"{f.name}: caption says Fig. {num} but it is figure {fig_n} in document order")
                add_figure(doc, (f.parent / m.group(2)).resolve(), CITE_RE.sub(cite, m.group(1)))
                i += 1; continue
            m = re.match(r"^\$\$(.+)\$\$\s*\((\d+)\)$", s)
            if m:
                p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                add_runs(p, m.group(1).strip() + f"     ({m.group(2)})", italic=False)
                i += 1; continue
            if s.startswith("- "):
                while i < len(lines) and lines[i].strip().startswith("- "):
                    p = doc.add_paragraph(style="List Bullet")
                    add_runs(p, CITE_RE.sub(cite, lines[i].strip()[2:]))
                    i += 1
                continue
            # paragraph: join continuation lines
            buf = [s]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r"^(#|\||- |!\[|\*\*Table|\$\$|<!--)", lines[i].strip()):
                buf.append(lines[i].strip()); i += 1
            p = doc.add_paragraph()
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_runs(p, CITE_RE.sub(cite, " ".join(buf)))
    heading(doc, "References", 1)
    for n, k in enumerate(order, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.35)
        p.paragraph_format.first_line_indent = Inches(-0.35)
        p.paragraph_format.line_spacing = 1.15
        add_runs(p, f"{n}. {nlm(refs[k])}", size=Pt(11))
    unused = sorted(set(refs) - set(order))
    doc.save(OUT)
    (BASE / "literature" / "references.md").write_text(
        "# References (citation order, generated)\n\n" + "\n".join(f"{n}. {nlm(refs[k])}" for n, k in enumerate(order, 1))
        + ("\n\nVerified but not cited: " + ", ".join(unused) if unused else "") + "\n", encoding="utf-8")
    print(f"wrote {OUT.name}: {tab_n} tables, {fig_n} figures, {len(order)} references cited; not cited: {unused}")


if __name__ == "__main__":
    build()
