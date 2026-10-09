"""Document viewer: renders a markdown doc from docs/ as HTML, highlighting a section/sub-section.

Documents are resolved ONLY through a title -> path index (first `# Title` line of each docs/**/*.md),
never from user-supplied paths, so path traversal is impossible.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
router = APIRouter()

_index: dict[str, Path] | None = None


def document_url(document: str, section: str | None = None, sub_section: str | None = None,
                 page: int | None = None) -> str:
    params = {"document": document}
    if section:
        params["section"] = section
    if sub_section:
        params["sub_section"] = sub_section
    if page:
        params["page"] = str(page)
    return "/documents/view?" + urlencode(params)  # urlencode quotes with quote_plus (%20 -> +), valid


def _title_of(path: Path) -> str | None:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("# "):
                return line[2:].strip()
    return None


def get_index(refresh: bool = False) -> dict[str, Path]:
    global _index
    if _index is None or refresh:
        idx: dict[str, Path] = {}
        for p in sorted(DOCS_DIR.rglob("*.md")):
            title = _title_of(p)
            if title:
                idx.setdefault(title, p)
        _index = idx
    return _index


_PAGE_RE = re.compile(r"^\[\[PAGE (\d+)\]\]\s*$", re.M)
_HEAD_RE = re.compile(r"<h([123])>(.*?)</h\1>", re.S)

CSS = """
body{font-family:system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#222}
h1{border-bottom:2px solid #ddd;padding-bottom:.3rem}
.page-marker{margin:1.5rem 0 .5rem;padding:.15rem .6rem;background:#eef1f5;color:#556;font-size:.8rem;
 text-transform:uppercase;letter-spacing:.06em;border-left:4px solid #9aa5b5}
.page-marker.target{background:#ffe9a8;border-left-color:#e0a800}
.blk{scroll-margin-top:1rem}
.blk.highlight{background:#fff3b0;border-left:5px solid #f0b400;padding:.2rem .8rem;margin:.5rem 0;border-radius:4px}
table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.3rem .6rem}
"""
JS = "var t=document.querySelector('.highlight')||document.querySelector('.page-marker.target');" \
     "if(t){t.scrollIntoView({block:'center'});}"


def _norm(s: str | None) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip().casefold()


def render_document(path: Path, section: str | None, sub_section: str | None, page: int | None) -> str:
    import markdown

    text = path.read_text(encoding="utf-8")
    text = _PAGE_RE.sub(lambda m: f'\n<div class="page-marker" id="page-{m[1]}" data-page="{m[1]}">'
                                  f"Page {m[1]}</div>\n", text)
    body = markdown.markdown(text, extensions=["extra", "sane_lists"])
    want_sec, want_sub = _norm(section), _norm(sub_section)

    # Wrap each heading + its following content in a block so the whole target can be highlighted.
    parts = _HEAD_RE.split(body)  # [pre, level, title, content, level, title, content, ...]
    out = [parts[0]]
    cur_section = ""
    n = 0
    matched = False
    for i in range(1, len(parts), 3):
        level, title, content = parts[i], parts[i + 1], parts[i + 2]
        if level == "1":
            out.append(f"<h1>{title}</h1>{content}")
            continue
        n += 1
        t = _norm(title)
        hl = False
        if level == "2":
            cur_section = t
            hl = bool(want_sec) and t == want_sec and not want_sub
        elif want_sub and t == want_sub and (not want_sec or cur_section == want_sec):
            hl = True
        matched = matched or hl
        cls = "blk highlight" if hl else "blk"
        out.append(f'<div class="{cls}" id="sec-{n}"><h{level}>{title}</h{level}>{content}</div>')
    doc = "".join(out)
    if not matched and page:  # fall back to the page marker
        doc = doc.replace(f'class="page-marker" id="page-{page}"', f'class="page-marker target" id="page-{page}"')
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(path.stem)}</title>"
            f"<style>{CSS}</style></head><body>{doc}<script>{JS}</script></body></html>")


@router.get("/documents/view", response_class=HTMLResponse)
async def view_document(document: str = Query(...), section: str | None = None,
                        sub_section: str | None = None, page: int | None = None) -> HTMLResponse:
    path = get_index().get(document.strip())
    if path is None:
        raise HTTPException(status_code=404, detail=f"Unknown document: {document!r}")
    return HTMLResponse(render_document(path, section, sub_section, page))
