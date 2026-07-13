"""
Markdown -> PDF for assignment submissions.

Pandoc renders the Markdown to a self-contained HTML document (images inlined as data
URIs, CSS embedded), and headless Chrome prints it to A4. No LaTeX toolchain required,
and the output matches what GitHub shows — the same source of truth, one rendering path.

Why not `pandoc -o file.pdf` directly: that route needs a LaTeX engine, and LaTeX
silently mangles the things these reports depend on — wide Markdown tables overflow the
text block, emoji and box-drawing characters fail to typeset, and `<sub>`/`<picture>`
HTML is dropped. Chrome's print engine is the same one that renders the page you review,
so what you check on screen is what the professor receives.

The generated HTML strips `<picture>` dark-mode variants: the PDF is printed on white,
and a dark-theme image inside it would come out unreadable.

Usage:
    python scripts/build_pdf.py activities/avaliacao-pratica-2-respostas.md
    python scripts/build_pdf.py activities/*-respostas.md
    python scripts/build_pdf.py activities/x.md --output custom.pdf --open
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
)

CSS = """
@page { size: A4; margin: 20mm 18mm 18mm 18mm; }

/* Pandoc's standalone template prints the document metadata title above the content.
   The report already opens with its own H1, so the filename would appear as a stray
   heading on page one. */
#title-block-header { display: none; }

body {
  font-family: Arial, "Helvetica Neue", Helvetica, sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #1a1a1a;
  max-width: none;
  margin: 0;
  hyphens: auto;
  text-align: justify;
}

h1 { font-size: 19pt; line-height: 1.25; margin: 0 0 0.4em; text-align: left; }
h2 {
  font-size: 14pt; margin: 1.6em 0 0.5em; padding-bottom: 0.18em;
  border-bottom: 1px solid #c8ccd0; text-align: left; break-after: avoid;
}
h3 { font-size: 11.5pt; margin: 1.2em 0 0.4em; text-align: left; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 1em 0 0.3em; text-align: left; break-after: avoid; }

/* The metadata block that opens every report. */
blockquote {
  margin: 0 0 1.4em; padding: 0.7em 1em;
  background: #f5f6f7; border-left: 3px solid #8a9098;
  font-size: 9.5pt; line-height: 1.45; text-align: left;
}
blockquote p { margin: 0.2em 0; }

table {
  border-collapse: collapse; width: 100%;
  margin: 0.9em 0; font-size: 8.8pt; line-height: 1.35;
  break-inside: avoid; text-align: left;
}
th, td { border: 1px solid #ccd0d4; padding: 4px 7px; text-align: left; }
th { background: #eef0f2; font-weight: 600; }
tr:nth-child(even) td { background: #fafbfc; }

code {
  font-family: "Consolas", "DejaVu Sans Mono", monospace;
  font-size: 0.86em; background: #f0f2f4;
  padding: 0.08em 0.3em; border-radius: 2px;
}
pre {
  background: #f6f8fa; border: 1px solid #dfe2e5; border-radius: 3px;
  padding: 0.7em 0.9em; overflow-x: auto; font-size: 8.5pt; line-height: 1.4;
  break-inside: avoid; text-align: left;
}
pre code { background: none; padding: 0; font-size: 1em; }

img { max-width: 100%; height: auto; display: block; margin: 0.8em auto; }
p:has(> img) { break-inside: avoid; text-align: center; }

a { color: #0b4f9e; text-decoration: none; }
a:hover { text-decoration: underline; }

hr { border: none; border-top: 1px solid #dcdfe3; margin: 1.6em 0; }

sub { font-size: 0.72em; color: #55595e; }
strong { font-weight: 600; }

/* Never orphan a heading or a table caption at the foot of a page. */
h1, h2, h3, h4 { page-break-after: avoid; }
"""


def find_chrome() -> str:
    for candidate in CHROME_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    found = shutil.which("chrome") or shutil.which("chromium")
    if found:
        return found
    raise SystemExit("no Chrome/Edge found — needed to print the PDF")


def repo_url(branch: str = "main") -> str | None:
    """`https://github.com/owner/repo/{}/main` from the git remote, or None."""
    try:
        remote = subprocess.run(
            ["git", "remote", "get-url", "origin"], cwd=ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", remote)
    return f"https://github.com/{match.group(1)}/{{}}/{branch}" if match else None


def absolutise_links(markdown: str, source: Path) -> str:
    """Relative Markdown links -> absolute GitHub URLs.

    A PDF travels alone. `[protocolo](avaliacao-pratica-2/common.py)` resolves against
    the repository on GitHub and against the *filesystem* in a PDF — so on the author's
    machine it opens the file, and on the reader's machine it opens nothing. Since the
    assignment explicitly requires the report to link to the scripts, a link that only
    works on the machine that built the PDF is a failed requirement, not a cosmetic one.

    Absolute links, anchors, and image embeds are left alone: images are already inlined
    as data URIs by pandoc, and rewriting their `src` would break that.
    """
    base = repo_url()
    if base is None:
        print("  ! no GitHub remote — relative links will stay relative", file=sys.stderr)
        return markdown

    def rewrite(match: re.Match) -> str:
        bang, text, target = match.group(1), match.group(2), match.group(3).strip()
        if bang or re.match(r"^(https?:|mailto:|#|<)", target):
            return match.group(0)

        target, _, anchor = target.partition("#")
        if not target:                       # pure in-document anchor
            return match.group(0)

        resolved = (source.parent / target).resolve()
        try:
            rel = resolved.relative_to(ROOT).as_posix()
        except ValueError:                   # escapes the repository — leave it
            return match.group(0)

        kind = "tree" if resolved.is_dir() or target.endswith("/") else "blob"
        url = f"{base.format(kind)}/{rel}" + (f"#{anchor}" if anchor else "")
        return f"[{text}]({url})"

    return re.sub(r"(!?)\[([^\]]*)\]\(([^)]+)\)", rewrite, markdown)


def preprocess(markdown: str, source: Path) -> str:
    """Make the Markdown printable.

    Three edits, all because a PDF is not a web page:

    * `<picture>` blocks carry a dark-mode `<source>`; printed on white paper the dark
      variant is unreadable, so the block collapses to its light `<img>`.
    * The invisible Obsidian-only fallback images would print as broken boxes.
    * Relative links become absolute GitHub URLs — see `absolutise_links`.
    """
    def unwrap_picture(match: re.Match) -> str:
        block = match.group(0)
        img = re.search(r'<img[^>]*src="([^"]*(?:light|confusion)[^"]*)"[^>]*>', block)
        if not img:
            img = re.search(r"<img[^>]*>", block)
        return f'<p align="center">{img.group(0)}</p>' if img else ""

    markdown = re.sub(r"<picture.*?</picture>", unwrap_picture, markdown,
                      flags=re.DOTALL)
    markdown = re.sub(r'<img class="obsidian-[^"]*"[^>]*>\s*', "", markdown)
    markdown = hard_break_blockquotes(markdown)
    return absolutise_links(markdown, source)


def hard_break_blockquotes(markdown: str) -> str:
    """Force line breaks inside the metadata blockquote that opens every report.

    Markdown folds consecutive lines into one paragraph, so

        > **Disciplina:** ...
        > **Aluno:** Fernando Dantas

    prints as a single run-on line — the discipline, the student and the reproducibility
    note all colliding. Two trailing spaces is Markdown's hard break; it is applied only
    to blockquote lines that are followed by another blockquote line, so prose paragraphs
    keep reflowing normally (turning on `hard_line_breaks` globally would instead break
    every wrapped line in the document).
    """
    lines = markdown.split("\n")
    for i, line in enumerate(lines[:-1]):
        follows = lines[i + 1].lstrip()
        if (line.lstrip().startswith(">") and follows.startswith(">")
                and follows != ">" and not line.rstrip().endswith("  ")):
            lines[i] = line.rstrip() + "  "
    return "\n".join(lines)


def build(source: Path, output: Path | None, chrome: str) -> Path:
    output = output or source.with_suffix(".pdf")
    markdown = preprocess(source.read_text(encoding="utf-8"), source)

    # Windows locks a PDF that is open in a viewer. Chrome then fails to write, exits 0,
    # and the stale file is still sitting there — so a check for `output.exists()` passes
    # and the build reports success while shipping the previous version. Fail up front.
    if output.exists():
        try:
            with output.open("r+b"):
                pass
        except OSError:
            raise SystemExit(
                f"{output.name} is locked — close it in your PDF viewer and re-run. "
                "(Refusing to 'succeed' while leaving the old file in place.)")
    before = output.stat().st_mtime if output.exists() else 0.0

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # written next to the source so relative image paths still resolve
        staged_md = source.parent / f".{source.stem}.print.md"
        staged_md.write_text(markdown, encoding="utf-8")
        css_file = tmp / "style.css"
        css_file.write_text(CSS, encoding="utf-8")
        html = source.parent / f".{source.stem}.print.html"

        try:
            # cwd is the document's own directory: the reports reference images as
            # `../assets/img/...`, and that only resolves from where the Markdown lives.
            subprocess.run(
                ["pandoc", staged_md.name, "-f", "gfm", "-t", "html5",
                 "--standalone", "--embed-resources", "--css", str(css_file),
                 "--resource-path", f".{os.pathsep}{ROOT}",
                 "--metadata", f"title={source.stem}", "-o", html.name],
                check=True, cwd=source.parent)

            subprocess.run(
                [chrome, "--headless", "--disable-gpu", "--no-sandbox",
                 "--no-pdf-header-footer", f"--print-to-pdf={output.resolve()}",
                 html.resolve().as_uri()],
                check=True, capture_output=True)
        finally:
            staged_md.unlink(missing_ok=True)
            html.unlink(missing_ok=True)

    if not output.exists():
        raise SystemExit(f"Chrome produced no output at {output}")
    if output.stat().st_mtime <= before:
        raise SystemExit(f"{output.name} was not rewritten — the file on disk is stale")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", type=Path,
                        help="only valid with a single source")
    args = parser.parse_args()

    if args.output and len(args.sources) > 1:
        raise SystemExit("--output takes a single source")

    chrome = find_chrome()
    for source in args.sources:
        if not source.exists():
            raise SystemExit(f"{source} not found")
        pdf = build(source, args.output, chrome)
        size = pdf.stat().st_size / 1024
        print(f"{source.name}  ->  {pdf}  ({size:.0f} KB)")


if __name__ == "__main__":
    main()
