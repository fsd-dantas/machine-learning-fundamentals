"""
Regenerate every mind-map view from the single source of truth `mind-map.json`.

Outputs (only the regions between the GENERATED markers are touched):
  - mind-map.md ............ Mermaid `mindmap` (View 1) and `flowchart` (View 2)
  - pages/mapa-mental.html . the `const modules/nodes/edges` data block used by the SVG

Usage:
    python scripts/build_mindmap.py           # rewrite the files
    python scripts/build_mindmap.py --check    # fail if anything is out of date (CI)

Stdlib only. Edit `mind-map.json`, never the generated regions.
"""
from __future__ import annotations
import json, re, sys, pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "mind-map.json"
MD = REPO / "mind-map.md"
HTML = REPO / "pages" / "mapa-mental.html"

CIRCLED = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤"}


def load():
    return json.loads(SRC.read_text(encoding="utf-8"))


def splice(text: str, tag: str, body: str) -> str:
    """Replace the text between `BEGIN GENERATED: <tag>` and `END GENERATED: <tag>`."""
    pat = re.compile(
        r"(BEGIN GENERATED: %s[^\n]*\n).*?(\n[^\n]*END GENERATED: %s)" % (re.escape(tag), re.escape(tag)),
        re.DOTALL,
    )
    if not pat.search(text):
        raise SystemExit(f"marker for '{tag}' not found in target file")
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def mm_label(raw: str) -> str:
    """Mermaid-mindmap-safe label: no newlines or shape-delimiter brackets."""
    s = raw.replace("\n", " ")
    for ch in "()[]{}":
        s = s.replace(ch, " ")
    return re.sub(r"\s+", " ", s).strip()


# --------------------------------------------------------------------------- #
# View 1 — Mermaid mindmap (derived from nodes + tree edges)
# --------------------------------------------------------------------------- #
def build_mindmap(data) -> str:
    by_id = {n["id"]: n for n in data["nodes"]}
    children: dict[str, list[str]] = {}
    for s, t in data["edges"]:
        children.setdefault(s, []).append(t)

    root = next(n for n in data["nodes"] if n.get("type") == "root")
    lines = ["```mermaid", "mindmap", f"  root(({mm_label(root['label'])}))"]
    for mod_id in children.get(root["id"], []):
        lines.append(f"    {mm_label(by_id[mod_id]['label'])}")
        for concept_id in children.get(mod_id, []):
            lines.append(f"      {mm_label(by_id[concept_id]['label'])}")
    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# View 2 — Mermaid flowchart (modules + prerequisite edges)
# --------------------------------------------------------------------------- #
def build_flowchart(data) -> str:
    mods = data["modules"]
    order = sorted(mods, key=lambda k: mods[k]["order"])
    lines = ["```mermaid", "flowchart LR"]
    for k in order:
        m = mods[k]
        lines.append(f'    {k}["{CIRCLED[m["order"]]} {m["label"]}<br/>{m["blurb"]}"]')
    lines.append("")
    for e in data["prerequisites"]:
        arrow = "-.->" if e.get("dotted") else "-->"
        label = f'|{e["label"]}|' if e.get("label") else ""
        lines.append(f'    {e["from"]} {arrow}{label} {e["to"]}')
    lines.append("")
    for k in order:
        num = f"{mods[k]['order']:02d}"
        fname = {1: "01-foundations", 2: "02-protocols", 3: "03-shallow",
                 4: "04-descriptive", 5: "05-deep"}[mods[k]["order"]]
        lines.append(f'    click {k} "modules/{fname}.md" "{mods[k]["label"]}"')
    lines.append("")
    for k in order:
        lines.append(f'    style {k} fill:{mods[k]["color"]},color:#fff,stroke:#333')
    lines.append("```")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# HTML data block (modules + nodes + edges consumed by the SVG script)
# --------------------------------------------------------------------------- #
def build_html_data(data) -> str:
    modules_min = {k: {"label": v["label"], "color": v["color"]} for k, v in data["modules"].items()}
    def js(obj):
        return json.dumps(obj, ensure_ascii=False, indent=2)
    parts = [
        "  const modules = " + js(modules_min) + ";",
        "",
        "  const nodes = " + js(data["nodes"]) + ";",
        "",
        "  const edges = " + js(data["edges"]) + ";",
    ]
    return "\n".join(parts)


def main():
    check = "--check" in sys.argv
    data = load()

    targets = {
        MD: [("mindmap", build_mindmap(data)), ("flowchart", build_flowchart(data))],
        HTML: [("mindmap-data", build_html_data(data))],
    }

    stale = []
    for path, blocks in targets.items():
        text = path.read_text(encoding="utf-8")
        new = text
        for tag, body in blocks:
            new = splice(new, tag, body)
        if new != text:
            stale.append(path.name)
            if not check:
                path.write_text(new, encoding="utf-8", newline="\n")  # keep LF; avoid CRLF diff noise on Windows

    if check:
        if stale:
            print("OUT OF DATE (run scripts/build_mindmap.py):", ", ".join(stale))
            sys.exit(1)
        print("mind map views are up to date")
    else:
        print("regenerated:", ", ".join(p.name for p in targets) if stale else "no changes")


if __name__ == "__main__":
    main()
