#!/usr/bin/env python3
"""Generate an ODP presentation from a markdown deck separated by --- blocks."""

from __future__ import annotations

import argparse
from pathlib import Path

from odf import draw, style, text
from odf.opendocument import OpenDocumentPresentation


SLIDE_W = "28cm"
SLIDE_H = "21cm"


def _parse_blocks(md_text: str) -> list[dict[str, list[str] | str]]:
    blocks: list[dict[str, list[str] | str]] = []
    raw_blocks = [b.strip() for b in md_text.split("\n---\n")]
    for raw in raw_blocks:
        if not raw:
            continue
        lines = [ln.rstrip() for ln in raw.splitlines() if ln.strip()]
        if not lines:
            continue

        title = ""
        body: list[str] = []

        for idx, line in enumerate(lines):
            stripped = line.strip()
            if idx == 0 and stripped.startswith("#"):
                title = stripped.lstrip("#").strip()
                continue
            if stripped.lower().startswith("speaker note"):
                continue
            if stripped.startswith("- "):
                body.append("• " + stripped[2:].strip())
                continue
            if stripped[:3].isdigit() and stripped[1:3] == ". ":
                body.append(stripped)
                continue
            body.append(stripped)

        if not title:
            title = "Slide"

        blocks.append({"title": title, "body": body})
    return blocks


def _build_odp(slides: list[dict[str, list[str] | str]], out_path: Path) -> None:
    doc = OpenDocumentPresentation()

    page_layout = style.PageLayout(name="PM1")
    page_layout.addElement(
        style.PageLayoutProperties(
            pagewidth=SLIDE_W,
            pageheight=SLIDE_H,
            printorientation="landscape",
            margintop="0cm",
            marginbottom="0cm",
            marginleft="0cm",
            marginright="0cm",
        )
    )
    doc.automaticstyles.addElement(page_layout)

    master = style.MasterPage(name="Default", pagelayoutname="PM1")
    doc.masterstyles.addElement(master)

    for i, slide in enumerate(slides, start=1):
        page = draw.Page(name=f"slide{i}", masterpagename="Default")

        title_frame = draw.Frame(x="1.0cm", y="0.8cm", width="26cm", height="2.5cm")
        title_box = draw.TextBox()
        title_box.addElement(text.P(text=str(slide["title"])))
        title_frame.addElement(title_box)
        page.addElement(title_frame)

        body_frame = draw.Frame(x="1.2cm", y="3.4cm", width="25.2cm", height="16.8cm")
        body_box = draw.TextBox()
        for line in slide["body"]:
            body_box.addElement(text.P(text=str(line)))
        body_frame.addElement(body_box)
        page.addElement(body_frame)

        doc.presentation.addElement(page)

    doc.save(str(out_path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="PITCH_SLIDES.md", help="Input markdown deck path")
    parser.add_argument("--output", default="PITCH_SLIDES.odp", help="Output ODP path")
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    md_text = in_path.read_text(encoding="utf-8")
    slides = _parse_blocks(md_text)
    if not slides:
        raise SystemExit("No slides found in markdown input")

    _build_odp(slides, out_path)
    print(f"Generated {out_path} ({len(slides)} slides)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
