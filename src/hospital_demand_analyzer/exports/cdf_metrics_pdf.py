"""Generate PDF documentation for CDF fit metrics from Markdown source."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from hospital_demand_analyzer.utils.paths import project_root

# A4 landscape in inches (better for tables)
PAGE_W = 11.69
PAGE_H = 8.27
MARGIN_X = 0.06
MARGIN_TOP = 0.94
MARGIN_BOTTOM = 0.06
LINE_H = 0.022
MAX_CHARS = 118


def _wrap(text: str, width: int = MAX_CHARS) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    current: list[str] = []
    length = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if length + extra > width and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += extra
    if current:
        lines.append(" ".join(current))
    return lines


def _parse_markdown_blocks(content: str) -> list[tuple[str, str]]:
    """Return list of (block_type, text) where type is title|h2|h3|para|table|hr|bullet."""
    blocks: list[tuple[str, str]] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line or line.strip() == "---":
            i += 1
            continue
        if line.startswith("# "):
            blocks.append(("title", line[2:].strip()))
            i += 1
            continue
        if line.startswith("## "):
            blocks.append(("h2", line[3:].strip()))
            i += 1
            continue
        if line.startswith("### "):
            blocks.append(("h3", line[4:].strip()))
            i += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = lines[i].strip()
                if not re.match(r"^\|[\s\-:|]+\|$", row):
                    table_lines.append(row)
                i += 1
            if table_lines:
                blocks.append(("table", "\n".join(table_lines)))
            continue
        if line.startswith("- "):
            bullets = []
            while i < len(lines) and lines[i].startswith("- "):
                bullets.append(lines[i][2:].strip())
                i += 1
            blocks.append(("bullet", "\n".join(bullets)))
            continue
        if line.startswith("```"):
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            blocks.append(("code", "\n".join(code_lines)))
            continue
        para_lines = [line]
        i += 1
        while i < len(lines):
            nxt = lines[i].rstrip()
            if (
                not nxt
                or nxt.startswith("#")
                or nxt.startswith("|")
                or nxt.startswith("- ")
                or nxt.startswith("```")
                or nxt.strip() == "---"
            ):
                break
            para_lines.append(nxt)
            i += 1
        blocks.append(("para", " ".join(para_lines)))
    return blocks


def _format_table(table_text: str) -> list[str]:
    rows = []
    for line in table_text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append("  |  ".join(cells))
    return rows


class _PageWriter:
    def __init__(self, pdf: PdfPages) -> None:
        self._pdf = pdf
        self._fig = None
        self._ax = None
        self._y = MARGIN_TOP

    def _new_page(self, footer: str = "") -> None:
        if self._fig is not None:
            if footer:
                self._ax.text(
                    0.5, 0.02, footer, ha="center", va="bottom",
                    fontsize=7, color="#666666", transform=self._ax.transAxes,
                )
            self._pdf.savefig(self._fig)
            plt.close(self._fig)
        self._fig, self._ax = plt.subplots(figsize=(PAGE_W, PAGE_H))
        self._ax.set_xlim(0, 1)
        self._ax.set_ylim(0, 1)
        self._ax.axis("off")
        self._y = MARGIN_TOP

    def _ensure_space(self, lines_needed: int, footer: str) -> None:
        if self._fig is None:
            self._new_page(footer)
            return
        if self._y - lines_needed * LINE_H < MARGIN_BOTTOM:
            self._new_page(footer)

    def write_line(
        self,
        text: str,
        *,
        fontsize: float = 9,
        weight: str = "normal",
        color: str = "#222222",
        indent: float = 0.0,
        footer: str = "",
    ) -> None:
        self._ensure_space(1, footer)
        assert self._ax is not None
        self._ax.text(
            MARGIN_X + indent, self._y, text,
            ha="left", va="top", fontsize=fontsize,
            fontweight=weight, color=color, transform=self._ax.transAxes,
            family="sans-serif",
        )
        self._y -= LINE_H * (0.85 if fontsize <= 9 else 1.0)

    def write_wrapped(
        self,
        text: str,
        *,
        fontsize: float = 9,
        weight: str = "normal",
        indent: float = 0.0,
        footer: str = "",
    ) -> None:
        for line in _wrap(text):
            self.write_line(line, fontsize=fontsize, weight=weight, indent=indent, footer=footer)

    def finish(self, footer: str = "") -> None:
        if self._fig is not None:
            if footer:
                self._ax.text(
                    0.5, 0.02, footer, ha="center", va="bottom",
                    fontsize=7, color="#666666", transform=self._ax.transAxes,
                )
            self._pdf.savefig(self._fig)
            plt.close(self._fig)
            self._fig = None


def generate_cdf_metrics_pdf(
    output_path: Path | None = None,
    markdown_path: Path | None = None,
) -> Path:
    """Render METRICAS_CDF markdown to PDF. Returns path to written file."""
    root = project_root()
    md_path = markdown_path or root / "docs" / "METRICAS_CDF.md"
    out_path = output_path or root / "METRICAS_CDF.pdf"

    content = md_path.read_text(encoding="utf-8")
    blocks = _parse_markdown_blocks(content)
    generated = datetime.now().strftime("%d/%m/%Y %H:%M")
    footer = f"Hospital Demand Distribution Analyzer — gerado em {generated}"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(out_path) as pdf:
        writer = _PageWriter(pdf)
        toc: list[str] = []

        for block_type, text in blocks:
            if block_type == "h2" and not text.startswith("15."):
                toc.append(text)

        # Cover + TOC
        writer.write_line("Métricas de Ajuste da FDC (CDF)", fontsize=18, weight="bold")
        writer.write_line(
            "Hospital Demand Distribution Analyzer",
            fontsize=11, color="#444444",
        )
        writer._y -= LINE_H
        writer.write_line("Sumário", fontsize=13, weight="bold")
        for item in toc:
            writer.write_wrapped(f"• {item}", fontsize=9, indent=0.02)
        writer._y -= LINE_H
        writer.write_wrapped(
            "Este documento descreve cada métrica usada para comparar a FDC empírica "
            "com CDFs teóricas ajustadas à demanda hospitalar horária (demanda ≥ 0).",
            fontsize=9,
        )
        writer.finish(footer)

        for block_type, text in blocks:
            if block_type == "title":
                continue
            if block_type == "h2":
                writer._y -= LINE_H * 0.5
                writer.write_line(text, fontsize=13, weight="bold", color="#1a5276", footer=footer)
                writer._y -= LINE_H * 0.3
            elif block_type == "h3":
                writer.write_line(text, fontsize=10, weight="bold", footer=footer)
            elif block_type == "para":
                writer.write_wrapped(text, fontsize=9, footer=footer)
                writer._y -= LINE_H * 0.2
            elif block_type == "bullet":
                for bullet in text.split("\n"):
                    writer.write_wrapped(f"• {bullet}", fontsize=9, indent=0.02, footer=footer)
                writer._y -= LINE_H * 0.2
            elif block_type == "code":
                for cline in text.split("\n"):
                    writer.write_line(cline, fontsize=8, color="#333333", indent=0.04, footer=footer)
                writer._y -= LINE_H * 0.2
            elif block_type == "table":
                writer._y -= LINE_H * 0.2
                for tline in _format_table(text):
                    writer.write_line(tline, fontsize=7.5, footer=footer)
                writer._y -= LINE_H * 0.3

        writer.finish(footer)

    return out_path
