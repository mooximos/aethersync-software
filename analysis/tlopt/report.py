"""REPORT.html (self-contained) and REPORT.md, in plain English, from the files in results/."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

from tlopt import figures
from tlopt.intersection import Intersection

NAMES = {"delay_s": "control delay (s per vehicle)", "time_loss_s": "SUMO time loss (s per vehicle)",
         "waiting_s": "time stopped (s per vehicle)", "p95_wait_s": "95th percentile time stopped (s)",
         "max_wait_s": "longest time stopped (s)", "stops": "stops per vehicle", "co2_g": "CO2 (g per vehicle)",
         "nox_g": "NOx (g per vehicle)", "fuel_g": "fuel (g per vehicle)"}


class Doc:
    """Collects the report once and writes it as Markdown and as HTML."""

    def __init__(self, figures_dir: Path) -> None:
        self.parts: list[tuple[str, object]] = []
        self.figures_dir = figures_dir

    def h(self, text: str, level: int = 2) -> None:
        self.parts.append(("h", (level, text)))

    def p(self, text: str) -> None:
        self.parts.append(("p", text))

    def bullets(self, items: list[str]) -> None:
        self.parts.append(("ul", items))

    def table(self, frame: pd.DataFrame, caption: str = "") -> None:
        self.parts.append(("table", (frame, caption)))

    def image(self, path: Path, alt: str) -> None:
        self.parts.append(("img", (path, alt)))

    def markdown(self) -> str:
        out = []
        for kind, value in self.parts:
            if kind == "h":
                out.append("#" * value[0] + " " + value[1])
            elif kind == "p":
                out.append(value)
            elif kind == "ul":
                out.append("\n".join(f"- {item}" for item in value))
            elif kind == "table":
                frame, caption = value
                out.append((f"*{caption}*\n\n" if caption else "") + _md_table(frame))
            elif kind == "img":
                out.append(f"![{value[1]}](figures/{value[0].name})")
        return "\n\n".join(out) + "\n"

    def html(self, title: str) -> str:
        body = []
        for kind, value in self.parts:
            if kind == "h":
                body.append(f"<h{value[0]}>{html.escape(value[1])}</h{value[0]}>")
            elif kind == "p":
                body.append(f"<p>{html.escape(value)}</p>")
            elif kind == "ul":
                body.append("<ul>" + "".join(f"<li>{html.escape(i)}</li>" for i in value) + "</ul>")
            elif kind == "table":
                frame, caption = value
                body.append((f"<p class='cap'>{html.escape(caption)}</p>" if caption else "")
                            + frame.to_html(index=False, border=0, na_rep="-", float_format=_number))
            elif kind == "img":
                data = base64.b64encode(value[0].read_bytes()).decode()
                body.append(f"<img alt='{html.escape(value[1])}' src='data:image/png;base64,{data}'>")
        style = ("body{font-family:system-ui,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;color:#0b0b0b;"
                 "background:#fcfcfb;line-height:1.5}table{border-collapse:collapse;margin:8px 0 18px;font-size:14px}"
                 "td,th{border-bottom:1px solid #e4e3df;padding:4px 10px;text-align:right}th{color:#52514e}"
                 "img{max-width:100%}.cap{color:#52514e;font-style:italic;margin-bottom:0}h2{margin-top:36px}")
        return (f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
                f"<style>{style}</style></head><body>{''.join(body)}</body></html>")


def _number(value: float) -> str:
    """1 decimal, but enough digits that small values (like 0.04 g of NOx) do not show as 0.0."""
    return f"{value:.3g}" if 0 < abs(value) < 1 else f"{value:.1f}"


def _md_table(frame: pd.DataFrame) -> str:
    cells = frame.astype(object).where(frame.notna(), "-")
    fmt = lambda v: _number(v) if isinstance(v, float) else str(v)  # noqa: E731
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in cells.itertuples(index=False)]
    return "\n".join(lines)


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def build(inter: Intersection, results: Path) -> tuple[Path, Path]:
    measures, plans_dir, sumo = results / "measures", results / "plans", results / "sumo"
    figs = results / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    summary = json.loads((measures / "summary.json").read_text())
    periods = list(summary["periods"])
    doc = Doc(figs)
    doc.h(f"Signal timing study: {summary['intersection']}", 1)
    doc.p("This report measures how traffic uses one signalized intersection today and recommends a better "
          "fixed timing plan. It is a recommendation for the city's traffic engineers to review: the measuring "
          "device never controls the traffic light, and no images or video were stored.")

    from tlopt import report_sections as sections
    sections.measured(doc, inter, summary, measures)
    sections.traffic(doc, periods, measures, figs)
    sections.timing(doc, inter, periods, measures, plans_dir)
    sections.waits(doc, periods, measures, figs)
    sections.model(doc, periods, plans_dir, figs)
    sections.plans(doc, inter, periods, plans_dir, figs)
    sections.change(doc, periods, plans_dir, sumo, figs)
    sections.robust(doc, periods, plans_dir)
    sections.limits(doc, inter, summary, measures, plans_dir, sumo)

    md_path, html_path = results / "REPORT.md", results / "REPORT.html"
    md_path.write_text(doc.markdown(), encoding="utf-8")
    html_path.write_text(doc.html(f"Signal timing study: {summary['intersection']}"), encoding="utf-8")
    return html_path, md_path
