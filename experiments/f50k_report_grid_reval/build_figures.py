"""Build three compact, source-grounded SVG trend figures for the human report."""

from __future__ import annotations

import json
import math
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[2] / "task1-artifacts/task1-tvm-50k-final"
OUT = ROOT / "report_figures"
REVAL = ROOT / "f50k-report-grid-reval/metrics.json"
FORMAL = ROOT / "F-run/evaluation"
AUDIT = ROOT / "patch-30k-50k"
OWT_REFERENCE = Path(__file__).resolve().parents[2] / "references/owt128_real_reference/owt128-real-reference-v1.json"
XSTEPS = [20, 24, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]


def score(path: Path):
    data = json.loads(path.read_text())
    q = data["sample_quality"]
    return {"ppl": float(data["generative_ppl"]),
            "entropy": float(q["mean_sample_unigram_entropy_nats"]),
            "distinct_2": float(q["distinct_2"]),
            "samples": int(data["num_samples"])}


def dataset():
    new = json.loads(REVAL.read_text())
    online = {row["step"] // 1000: row for row in new if row["weight"] == "online"}
    ema = {row["step"] // 1000: row for row in new if row["weight"] == "eval_ema"}
    rolling = {}
    for step in [20, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50]:
        if step in (20, 30, 40, 50):
            path = FORMAL / f"step_{step * 1000:06d}__eval_ema__legacy_rolling_T1_512__n128/samples.json"
        else:
            path = AUDIT / f"step_{step * 1000:06d}/generation/legacy_local_512/samples.json"
        rolling[step] = score(path)
        assert rolling[step]["samples"] == 128
    return [("512 NFE · eval EMA · n128", rolling, "#2466a8", False),
            ("4 NFE .575 · eval EMA · n128", ema, "#d96a10", False),
            ("4 NFE .575 · online · n128", online, "#bd3b52", False)]


def draw(metric, limits, ticks, label, logscale=False):
    series = dataset()
    owt = json.loads(OWT_REFERENCE.read_text())["sample_quality"]
    owt_values = {"entropy": float(owt["mean_sample_unigram_entropy_nats"]),
                  "distinct_2": float(owt["distinct_2"])}
    width, height = 900, 405
    left, right, top, bottom = 78, 25, 90, 55
    w, h = width - left - right, height - top - bottom
    lo, hi = limits

    def px(x):
        return left + (x - 20) * w / 30

    def py(value):
        if logscale:
            value, floor, ceiling = math.log(value), math.log(lo), math.log(hi)
        else:
            floor, ceiling = lo, hi
        return top + (ceiling - value) * h / (ceiling - floor)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<text x="{left}" y="27" font-size="20" font-family="Arial" font-weight="bold" fill="#1b2733">{escape(label)}</text>']
    for tick in ticks:
        y = py(tick)
        parts += [f'<line x1="{left}" x2="{width-right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e4e8ed"/>',
                  f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#52606d">{tick:g}</text>']
    for step in XSTEPS:
        x = px(step)
        parts += [f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top+h}" y2="{top+h+5}" stroke="#74808b"/>',
                  f'<text x="{x:.1f}" y="{top+h+22}" text-anchor="middle" font-size="11" font-family="Arial" fill="#52606d">{step}</text>']
    parts += [f'<line x1="{left}" x2="{left}" y1="{top}" y2="{top+h}" stroke="#74808b"/>',
              f'<line x1="{left}" x2="{left+w}" y1="{top+h}" y2="{top+h}" stroke="#74808b"/>',
              f'<text x="{left+w/2:.1f}" y="{height-13}" text-anchor="middle" font-size="12" font-family="Arial" fill="#52606d">optimizer steps (k)</text>']
    if metric in owt_values:
        y = py(owt_values[metric])
        parts.append(f'<line x1="{left}" x2="{left+w}" y1="{y:.1f}" y2="{y:.1f}" stroke="#82909c" stroke-width="2" stroke-dasharray="6 5"><title>OWT raw · n1024: {owt_values[metric]:.4f}</title></line>')
    for name, points, color, dashed in series:
        coords = [(step, px(step), py(row[metric])) for step, row in sorted(points.items())]
        path = " ".join(f"{x:.1f},{y:.1f}" for _, x, y in coords)
        dash = ' stroke-dasharray="5 5"' if dashed else ""
        parts.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2.5"{dash}/>')
        for step, x, y in coords:
            fill = "white" if dashed else color
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{fill}" stroke="{color}" stroke-width="1.8"><title>{escape(name)} @ {step}k: {points[step][metric]:.4f}</title></circle>')
    for i, (name, _, color, dashed) in enumerate(series):
        col, row = i % 2, i // 2
        x, y = left + col * 405, 48 + row * 18
        dash = ' stroke-dasharray="5 5"' if dashed else ""
        parts += [f'<line x1="{x}" x2="{x+23}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="2.5"{dash}/>',
                  f'<text x="{x+29}" y="{y+4}" font-size="11" font-family="Arial" fill="#283744">{escape(name)}</text>']
    if metric in owt_values:
        x, y = left + 405, 66
        parts += [f'<line x1="{x}" x2="{x+23}" y1="{y}" y2="{y}" stroke="#82909c" stroke-width="2" stroke-dasharray="6 5"/>',
                  f'<text x="{x+29}" y="{y+4}" font-size="11" font-family="Arial" fill="#283744">OWT raw · n1024 · {owt_values[metric]:.3f}</text>']
    parts.append("</svg>")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{metric}_20k_50k.svg").write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    draw("ppl", (350, 3000), [400, 600, 1000, 1600, 2500], "Generation PPL · 20k–50k", True)
    draw("entropy", (3.7, 4.4), [3.8, 4.0, 4.2, 4.4], "Per-sample entropy · 20k–50k")
    draw("distinct_2", (0.45, 0.84), [0.5, 0.6, 0.7, 0.8], "Distinct-2 · 20k–50k")
