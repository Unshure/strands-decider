"""Minimal, dependency-free SVG charts for the strands-decider report.

Conventions (following the dataviz reference palette):
  * one y-axis per chart; separate measures get separate panels, never a second axis;
  * the default lineage (runs adopted as the default recipe) is series 1, a 2px line with
    8px dots; experiments that were not adopted are series 2, dots only, since they are
    branches, not steps along a line;
  * reference systems are hairline rules labelled at the right edge in muted ink;
  * text never wears a series colour; every mark carries a <title> for hover;
  * light and dark themes come from CSS custom properties and prefers-color-scheme.
Palette validated with the dataviz validator (both modes, all pairs): series 1/2 are
blue #2a78d6 / orange #eb6834 (dark #3987e5 / #d95926).
"""

from __future__ import annotations

import itertools
import math
from html import escape

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
STYLE = f"""<style>
.viz {{ --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --grid:#e1e0d9;
  --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a;
  --h0:#cde2fb; --h1:#9ec5f4; --h2:#6da7ec; --h3:#3987e5; --h4:#256abf; --h5:#184f95; --h6:#0d366b; }}
@media (prefers-color-scheme: dark) {{
  .viz {{ --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --grid:#2c2c2a; --axis:#383835;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70;
    --h0:#0d366b; --h1:#104281; --h2:#184f95; --h3:#1c5cab; --h4:#256abf; --h5:#3987e5; --h6:#6da7ec; }} }}
.viz text {{ font-family:{FONT}; fill:var(--ink2); font-size:11px; }}
.viz .title {{ fill:var(--ink); font-size:15px; font-weight:600; }}
.viz .subtitle {{ fill:var(--ink2); font-size:12px; }}
.viz .panel {{ fill:var(--ink); font-size:12px; font-weight:600; }}
.viz .muted {{ fill:var(--muted); }}
.viz .value {{ fill:var(--ink); font-weight:600; }}
.viz .tick {{ font-variant-numeric:tabular-nums; fill:var(--muted); }}
.viz .grid {{ stroke:var(--grid); stroke-width:1; }}
.viz .axis {{ stroke:var(--axis); stroke-width:1; }}
.viz .ref {{ stroke:var(--muted); stroke-width:1; }}
.viz .line1 {{ stroke:var(--s1); stroke-width:2; fill:none; stroke-linejoin:round; stroke-linecap:round; }}
.viz .line0, .viz .line2, .viz .line3 {{ stroke-width:1.5; fill:none; stroke-linejoin:round; stroke-linecap:round; }}
.viz .line0 {{ stroke:var(--ink2); }}
.viz .line2 {{ stroke:var(--s2); }}
.viz .line3 {{ stroke:var(--s3); }}
.viz .strip {{ fill:transparent; }}
.viz .strip:hover {{ fill:var(--grid); fill-opacity:0.5; }}
.viz .dot1 {{ fill:var(--s1); stroke:var(--surface); stroke-width:2; }}
.viz .dot2 {{ fill:var(--s2); stroke:var(--surface); stroke-width:2; }}
.viz .dot3 {{ fill:var(--s3); stroke:var(--surface); stroke-width:2; }}
.viz .dot1s {{ fill:var(--s1); fill-opacity:0.55; }}
.viz .arrow {{ fill:var(--s1); }}
.viz .hit {{ fill:transparent; }}
</style>"""


def _svg(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="viz" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n{STYLE}\n'
        f'<rect width="{width}" height="{height}" fill="var(--surface)"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def text_width(s, size=11):
    return len(s) * size * 0.56


def nice_ticks(lo, hi, n=5):
    span = hi - lo
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start, end = math.floor(lo / step + 1e-9) * step, math.ceil(hi / step - 1e-9) * step
    ticks, t = [], start
    while t <= end + step * 1e-6:  # the range always covers the data: ends on a tick past hi
        ticks.append(round(t, 10))
        t += step
    return ticks, step


def _fmt(v, step):
    """Enough decimals to show the step exactly: a 0.025 step needs three, not two."""
    decimals = max(0, -math.floor(math.log10(step))) if step < 1 else 0
    while decimals < 6 and abs(round(step, decimals) - step) > 1e-9:
        decimals += 1
    return f"{v:.{decimals}f}"


def _title(s):
    return f"<title>{escape(s)}</title>"


def _plot(
    body,
    box,
    xs_labels,
    points,
    refs,
    ylabel=None,
    lower_is_better=False,
    label_last=True,
    y_range=None,
    cap=None,
):
    """Draw one dot/lineage panel into `body` inside box=(x, y, w, h).

    points: list of dicts {x_label, value, kind ('default'|'experiment'), tip}
    refs:   list of (label, value) horizontal reference rules
    """
    x0, y0, w, h = box
    vals = [p["value"] for p in points] + [v for _, v in refs]
    if y_range is not None:  # an explicit range is used as given
        ticks, step = nice_ticks(*y_range, 4)
    else:
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.12 or 0.01
        ticks, step = nice_ticks(lo - pad, min(hi + pad, cap) if cap is not None else hi + pad, 4)
        if cap is not None:  # e.g. accuracy: no tick above 1.0
            ticks = [t for t in ticks if t <= cap + 1e-9]
            if ticks[-1] < hi:
                ticks.append(cap)
    ylo, yhi = ticks[0], ticks[-1]
    ref_space = max((text_width(lbl) for lbl, _ in refs), default=0) + (10 if refs else 0)
    pw = w - ref_space

    def ypos(v):
        return y0 + h - (v - ylo) / (yhi - ylo) * h

    band = pw / len(xs_labels)

    def xpos(lbl):
        return x0 + band * (xs_labels.index(lbl) + 0.5)

    for t in ticks:  # recessive hairline grid + tick labels
        y = ypos(t)
        body.append(f'<line class="grid" x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}"/>')
        body.append(
            f'<text class="tick" x="{x0 - 6}" y="{y + 3.5:.1f}" text-anchor="end">{_fmt(t, step)}</text>'
        )
    body.append(f'<line class="axis" x1="{x0}" x2="{x0 + pw}" y1="{y0 + h}" y2="{y0 + h}"/>')
    for lbl in xs_labels:
        body.append(
            f'<text x="{xpos(lbl):.1f}" y="{y0 + h + 16}" text-anchor="middle">{escape(lbl)}</text>'
        )
    note = ", ".join(x for x in (ylabel, "lower is better" if lower_is_better else None) if x)
    if note:  # above the plot, starting at the tick-label column so it is never clipped
        body.append(f'<text class="muted" x="{x0 - 40}" y="{y0 - 10}">{escape(note)}</text>')

    # reference systems, labelled at the right edge; labels closer than 12px are spread
    # apart so they never overprint (the rules themselves stay at their true values)
    placed = sorted(((ypos(v), lbl, v) for lbl, v in refs), key=lambda t: t[0])
    label_y = []
    for y, _, _ in placed:
        label_y.append(max(y, label_y[-1] + 12) if label_y else y)
    for (y, lbl, v), ly in zip(placed, label_y, strict=True):
        body.append(
            f"<g>{_title(f'{lbl}: {v:.3f}')}"
            f'<line class="ref" x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}"/>'
            f'<text class="muted" x="{x0 + pw + 6}" y="{ly + 3.5:.1f}">{escape(lbl)}</text></g>'
        )

    lineage = [p for p in points if p["kind"] == "default"]
    if len(lineage) > 1:
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{xpos(p['x_label']):.1f},{ypos(p['value']):.1f}"
            for i, p in enumerate(lineage)
        )
        body.append(f'<path class="line1" d="{d}"/>')
    for p in points:
        cx, cy = xpos(p["x_label"]), ypos(p["value"])
        cls = "dot1" if p["kind"] == "default" else "dot2"
        body.append(
            f'<g>{_title(p["tip"])}<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="11"/>'
            f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="5"/></g>'
        )
    if label_last and lineage:  # one direct label: where the lineage ends
        p = lineage[-1]
        label = p.get("label") or f"{p['value']:.3f}"
        body.append(
            f'<text class="value" x="{xpos(p["x_label"]):.1f}" y="{ypos(p["value"]) - 10:.1f}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )


def _legend(body, x, y, with_experiments=True):
    body.append(
        f'<line class="line1" x1="{x}" x2="{x + 18}" y1="{y}" y2="{y}"/>'
        f'<circle class="dot1" cx="{x + 9}" cy="{y}" r="4"/>'
        f'<text x="{x + 24}" y="{y + 4}">adopted as the default recipe</text>'
    )
    if with_experiments:
        x2 = x + 24 + text_width("adopted as the default recipe") + 24
        body.append(
            f'<circle class="dot2" cx="{x2 + 9}" cy="{y}" r="4"/>'
            f'<text x="{x2 + 22}" y="{y + 4}">experiment, not adopted</text>'
        )


def dot_chart(
    title,
    subtitle,
    xs_labels,
    points,
    refs=(),
    ylabel=None,
    lower_is_better=False,
    width=760,
    height=380,
    y_range=None,
    cap=None,
):
    body = [
        f'<text class="title" x="24" y="30">{escape(title)}</text>',
        f'<text class="subtitle" x="24" y="50">{escape(subtitle)}</text>',
    ]
    _legend(body, 24, 74, with_experiments=any(p["kind"] == "experiment" for p in points))
    _plot(
        body,
        (72, 118, width - 72 - 24, height - 118 - 44),
        xs_labels,
        points,
        list(refs),
        ylabel=ylabel,
        lower_is_better=lower_is_better,
        y_range=y_range,
        cap=cap,
    )
    return _svg(width, height, body)


def small_multiples(title, subtitle, panels, xs_labels, width=760, panel_height=170, cols=1):
    """panels: list of dicts {title, points, refs, lower_is_better, ylabel}."""
    rows = math.ceil(len(panels) / cols)
    top = 100
    ph = panel_height
    height = top + rows * (ph + 70) + 10
    body = [
        f'<text class="title" x="24" y="30">{escape(title)}</text>',
        f'<text class="subtitle" x="24" y="50">{escape(subtitle)}</text>',
    ]
    _legend(
        body,
        24,
        74,
        with_experiments=any(p["kind"] == "experiment" for pan in panels for p in pan["points"]),
    )
    colw = (width - 24) / cols
    for i, pan in enumerate(panels):
        r, c = divmod(i, cols)
        x = 24 + c * colw
        y = top + r * (ph + 70) + 30
        body.append(f'<text class="panel" x="{x}" y="{y - 8}">{escape(pan["title"])}</text>')
        _plot(
            body,
            (x + 48, y + 22, colw - 48 - 24, ph - 22),
            xs_labels,
            pan["points"],
            list(pan.get("refs", ())),
            ylabel=pan.get("ylabel"),
            lower_is_better=pan.get("lower_is_better", False),
            y_range=pan.get("y_range"),
            cap=pan.get("cap"),
        )
    return _svg(width, height, body)


def heatmap(title, subtitle, row_labels, col_labels, cells, gap_before=None, width=None):
    """cells[(row, col)] = (value in [0, 1], tip). gap_before: col label to set apart."""
    cw, ch = 44, 26
    left = 24 + max(text_width(r, 11) for r in row_labels) + 12
    top = 96
    extra = 16 if gap_before else 0
    width = width or int(left + len(col_labels) * cw + extra + 24)
    height = top + len(row_labels) * ch + 90
    body = [
        f'<text class="title" x="24" y="30">{escape(title)}</text>',
        f'<text class="subtitle" x="24" y="50">{escape(subtitle)}</text>',
    ]

    def cx(j):
        return left + j * cw + (extra if gap_before and j >= col_labels.index(gap_before) else 0)

    for j, c in enumerate(col_labels):
        body.append(
            f'<text x="{cx(j) + cw / 2:.1f}" y="{top - 8}" text-anchor="middle">{escape(c)}</text>'
        )
    for i, r in enumerate(row_labels):
        y = top + i * ch
        body.append(
            f'<text x="{left - 10}" y="{y + ch / 2 + 4:.1f}" text-anchor="end">{escape(r)}</text>'
        )
        for j, c in enumerate(col_labels):
            if (r, c) not in cells:
                continue
            v, tip = cells[(r, c)]
            b = min(6, int(v * 7))  # seven sequential bins, 0-1
            body.append(
                f'<rect x="{cx(j) + 1:.1f}" y="{y + 1}" width="{cw - 2}" height="{ch - 2}" rx="3" '
                f'fill="var(--h{b})">{_title(tip)}</rect>'
            )
    # ramp legend
    ly = top + len(row_labels) * ch + 30
    body.append(f'<text x="{left}" y="{ly - 8}">accuracy</text>')
    for b in range(7):
        body.append(
            f'<rect x="{left + b * 34}" y="{ly}" width="32" height="12" rx="2" fill="var(--h{b})"/>'
        )
        body.append(f'<text class="tick" x="{left + b * 34}" y="{ly + 26}">{b / 7:.2f}</text>')
    body.append(f'<text class="tick" x="{left + 7 * 34}" y="{ly + 26}">1.00</text>')
    return _svg(width, height, body)


def _scatter_panel(body, box, spec, points, lineage, x_range, y_range, arrow_id):
    """One accuracy-vs-error panel. points: dicts {id, x, y, kind, tip, label?, dx?, dy?}."""
    x0, y0, w, h = box
    xt, xs = nice_ticks(*x_range, 5)
    yt, ys = nice_ticks(*y_range, 4)

    def X(v):
        return x0 + (v - xt[0]) / (xt[-1] - xt[0]) * w

    def Y(v):
        return y0 + h - (v - yt[0]) / (yt[-1] - yt[0]) * h

    for t in yt:
        body.append(f'<line class="grid" x1="{x0}" x2="{x0 + w}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        body.append(
            f'<text class="tick" x="{x0 - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{_fmt(t, ys)}</text>'
        )
    for t in xt:
        body.append(f'<line class="grid" x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{y0}" y2="{y0 + h}"/>')
        body.append(
            f'<text class="tick" x="{X(t):.1f}" y="{y0 + h + 16}" text-anchor="middle">{_fmt(t, xs)}</text>'
        )
    body.append(f'<line class="axis" x1="{x0}" x2="{x0 + w}" y1="{y0 + h}" y2="{y0 + h}"/>')
    body.append(f'<text class="panel" x="{x0 - 40}" y="{y0 - 14}">{escape(spec["title"])}</text>')
    xlabel = spec.get("xlabel", "JevBench accuracy")
    body.append(
        f'<text class="muted" x="{x0 + w}" y="{y0 + h + 34}" text-anchor="end">{escape(xlabel)} &#8594;</text>'
    )
    better, corner = spec.get("better", ("better: right and down", "bottom"))
    by = y0 + h - 8 if corner == "bottom" else y0 + 14
    body.append(
        f'<text class="muted" x="{x0 + w - 4}" y="{by}" text-anchor="end">{escape(better)}</text>'
    )

    by_id = {p["id"]: p for p in points}
    path = [by_id[i] for i in lineage if i in by_id]
    for a, b in itertools.pairwise(path):  # the lineage, one arrow per adoption
        body.append(
            f'<line class="line1" x1="{X(a["x"]):.1f}" y1="{Y(a["y"]):.1f}" x2="{X(b["x"]):.1f}" '
            f'y2="{Y(b["y"]):.1f}" marker-end="url(#{arrow_id})"/>'
        )
    order = {"experiment": 0, "reference": 1, "default": 2}  # lineage drawn on top
    for p in sorted(points, key=lambda p: order[p["kind"]]):
        cx, cy = X(p["x"]), Y(p["y"])
        if p["kind"] == "reference":  # squares: shape as well as colour
            mark = f'<rect class="dot3" x="{cx - 5:.1f}" y="{cy - 5:.1f}" width="10" height="10" rx="2"/>'
        else:
            mark = f'<circle class="{"dot1" if p["kind"] == "default" else "dot2"}" cx="{cx:.1f}" cy="{cy:.1f}" r="5"/>'
        body.append(
            f'<g>{_title(p["tip"])}<circle class="hit" cx="{cx:.1f}" cy="{cy:.1f}" r="11"/>{mark}</g>'
        )
        if p.get("label"):
            dx, dy = p.get("dx", 8), p.get("dy", -8)
            anchor = "end" if dx < 0 else "middle" if dx == 0 else "start"
            if p.get("leader"):  # label set away from a crowded spot, joined by a hairline
                body.append(
                    f'<line class="ref" x1="{cx:.1f}" y1="{cy - 6:.1f}" x2="{cx + dx:.1f}" y2="{cy + dy + 3:.1f}"/>'
                )
            body.append(
                f'<text x="{cx + dx:.1f}" y="{cy + dy:.1f}" text-anchor="{anchor}">{escape(p["label"])}</text>'
            )


def trajectory(
    title,
    subtitle,
    panels,
    lineage,
    note=None,
    width=980,
    panel_h=360,
    ref_label="other systems, same 231 tasks",
):
    """Side-by-side accuracy-vs-error panels. panels: dicts {title, points, x_range, y_range}."""
    top = 104
    height = top + panel_h + 70 + (22 if note else 0)
    body = [
        f'<text class="title" x="24" y="30">{escape(title)}</text>',
        f'<text class="subtitle" x="24" y="50">{escape(subtitle)}</text>',
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path class="arrow" d="M0,1 L10,5 L0,9 z"/></marker></defs>',
    ]
    x = 24
    has_exp = any(p["kind"] == "experiment" for pan in panels for p in pan["points"])
    _legend(body, x, 74, with_experiments=has_exp)
    lx = x + 24 + text_width("adopted as the default recipe") + 24
    if has_exp:
        lx += 22 + text_width("experiment, not adopted") + 24
    body.append(
        f'<rect class="dot3" x="{lx + 4:.1f}" y="69" width="10" height="10" rx="2"/>'
        f'<text x="{lx + 20:.1f}" y="78">{escape(ref_label)}</text>'
    )
    pw = (width - 2 * 24 - 64 * len(panels) - 24 * (len(panels) - 1)) / len(panels)
    for i, pan in enumerate(panels):
        bx = 24 + 64 + i * (pw + 64 + 24)
        _scatter_panel(
            body,
            (bx, top + 20, pw, panel_h - 20),
            pan,
            pan["points"],
            lineage,
            pan["x_range"],
            pan["y_range"],
            "arr",
        )
    if note:
        body.append(f'<text class="muted" x="24" y="{height - 14}">{escape(note)}</text>')
    return _svg(width, height, body)


def _numeric_axes(body, box, x_range, y_range, xlabel, ylabel, x_fmt, y_fmt, y_ticks, cap, x_ticks):
    """Grid, ticks and axis labels for a plot with numeric x; returns (X, Y, y formatter)."""
    x0, top, pw, plot_h = box
    yt, ys = nice_ticks(*y_range, y_ticks)
    if cap is not None:
        yt = [t for t in yt if t <= cap + 1e-9]
    xlo, xhi = x_range
    if x_ticks is None:
        x_ticks, _ = nice_ticks(xlo, xhi, 5)
    xt = [t for t in x_ticks if xlo - 1e-9 <= t <= xhi + 1e-9]

    def X(v):
        return x0 + (v - xlo) / (xhi - xlo) * pw

    def Y(v):
        return top + plot_h - (v - yt[0]) / (yt[-1] - yt[0]) * plot_h

    yf = y_fmt or (lambda v: _fmt(v, ys))
    for t in yt:
        body.append(
            f'<line class="grid" x1="{x0}" x2="{x0 + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
        )
        body.append(
            f'<text class="tick" x="{x0 - 6}" y="{Y(t) + 3.5:.1f}" text-anchor="end">{yf(t)}</text>'
        )
    body.append(
        f'<line class="axis" x1="{x0}" x2="{x0 + pw}" y1="{top + plot_h}" y2="{top + plot_h}"/>'
    )
    for t in xt:
        body.append(
            f'<text class="tick" x="{X(t):.1f}" y="{top + plot_h + 16}" text-anchor="middle">{x_fmt.format(t)}</text>'
        )
    body.append(
        f'<text class="muted" x="{x0 + pw}" y="{top + plot_h + 34}" text-anchor="end">{escape(xlabel)} &#8594;</text>'
    )
    body.append(f'<text class="muted" x="{x0 - 40}" y="{top - 12}">{escape(ylabel)}</text>')
    return X, Y, yf


def line_chart(
    title,
    subtitle,
    series,
    xlabel,
    ylabel,
    x_range,
    y_range,
    notes=(),
    width=760,
    plot_h=300,
    cap=None,
    markers=False,
    x_fmt="{:,.0f}",
    y_fmt=None,
    y_ticks=4,
    x_ticks=None,
    x_name="step",
):
    """Lines over a numeric x axis (e.g. training step).

    series: dicts {label, end_label, cls ('0'-'3'), points [(x, y)]}. Two or more series get a
    legend and a direct label at each line's end; one series gets neither (the title names
    it). A hover strip per x value lists every series' value there, a static stand-in for a
    crosshair; with markers, each point is also a dot with its own tooltip. x_ticks fixes the
    x tick positions; x_name is what the tooltips call x.
    """
    multi = len(series) > 1
    top = 116 if multi else 84
    x0 = 72
    end_space = max((text_width(s["end_label"]) for s in series), default=0) + 20 if multi else 24
    pw = width - x0 - end_space
    height = top + plot_h + 44 + 18 * len(notes) + 10
    body = [
        f'<text class="title" x="24" y="30">{escape(title)}</text>',
        f'<text class="subtitle" x="24" y="50">{escape(subtitle)}</text>',
    ]
    if multi:  # legend, in series order
        lx = 24
        for s in series:
            body.append(
                f'<line class="line{s["cls"]}" x1="{lx}" x2="{lx + 18}" y1="74" y2="74"/>'
                f'<text x="{lx + 24}" y="78">{escape(s["label"])}</text>'
            )
            lx += 24 + text_width(s["label"]) + 22
    X, Y, yf = _numeric_axes(
        body,
        (x0, top, pw, plot_h),
        x_range,
        y_range,
        xlabel,
        ylabel,
        x_fmt,
        y_fmt,
        y_ticks,
        cap,
        x_ticks,
    )

    # hover strips, drawn under the lines
    xs = sorted({x for s in series for x, _ in s["points"]})
    at = [{x: y for x, y in s["points"]} for s in series]
    for i, x in enumerate(xs):
        left = X((xs[i - 1] + x) / 2) if i else X(x) - 4
        right = X((x + xs[i + 1]) / 2) if i + 1 < len(xs) else X(x) + 4
        vals = "; ".join(
            f"{s['label']} {yf(a[x]) if y_fmt else f'{a[x]:.3f}'}"
            for s, a in zip(series, at, strict=True)
            if x in a
        )
        body.append(
            f'<rect class="strip" x="{left:.1f}" y="{top}" width="{right - left:.1f}" height="{plot_h}">'
            f"{_title(f'{x_name} {x_fmt.format(x)}: {vals}')}</rect>"
        )

    for s in reversed(series):  # the first series is drawn last, on top
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{X(x):.1f},{Y(y):.1f}" for i, (x, y) in enumerate(s["points"])
        )
        body.append(f'<path class="line{s["cls"]}" d="{d}" pointer-events="none"/>')
        if markers:
            for x, y in s["points"]:
                tip = f"{x_name} {x_fmt.format(x)}: {s['label']} {yf(y) if y_fmt else f'{y:.3f}'}"
                body.append(
                    f'<g>{_title(tip)}<circle class="hit" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="11"/>'
                    f'<circle class="dot{s["cls"]}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4"/></g>'
                )

    if multi:  # direct labels at each line's end, spread to a 12px minimum gap
        ends = sorted(((Y(s["points"][-1][1]), s) for s in series), key=lambda t: t[0])
        placed = []
        for y, _ in ends:
            placed.append(max(y, placed[-1] + 12) if placed else y)
        for (_, s), ly in zip(ends, placed, strict=True):
            body.append(
                f'<text x="{X(s["points"][-1][0]) + 6:.1f}" y="{ly + 3.5:.1f}">{escape(s["end_label"])}</text>'
            )
    elif series:  # one series: label its last value
        x, y = series[0]["points"][-1]
        body.append(
            f'<text class="value" x="{X(x):.1f}" y="{Y(y) - 10:.1f}" text-anchor="end">'
            f"{yf(y) if y_fmt else f'{y:.3f}'}</text>"
        )
    for i, n in enumerate(notes):
        body.append(
            f'<text class="muted" x="24" y="{top + plot_h + 60 + 18 * i}">{escape(n)}</text>'
        )
    return _svg(width, height, body)


def scatter_chart(
    title,
    subtitle,
    points,
    xlabel,
    ylabel,
    x_range,
    y_range,
    refs=(),
    notes=(),
    width=760,
    plot_h=300,
    x_fmt="{:,.0f}",
    y_fmt=None,
    y_ticks=4,
    x_ticks=None,
):
    """One series of points over numeric axes, with optional horizontal reference rules.

    points: dicts {x, y, tip}. refs: (label, value) rules, labelled at the right edge and
    spread to a 12px minimum gap, as in the dot charts. Dots are translucent so dense
    regions read darker; each carries its own tooltip.
    """
    top, x0 = 84, 72
    ref_space = max((text_width(lbl) for lbl, _ in refs), default=0) + (16 if refs else 0)
    pw = width - x0 - 24 - ref_space
    height = top + plot_h + 44 + 18 * len(notes) + 10
    body = [
        f'<text class="title" x="24" y="30">{escape(title)}</text>',
        f'<text class="subtitle" x="24" y="50">{escape(subtitle)}</text>',
    ]
    X, Y, yf = _numeric_axes(
        body,
        (x0, top, pw, plot_h),
        x_range,
        y_range,
        xlabel,
        ylabel,
        x_fmt,
        y_fmt,
        y_ticks,
        None,
        x_ticks,
    )
    for p in points:
        body.append(
            f'<g>{_title(p["tip"])}<circle class="hit" cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="6"/>'
            f'<circle class="dot1s" cx="{X(p["x"]):.1f}" cy="{Y(p["y"]):.1f}" r="3.5"/></g>'
        )
    placed = sorted(((Y(v), lbl, v) for lbl, v in refs), key=lambda t: t[0])
    label_y = []
    for y, _, _ in placed:
        label_y.append(max(y, label_y[-1] + 12) if label_y else y)
    for (y, lbl, v), ly in zip(placed, label_y, strict=True):
        body.append(
            f"<g>{_title(f'{lbl}: {yf(v)}')}"
            f'<line class="ref" x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke-dasharray="4 3"/>'
            f'<text x="{x0 + pw + 6}" y="{ly + 3.5:.1f}">{escape(lbl)}</text></g>'
        )
    for i, n in enumerate(notes):
        body.append(
            f'<text class="muted" x="24" y="{top + plot_h + 60 + 18 * i}">{escape(n)}</text>'
        )
    return _svg(width, height, body)
