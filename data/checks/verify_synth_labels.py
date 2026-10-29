"""Re-derive every label by parsing the rendered document.

The generator computes its answer from internal variables and then renders a document.
If the two ever disagree -- a figure rounded for display, a clause that says something
the code does not do -- the corpus teaches the wrong thing and nothing downstream would
catch it. So this reads the document back as a model would and recomputes independently,
including a brute-force month adder rather than reusing `_add_months`.
"""

import calendar
import json
import re
import sys
from datetime import date

from training.data import synth as G

LB = 0.45359237


def months_bruteforce(d: date, n: int) -> date:
    """Add n months one at a time, clamping each step -- a different algorithm."""
    y, m = d.year, d.month
    for _ in range(n):
        m += 1
        if m == 13:
            m, y = 1, y + 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last))


def check_units(it):
    s = it.state
    limit = float(re.search(r"Permitted mass: (\d+) kg", s).group(1))
    rounding = re.search(r"round the total (\w+) to a whole", s).group(1)
    include = "+ protective wrap" in s
    comps = re.findall(
        r"^  (net load|carrier tare|securing gear|protective wrap)\s+([\d.]+) (kg|lb)$", s, re.M
    )
    total = 0.0
    for name, val, unit in comps:
        if name == "protective wrap" and not include:
            continue
        v = float(val)
        total += v * LB if unit == "lb" else v
    final = (
        -(-total // 1)
        if rounding == "up"
        else total // 1
        if rounding == "down"
        else float(round(total))
    )
    want = "true" if final <= limit else "false"
    return it.options[it.label][0] == want, f"total={total:.3f} final={final} limit={limit}"


def check_window(it):
    s = it.state
    months = int(re.search(r"runs for (\d+) months", s).group(1))
    inclusive = "included in" in s
    start = date.fromisoformat(re.search(r"Commencement date : (\S+)", s).group(1))
    event = date.fromisoformat(re.search(r"Event date        : (\S+)", s).group(1))
    end = months_bruteforce(start, months)
    inside = event <= end if inclusive else event < end
    want = "true" if inside else "false"
    return it.options[it.label][0] == want, f"end={end} event={event} incl={inclusive}"


def check_threshold(it):
    s = it.state
    trigger = float(re.search(r"\(([\d.]+) units\)", s).group(1))
    counted = re.search(r"Only (\w+) entries count", s).group(1)
    rows = re.findall(r"^  (\d{4}-\d{2}-\d{2})\s+([\d.]+)\s+(\w+)$", s, re.M)
    run = 0.0
    fired = None
    for d, amt, cat in rows:
        if cat == counted:
            run += float(amt)
        if fired is None and run >= trigger:
            fired = d
    return it.options[it.label][0] == fired, f"fired={fired} trigger={trigger}"


def _num(v):
    return float(v.replace(",", ""))


def check_bands(it):
    s = it.state
    low = re.search(r"^  below\s+([\d,]+)\s+(\w+)$", s, re.M)
    mids = re.findall(r"^\s+([\d,]+) to below\s+([\d,]+)\s+(\w+)$", s, re.M)
    top = re.search(r"^\s+([\d,]+) and above\s+(\w+)$", s, re.M)
    primary = _num(re.search(r"This request   :\s+([\d,.]+)", s).group(1))
    sibs = [_num(v) for v in re.findall(r"^    \S+\s+([\d,.]+)$", s, re.M)]
    agg = primary + sum(sibs)

    name = None
    if agg < _num(low.group(1)):
        name = low.group(2)
    for lo, hi, nm in mids:
        if _num(lo) <= agg < _num(hi):
            name = nm
    if agg >= _num(top.group(1)):
        name = top.group(2)
    return it.options[it.label][0] == name, f"agg={agg:,.2f} -> {name}"


def check_verify(it):
    d = json.loads(it.state)
    req, resp = d["request"], d["response"]
    a = float(re.search(r"holds ([\d.]+) units", req).group(1))
    pct = float(re.search(r"reduced by ([\d.]+)%", req).group(1))
    added = float(re.search(r"([\d.]+) units are added", req).group(1))
    dp = int(re.search(r"to (\d+) decimal place", req).group(1))
    truth = round(a * (1 - pct / 100) + added, dp)
    claim_s = re.search(r"final quantity ([\d.-]+)\.$", resp).group(1)
    claim = float(claim_s)
    # Count decimals on the STRING. Parsing to float first loses a trailing zero, so
    # a correctly rendered "230.130" looked like 2 dp and was wrongly flagged.
    shown_dp = len(claim_s.split(".")[1]) if "." in claim_s else 0
    ok = abs(claim - truth) < 10 ** -(dp + 2) and shown_dp == dp
    want = "true" if ok else "false"
    return it.options[it.label][0] == want, f"truth={truth} claim={claim_s} dp={dp}"


CHECKS = {
    "syn_units": check_units,
    "syn_window": check_window,
    "syn_threshold": check_threshold,
    "syn_bands": check_bands,
    "syn_verify": check_verify,
}

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
items = G.generate(n, seed=4242)
fails = {}
counts = {}
for it in items:
    fn = CHECKS.get(it.task)
    if fn is None:
        continue
    counts[it.task] = counts.get(it.task, 0) + 1
    try:
        ok, why = fn(it)
    except Exception as exc:
        ok, why = False, f"{type(exc).__name__}: {exc}"
    if not ok:
        fails.setdefault(it.task, []).append((it, why))

print(f"re-derived {sum(counts.values())} items from their rendered documents\n")
for t in sorted(counts):
    bad = fails.get(t, [])
    print(
        f"  {t:<16} {counts[t]:>5} checked   {len(bad):>4} disagree "
        f"({len(bad) / counts[t] * 100:5.1f}%)"
    )
if fails:
    print("\nfirst disagreement per generator:")
    for t, lst in fails.items():
        it, why = lst[0]
        print(f"\n--- {t}: {why}")
        print(f"    label says {it.options[it.label][0]!r}  meta={it.meta}")
        print("    " + it.state[:500].replace("\n", "\n    "))
