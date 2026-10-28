"""Re-derive every label in the minimal-pair corpus from its rendered document.

The same discipline that caught three defects in the first generator: read the
document back as a model would, recompute the answer independently, and compare with
the stored label. Also checks the property the whole design rests on -- that the two
halves of a pair differ in only a few lines and have different answers.

    python -m data.checks.verify_synth_pairs [n_pairs] [train|heldout]
"""

import difflib
import re
import sys
from collections import Counter, defaultdict

from training.data.synth_pairs import SKINS, generate_pairs


def _cells(line: str):
    line = line.strip()
    if line.startswith("|"):
        return [c.strip() for c in line.strip("|").split("|")]
    if ": " in line and "  " not in line.split(": ", 1)[0]:
        return [x.strip() for x in line.split(": ", 1)]
    return [c for c in re.split(r"\s{2,}", line) if c]


def _section(doc: str, head_re: str) -> list:
    """Lines of the first section whose header matches, up to the next blank line."""
    lines = doc.split("\n")
    for i, ln in enumerate(lines):
        if re.match(head_re, ln):
            out = []
            for x in lines[i + 1 :]:
                if not x.strip():
                    break
                out.append(x)
            return out
    raise KeyError(head_re)


def check_xref(it):
    s = SKINS[it.meta["skin"]]
    doc = it.state
    C, G, A = s["category"].upper(), s["condition"].upper(), s["assignee"].upper()
    E = s["entity"]

    table = _section(doc, rf"^{re.escape(C)} TABLE$")[1:]  # skip the prose line
    table = [r for r in table if not re.match(r"^\s*\|?\s*-{3}", r)]
    ent_cat = {}
    for r in table:
        c = _cells(r)
        if len(c) == 2 and c[0] != E:
            ent_cat[c[0]] = c[1]

    over_block = _section(doc, r"^OVERRIDE$")
    ocat = re.search(
        r"follow (.+?), whatever|handled as (.+?), whatever|routed to the (.+?) team"
        r"|allocated to the (.+?), whatever",
        over_block[0],
    )
    ocat = next(g for g in ocat.groups() if g)
    overridden = {ln.strip()[2:] for ln in over_block if ln.strip().startswith("- ")}

    groups = {}
    for r in _section(doc, rf"^{re.escape(G)} GROUPS$"):
        c = _cells(r)
        if len(c) == 2 and c[1] != "group" and not c[0].startswith("-"):
            groups[c[0]] = c[1]

    grid = _section(doc, rf"^{re.escape(A)} BY ")
    head = _cells(grid[0])[1:]
    assign = {}
    for r in grid[1:]:
        c = _cells(r)
        for g, name in zip(head, c[1:], strict=True):
            assign[(c[0], g)] = name

    rec = _section(doc, rf"^{re.escape(s['record'])}$")
    ent = next(
        ln.split(":", 1)[1].strip() for ln in rec if ln.strip().lower().startswith(E.lower())
    )
    val = next(
        ln.split(":", 1)[1].strip()
        for ln in rec
        if ln.strip().lower().startswith(s["condition"].split()[0].lower())
        and "note" not in ln.lower()
    )

    cat = ocat if ent in overridden else ent_cat[ent]
    want = assign[(cat, groups[val])]
    return it.options[it.label][0] == want, f"{ent}->{cat} {val}->{groups[val]} => {want}"


def check_req(it):
    doc = it.state
    req_lines = _section(doc, r"^Requirements:$")
    resp = _section(doc, r"^RESPONSE$")
    entries = [ln for ln in resp if not ln.startswith("Entries:")]
    countline = next((ln for ln in resp if ln.startswith("Entries:")), None)

    fails = []
    for rl in req_lines:
        t = rl.strip()[2:]
        if t.startswith("List exactly"):
            n = int(re.search(r"exactly (\d+)", t).group(1))
            if len(entries) != n:
                fails.append("count")
        elif t.startswith("Give every quantity in"):
            unit = t.rsplit(" ", 1)[1].rstrip(".")
            if any(not ln.rstrip().endswith(" " + unit) for ln in entries):
                fails.append("unit")
        elif t.startswith("Report every quantity to"):
            dp = int(re.search(r"to (\d+) decimal", t).group(1))
            for ln in entries:
                q = ln.split()[-2]
                if len(q.split(".")[1]) != dp:
                    fails.append("decimals")
                    break
        elif t.startswith("Order the entries"):
            asc = "ascending" in t
            qs = [float(ln.split()[-2]) for ln in entries]
            if qs != sorted(qs, reverse=not asc):
                fails.append("sort")
        elif t.startswith("Begin each entry"):
            if any(not re.match(r"^[A-Z]{2}-\d{3}\s", ln) for ln in entries):
                fails.append("prefix")
        elif t.startswith("Finish with a line"):
            if countline is None or int(countline.split(":")[1]) != len(entries):
                fails.append("countline")
        elif t.startswith("Write every date"):
            if any(not re.search(r"\s\d{4}-\d{2}-\d{2}\s", ln) for ln in entries):
                fails.append("date")
        else:
            raise ValueError(f"unparsed requirement: {t}")

    if len(fails) > 1:
        return False, f"AMBIGUOUS: {fails}"
    got = it.options[it.label][0]
    if it.kind == "noul":
        want = "false" if fails else "true"
    else:
        want = fails[0] if fails else "none"
    return got == want, f"fails={fails}"


n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
split = sys.argv[2] if len(sys.argv) > 2 else "train"
items = generate_pairs(n, seed=31337, split=split)

bad = defaultdict(list)
count = Counter()
for it in items:
    fn = check_xref if it.task == "syn_xref" else check_req
    count[it.task] += 1
    try:
        ok, why = fn(it)
    except Exception as exc:  # a parse failure is a defect too
        ok, why = False, f"{type(exc).__name__}: {exc}"
    if not ok:
        bad[it.task].append((it, why))

# The design property: halves of a pair differ minimally and disagree on the answer.
pairs = defaultdict(list)
for it in items:
    pairs[it.meta["pair"]].append(it)
same_label, diffs = 0, []
for p in pairs.values():
    a, b = p
    if a.options[a.label][0] == b.options[b.label][0]:
        same_label += 1
    d = [
        x
        for x in difflib.unified_diff(a.state.split("\n"), b.state.split("\n"), n=0)
        if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))
    ]
    diffs.append(len(d))

print(f"split={split}: re-derived {len(items)} items ({len(pairs)} pairs)\n")
for t in sorted(count):
    print(
        f"  {t:<14}{count[t]:>6} checked   {len(bad[t]):>4} disagree "
        f"({len(bad[t]) / count[t] * 100:5.1f}%)"
    )
print(f"\n  pairs whose halves share an answer : {same_label}")
print(
    f"  changed lines per pair             : min {min(diffs)}  max {max(diffs)}  "
    f"mean {sum(diffs) / len(diffs):.1f}"
)
for t, lst in [(t, x) for t, x in bad.items() if x]:
    it, why = lst[0]
    print(
        f"\n--- first {t} disagreement: {why}\n    label {it.options[it.label][0]!r}  meta {it.meta}"
    )
    print("    " + it.state[:900].replace("\n", "\n    "))
