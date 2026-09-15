#!/usr/bin/env python3
"""Stage 2 of 2 -- score the detection sample table and print the results.

Reads the CSV make_detection_samples.py writes:

    task,failure_condition,waypoint,gt,B2,A1,A2,A3,A4

where every column -- `gt` and each method -- is 'success', 'failure,<Type>' or
(B2 only) 'uncertain', in the vocabulary the GPT baseline uses
(assess_reactive_gpt.FAILTYPE_TO_TYPES).  Prints the overall binary table and the
failure-specific table.  Nothing here touches a run directory, so re-scoring
under a different policy is instant.

The columns are the only source of both the label and the failure type, so
nothing has to be re-derived from the case name.

Detection is binary: any 'failure,*' is an alarm regardless of the type it names,
which is what the paper tables report.  Because the methods also name a type, the
console adds a `Type` column -- the share of true detections whose predicted type
matches the ground-truth type.  A fused A4 label ('Wrongtransition|Grasp') counts
when the truth is among the types it names.  Reported per failure too.

Scoring window
--------------
`--window to-injection` (the DEFAULT) runs each failure episode from its first
waypoint up to and including the waypoint the failure happens, and drops every
waypoint after it.  Each failure episode therefore contributes exactly ONE
positive -- the moment of failure -- and its earlier waypoints are the negatives.
This is what a reactive system experiences, since it would stop there.

Two kinds of episode are never truncated: clean runs, which have no injection, and
cases whose name carries no _wpN (wrong_sequence_v2 shuffles the whole waypoint
order) -- there is no single moment of failure to stop at, so every waypoint counts.

`--window full` instead keeps every checkpoint to the end of the episode.  The
injected failure persists, so every later checkpoint is a positive too and a
method is repeatedly re-credited for catching the same failure.

Sample unit
-----------
Default is the *checkpoint* -- one CSV row.  `--unit episode` collapses each
episode to a single prediction first: a failure episode counts as detected when
any checkpoint at or after the injection carries an alarm; a clean episode
counts as a false alarm when any checkpoint does.

Per-failure pools
-----------------
Failure type F is scored over the checkpoints of F's episodes plus every clean
episode, so precision and FPR are driven by real false alarms -- pre-injection
checkpoints and clean runs -- rather than by other failure types.

Empty cells
-----------
An empty method cell means that method has no result for the episode -- the GPT
baseline, for instance, has not been run on every case.  By default each method is
scored on whatever it does cover, so one method's gaps never shrink another's
sample set; the per-method coverage is printed so the columns can be read
correctly.  --common-episodes instead restricts every method to the episodes all
of them cover, which makes the columns directly comparable at the cost of dropping
episodes.

Latency
-------
Read from the `<name>_frames.csv` sidecar when it sits next to the samples file:
control steps between the injection frame and the first crediting alarm.  Without
it the Lat. rows print '--'.

Usage
-----
    python aha_scripts/score_detection_samples.py                   # all five
    python aha_scripts/score_detection_samples.py --methods B2 A4   # just those two
    python aha_scripts/score_detection_samples.py --matrix          # + matrices
    python aha_scripts/score_detection_samples.py --matrix --entropy  # + entropies
    python aha_scripts/score_detection_samples.py --matrix-image    # + the figure

By default the window stops at the failure waypoint, all five methods are
reported, and every misclassified checkpoint is listed.  --methods narrows the
columns; --window full scores to the end of each episode; --no-list-errors drops
the listing.

--matrix-image draws the per-type matrices, one figure per method -- B2 and A4
unless --matrix-image-methods says otherwise -- into
aha_output/paper_tables/confusion_B2.pdf and confusion_A4.pdf, or wherever the
flag's optional path points (.pdf and .svg work too).  Every cell shows its row
share over the count behind it, and colour is that share on a scale down the
right edge, so the rare failure types stay readable next to the 259-checkpoint
success row.  The baseline's 'uncertain' verdict is drawn in the success
column, which is where the scorer counts it.  The figures carry no title and no
caption -- only the class labels and one line of scores, since they are meant
to be dropped into a paper where the LaTeX caption says the rest.  It needs
matplotlib and says so if it is missing.

Table 1 reports micro (checkpoint-pooled) and macro (each failure type weighted
equally) side by side; the failure types differ a lot in episode count, so the
two halves can disagree.  Results go to the console and to the metrics CSVs.
"""

from __future__ import annotations

from aha_publish import paths

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = (paths.PROJECT_ROOT)
DEFAULT_SAMPLES = (paths.SCORES_DIR / 'detection_samples.csv')
DEFAULT_OUT_DIR = (paths.SCORES_DIR)

METHODS = ("B2", "A1", "A2", "A3", "A4")
METHOD_LABELS = {
    "B2": "B2: GPT-5.6",
    "A1": "A1: Condition VLM",
    "A2": "A2: Detectors",
    "A3": "A3: Det.+Conf.",
    "A4": "A4: Full",
}

# Paper row order / display names, keyed by the ground-truth type name that
# appears in the `gt` column.
FAILURE_ROWS = [
    ("Slip", "Grasp loss"),
    ("collision", "Collision"),
    ("Wrongtransition", "Transition dev."),
    ("WrongOrientation", "Orientation dev."),
    ("freezing", "Freezing"),
    ("WrongObjectSelection", "Wrong object"),
    ("Grasp", "No grasp"),
    ("Wrong Sequence", "Sequence violation"),
]

# A method is structurally blind to a failure type when nothing in its input can
# express it directly:
#   * the condition VLM has no predicate for collision or freezing;
#   * the detector bank has no detector for grasp / wrong object / sequence.
# Such a method can still score above zero through downstream effects -- a
# collision that knocks the object out of the gripper violates object_in_gripper,
# so A1 catches it without ever naming it -- which is why the numbers are shown by
# default and `applicable` in metrics_per_failure.csv records the blindness
# instead.  --mask-blind prints these cells as '--' (the paper convention).
BLIND = {
    "A1": {"collision", "freezing"},
    "A2": {"Grasp", "WrongObjectSelection", "Wrong Sequence"},
    "A3": {"Grasp", "WrongObjectSelection", "Wrong Sequence"},
}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def parse_label(value: str) -> tuple[str, str | None] | None:
    """'failure,collision' -> ('failure', 'collision'); 'success' -> ('success', None).

    None (not a tuple) means the cell is empty: no result for this checkpoint.
    Unknown predictions count as success; in mixed labels only known types remain.
    """
    value = (value or "").strip()
    if not value:
        return None
    verdict, _, name = value.partition(",")
    verdict = verdict.strip().lower()
    if verdict == "unknown":
        return "success", None
    if verdict == "failure":
        names = [part.strip() for part in name.split("|")
                 if part.strip().lower() != "unknown"]
        if not names:
            return "success", None
        return "failure", ("|".join(names).strip() or "?")
    return verdict, None


def _int(value):
    try:
        if value in ("", None):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_samples(path: Path) -> list[dict]:
    """One dict per episode: key, failure type, and the per-waypoint rows."""
    if not path.exists():
        sys.exit(f"no sample table at {path}; run make_detection_samples.py first")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        sys.exit(f"{path} is empty")
    missing = [c for c in ("task", "failure_condition", "waypoint", "gt", *METHODS)
               if c not in rows[0]]
    if missing:
        sys.exit(f"{path} is missing column(s): {', '.join(missing)}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["failure_condition"])].append(row)

    episodes = []
    for (task, case), case_rows in grouped.items():
        case_rows.sort(key=lambda r: _int(r["waypoint"]) or 0)
        gt = [parse_label(r["gt"]) or ("success", None) for r in case_rows]
        # The episode's failure type is whatever its positive checkpoints name;
        # an episode with none is a clean run.
        failure = next((t for v, t in gt if v == "failure"), None)
        preds = {}
        for method in METHODS:
            parsed = [parse_label(r[method]) for r in case_rows]
            # A method with no result for the episode leaves every cell empty.
            preds[method] = None if all(p is None for p in parsed) else [
                p or ("success", None) for p in parsed]
        episodes.append({
            "task": task,
            "case": case,
            "failure": failure,
            "waypoints": [_int(r["waypoint"]) for r in case_rows],
            "gt": gt,
            "preds": preds,
        })
    return episodes


# A case name without a trailing _wpN has no single injection waypoint -- the
# whole episode is corrupted (wrong_sequence_v2 shuffles every waypoint), so
# there is no "before the failure" to truncate to.
NO_INJECTION_WAYPOINT = re.compile(r"_wp\d+$")


def apply_window(episodes: list[dict], window: str) -> list[dict]:
    """Truncate failure episodes at their injection waypoint when asked."""
    if window != "to-injection":
        return episodes
    out = []
    for episode in episodes:
        gts = [v == "failure" for v, _ in episode["gt"]]
        if not any(gts):
            out.append(episode)                     # clean run: nothing to cut
            continue
        if not NO_INJECTION_WAYPOINT.search(episode["case"]):
            out.append(episode)                     # corrupted throughout
            continue
        keep = gts.index(True) + 1                  # 0..inj inclusive
        trimmed = dict(episode)
        trimmed["gt"] = episode["gt"][:keep]
        trimmed["waypoints"] = episode["waypoints"][:keep]
        trimmed["preds"] = {m: (None if p is None else p[:keep])
                            for m, p in episode["preds"].items()}
        out.append(trimmed)
    return out


def load_frames(path: Path) -> dict[tuple[str, str], dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            out[(row["task"], row["failure_condition"])] = row
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def confusion(pairs) -> dict:
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for gt, pred in pairs:
        counts["TP" if gt and pred else
               "FN" if gt else
               "FP" if pred else "TN"] += 1
    return counts


def metrics(counts: dict) -> dict:
    tp, fp, fn, tn = counts["TP"], counts["FP"], counts["FN"], counts["TN"]

    def ratio(num, den):
        return 100.0 * num / den if den else None

    prec, rec = ratio(tp, tp + fp), ratio(tp, tp + fn)
    if prec is None or rec is None:
        f1 = None
    elif prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    return {"acc": ratio(tp + tn, tp + fp + fn + tn), "prec": prec, "rec": rec,
            "f1": f1, "fpr": ratio(fp, fp + tn), **counts}


def is_alarm(label: tuple[str, str | None], uncertain_is_failure: bool) -> bool:
    verdict = label[0]
    return verdict == "failure" or (uncertain_is_failure and verdict == "uncertain")


def pairs_for(episodes, method: str, unit: str, uncertain_is_failure=False):
    """(ground truth, prediction) booleans, one pair per sample."""
    for episode in episodes:
        preds = episode["preds"][method]
        if preds is None:
            continue
        gts = [v == "failure" for v, _ in episode["gt"]]
        alarms = [is_alarm(p, uncertain_is_failure) for p in preds]
        if unit == "episode":
            gt = episode["failure"] is not None
            # Credit only an alarm at or after the injection waypoint.
            pred = any(a for g, a in zip(gts, alarms) if g) if gt else any(alarms)
            yield gt, pred
        else:
            yield from zip(gts, alarms)


def type_accuracy(episodes, method: str, uncertain_is_failure=False):
    """Share of true detections whose predicted type matches the truth."""
    hits = total = 0
    for episode in episodes:
        preds = episode["preds"][method]
        if preds is None:
            continue
        for (gt_verdict, gt_type), pred in zip(episode["gt"], preds):
            if gt_verdict != "failure" or not is_alarm(pred, uncertain_is_failure):
                continue
            total += 1
            # A fused label ('Wrongtransition|Grasp') counts when the truth is
            # among the types it names.
            hits += int(gt_type in str(pred[1]).split("|"))
    return (100.0 * hits / total) if total else None


def class_matrix(episodes, method: str, uncertain_is_failure=False):
    """(true class, predicted class) -> count, over every checkpoint.

    Classes are 'success' plus the failure type names.  A fused prediction
    ('collision|WrongOrientation') resolves to the true class when that class is
    among the names -- the set-based convention assess_reactive_gpt.py uses for
    type scoring -- and to its first, highest-precedence name otherwise.
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for episode in episodes:
        preds = episode["preds"][method]
        if preds is None:
            continue
        for (gt_verdict, gt_type), (verdict, ptype) in zip(episode["gt"], preds):
            true_class = gt_type if gt_verdict == "failure" else "success"
            if not is_alarm((verdict, ptype), uncertain_is_failure):
                pred_class = verdict            # 'success' or 'uncertain'
            else:
                # An alarm that names no type is 'uncertain' counted as one
                # under --uncertain-as failure; it keeps its own verdict as its
                # class rather than becoming a column called 'None'.
                names = str(ptype).split("|") if ptype else [verdict]
                pred_class = true_class if true_class in names else names[0]
            counts[(true_class, pred_class)] += 1
    return counts


def misclassified(episodes, method: str, uncertain_is_failure=False):
    """Every checkpoint this method got wrong, as (kind, task, case, wp, gt, pred).

    kind is 'missed failure' (a real failure called success/uncertain),
    'false alarm' (an alarm on a clean checkpoint), or 'wrong type' (the failure
    was detected but none of the names it gave match the truth).
    """
    out = []
    for episode in episodes:
        preds = episode["preds"][method]
        if preds is None:
            continue
        for wp, ((gt_verdict, gt_type), pred) in zip(episode["waypoints"],
                                                     zip(episode["gt"], preds)):
            alarm = is_alarm(pred, uncertain_is_failure)
            raw = pred[0] if pred[1] is None else f"{pred[0]},{pred[1]}"
            if gt_verdict == "failure" and not alarm:
                kind = "missed failure"
            elif gt_verdict != "failure" and alarm:
                kind = "false alarm"
            elif gt_verdict == "failure" and gt_type not in str(pred[1]).split("|"):
                kind = "wrong type"
            else:
                continue
            out.append((kind, episode["task"], episode["case"], wp,
                        gt_type or "success", raw))
    order = {"missed failure": 0, "false alarm": 1, "wrong type": 2}
    return sorted(out, key=lambda r: (order[r[0]], r[1], r[2], r[3]))


def _entropy(p) -> float:
    """Shannon entropy in bits of a probability vector (zeros contribute 0)."""
    from math import log2
    total = sum(x * log2(x) for x in p if x > 0)
    return -total if total else 0.0     # `or 0.0` would also swallow a real 0


def matrix_entropies(counts: dict) -> dict:
    """Entropy summary of a class confusion matrix read as a Markov kernel.

    Rows (true class) are normalised into a transition kernel P: P[i][j] is the
    chance a checkpoint of true class i is called j.  Predicted labels that are
    not also true classes -- 'uncertain', which no ground truth ever carries --
    have no row, so they are dropped and the rows renormalised; `dropped` reports
    the share of mass that cost, since a method that answers 'uncertain' a lot is
    being scored on the remainder of its answers.

    Two readings of "stationary", because they answer different questions:

    markov   pi solves pi P = pi: the class mix the kernel would settle into if
             its output were fed back in as the next truth.  This is a property
             of P alone and ignores how often each class actually occurs, so it
             is the right lens on the kernel's structure -- but it degenerates to
             a point mass whenever some class is an absorbing state (a row that
             only ever predicts itself), which a perfect row is.
    empirical pi is the observed class mix (row totals / N).  Nothing settles;
             this is just the dataset as it stands, and the transition entropy is
             then exactly H(pred | true) over these checkpoints.

    Transition entropy is pi-weighted mean row entropy either way: the average
    bits of uncertainty left about the prediction once the true class is known --
    0.0 for a method that maps every class to a fixed label.  Stationary entropy
    is H(pi), the spread of the class mix itself.

    Returns {} when the matrix has no usable rows.
    """
    rows = sorted({t for t, _ in counts})
    idx = {c: i for i, c in enumerate(rows)}
    totals = [sum(v for (t, _), v in counts.items() if t == c) for c in rows]
    kept = [sum(v for (t, p), v in counts.items() if t == c and p in idx)
            for c in rows]
    if not rows or not any(kept):
        return {}
    n = sum(totals)
    dropped = 1.0 - sum(kept) / n if n else 0.0
    # Rows that lose all their mass to dropped labels get a self-loop, keeping P
    # stochastic without inventing a preference.
    P = [[(counts.get((c, p), 0) / kept[i]) if kept[i] else float(p == c)
          for p in rows] for i, c in enumerate(rows)]
    row_H = [_entropy(row) for row in P]

    import numpy as np
    values, vectors = np.linalg.eig(np.array(P, dtype=float).T)
    stat = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    stat = np.abs(stat)
    markov = (stat / stat.sum()).tolist() if stat.sum() else [1 / len(rows)] * len(rows)
    empirical = [t / n for t in totals]

    return {
        "classes": rows, "dropped": dropped, "row_H": row_H,
        "markov": {"pi": markov,
                   "stationary_H": _entropy(markov),
                   "transition_H": sum(w * h for w, h in zip(markov, row_H))},
        "empirical": {"pi": empirical,
                      "stationary_H": _entropy(empirical),
                      "transition_H": sum(w * h for w, h in zip(empirical, row_H))},
        "max_H": _entropy([1 / len(rows)] * len(rows)),
        "absorbing": [c for c in rows if P[idx[c]][idx[c]] == 1.0],
    }


def print_entropies(ent: dict) -> None:
    """Print the matrix_entropies summary under a confusion matrix."""
    if not ent:
        print("\n  entropies: no usable rows")
        return
    width = max(len(c) for c in ent["classes"]) + 2
    print(f"\n  entropy (bits; uniform over {len(ent['classes'])} classes "
          f"= {ent['max_H']:.3f})")
    if ent["dropped"] > 1e-9:
        print(f"    {ent['dropped'] * 100:.1f}% of mass dropped as 'uncertain' "
              f"before normalising")
    for name in ("markov", "empirical"):
        e = ent[name]
        note = ("  (degenerate -- absorbing state)"
                if name == "markov" and ent["absorbing"] else "")
        print(f"    {name + ':':<11} stationary H(pi) = {e['stationary_H']:.3f}"
              f"   transition H = {e['transition_H']:.3f}{note}")
    if ent["absorbing"]:
        print(f"    absorbing states (row predicts only itself): "
              f"{', '.join(ent['absorbing'])}"
              f"  -> pi collapses onto it; read the empirical row instead")
    print("    per-class row entropy H(pred | true = c), with both pi:")
    print("      " + "class".ljust(width) + "row H".rjust(9)
          + "pi_markov".rjust(12) + "pi_emp".rjust(10))
    for c, h, pm, pe in zip(ent["classes"], ent["row_H"],
                            ent["markov"]["pi"], ent["empirical"]["pi"]):
        print("      " + c.ljust(width) + f"{h:.3f}".rjust(9)
              + f"{pm:.3f}".rjust(12) + f"{pe:.3f}".rjust(10))


# --------------------------------------------------------------------------- #
# confusion-matrix image
# --------------------------------------------------------------------------- #
# A confusion matrix is magnitude on a grid, so the whole palette is ONE
# sequential hue read light -> dark (blue steps 100..700).  Nothing here is
# categorical: no class gets its own colour, because the classes have no order
# and colouring them would double-encode the count the cell already prints.
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]

# Dark mode is the same hue re-stepped for the dark surface, not a flipped
# image: 'near zero' has to recede toward whatever surface the figure sits on,
# so the ramp runs dark -> light there.
THEMES = {
    "light": {"surface": "#fcfcfb", "plane": "#f9f9f7", "ink": "#0b0b0b",
              "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "ramp": SEQ_BLUE},
    "dark": {"surface": "#1a1a19", "plane": "#0d0d0d", "ink": "#ffffff",
             "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "ramp": SEQ_BLUE[::-1]},
}

CLASS_LABELS = dict(FAILURE_ROWS)
CLASS_LABELS.update({"success": "Success", "uncertain": "Uncertain"})

FONT_STACK = ["Helvetica Neue", "Helvetica", "Nimbus Sans", "Liberation Sans",
              "Arial", "DejaVu Sans"]


def _class_label(name: str) -> str:
    return CLASS_LABELS.get(name, name)


def _ink_on(rgb) -> str:
    """White or near-black, whichever clears contrast on this cell fill.

    The one place text is allowed to sit inside a coloured fill, so the choice
    is made from the fill's luminance rather than from the theme.
    """
    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (lin(c) for c in rgb[:3])
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.40 else "#0b0b0b"


def _binary_from_matrix(counts: dict) -> dict:
    """TP/FP/FN/TN of the drawn matrix, so a panel's caption always matches it.

    'uncertain' is not an alarm here: class_matrix has already resolved it to a
    failure name whenever --uncertain-as failure is in force, so any surviving
    'uncertain' cell is by definition a non-alarm.
    """
    out = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for (true_class, pred_class), value in counts.items():
        gt = true_class != "success"
        alarm = pred_class not in ("success", "uncertain")
        out["TP" if gt and alarm else "FN" if gt else
            "FP" if alarm else "TN"] += value
    return out


def _fold_uncertain(counts: dict) -> dict:
    """Move the 'uncertain' column into 'success'.

    The baseline's abstain verdict is not an alarm, so the scorer already counts
    it exactly as a 'success' call; giving it a column of its own only splits
    one behaviour across two places.  Not applied under --uncertain-as failure,
    where 'uncertain' is an alarm and does not belong in the success column.
    """
    folded: dict[tuple[str, str], int] = defaultdict(int)
    for (true_class, pred_class), value in counts.items():
        folded[(true_class,
                "success" if pred_class == "uncertain" else pred_class)] += value
    return dict(folded)


def _matrix_axes_order(matrices: dict) -> tuple[list[str], dict[str, list[str]]]:
    """Row order shared by every figure, plus each figure's column order.

    Rows follow the paper's failure order with 'success' last, and the order is
    taken across every method being drawn -- the figures are meant to be read
    beside each other, so a class missing from one method's data still gets its
    row there.  Columns are that same order (the square core, which puts the
    diagonal on the diagonal) followed by any label a method predicts that no
    ground truth carries, which is why the figures can differ in width.
    """
    seen = {t for counts in matrices.values() for t, _ in counts}
    rows = [f for f, _ in FAILURE_ROWS if f in seen]
    rows += sorted(c for c in seen if c not in rows and c != "success")
    if "success" in seen:
        rows.append("success")
    cols = {}
    for method, counts in matrices.items():
        extra = sorted({p for (_, p), v in counts.items()
                        if v and p not in rows})
        cols[method] = rows + extra
    return rows, cols


def _image_path(path: Path, method: str, one_only: bool) -> Path:
    """Where one method's figure goes: <stem>_<METHOD><suffix>, unless it is the
    only method asked for and the caller named a file outright."""
    if one_only:
        return path
    return path.with_name(f"{path.stem}_{method}{path.suffix}")


def _draw_matrix(mpl, method: str, counts: dict, rows: list[str],
                 ccols: list[str], path: Path, pal: dict, dpi: int) -> Path:
    """One method's confusion matrix, on its own page.

    Colour is the row-normalised share -- a count ramp would paint the one
    259-checkpoint 'success' row and leave the 2-checkpoint wrong-object row
    invisible -- on a linear scale, so a pale cell really does mean a small
    share of its row.  Each cell prints that share, with the checkpoint count
    it came from underneath, which keeps the grid a table you can still read as
    a picture.

    Nothing is written on the figure but the class labels, the cells, the row
    totals and the score line: it is meant to go into a paper, where the method
    is named and everything else is explained by the LaTeX caption beside it.
    """
    np, plt, LinearSegmentedColormap, to_rgb, FancyBboxPatch = mpl
    box = dict(boxstyle="round,pad=0,rounding_size=0.09")
    # The surface anchors the low end so an empty cell reads as the page, and
    # the ramp climbs from there in one hue.
    cmap = LinearSegmentedColormap.from_list(
        "seq", [pal["surface"]] + pal["ramp"])
    totals = {r: sum(counts.get((r, c), 0) for c in ccols) for r in rows}

    # ---- geometry, in inches ------------------------------------------- #
    CELL, PAD = 0.86, 0.020          # cell pitch; surface gap, in cell units
    CHAR = 0.077                     # ~width of a character at 14pt
    row_w = max(len(_class_label(r)) for r in rows) * CHAR + 0.20
    col_h = max(len(_class_label(c)) for c in ccols) * CHAR * 0.71 + 0.14
    bar_gap, bar_w, bar_lab = 0.55, 0.26, 0.70   # scale rail on the right
    m_l, m_r = 0.26, 0.22
    top = 0.50 + col_h               # the score line, then the column labels
    bot = 0.26
    grid_h = len(rows) * CELL
    fig_w = (m_l + row_w + len(ccols) * CELL
             + bar_gap + bar_w + bar_lab + m_r)
    fig_h = top + grid_h + bot

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(pal["surface"])

    met = metrics(_binary_from_matrix(counts))
    # Diagonal over the failure rows / detections on them -- the same quantity
    # the tables' `Type` column reports, read straight off the cells drawn here.
    diag = sum(counts.get((c, c), 0) for c in rows if c != "success")
    type_acc = 100.0 * diag / met["TP"] if met["TP"] else None
    pct = lambda v: "--" if v is None else f"{v:.1f}%"      # noqa: E731
    fig.text(m_l / fig_w, (fig_h - 0.34) / fig_h,
             f"Recall {pct(met['rec'])}   ·   Precision {pct(met['prec'])}"
             f"   ·   Type accuracy {pct(type_acc)}",
             ha="left", va="bottom", fontsize=15, color=pal["ink"])

    ax = fig.add_axes([(m_l + row_w) / fig_w, bot / fig_h,
                       len(ccols) * CELL / fig_w, len(rows) * CELL / fig_h])
    ax.set_xlim(0, len(ccols))
    ax.set_ylim(len(rows), 0)                        # first row at the top
    ax.set_facecolor(pal["surface"])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Hairlines in the surface gap: above the negative class, and before any
    # column that is not part of the square core.
    if rows and rows[-1] == "success":
        ax.axhline(len(rows) - 1, color=pal["axis"], lw=0.9, zorder=0.5)
    if len(ccols) > len(rows):
        ax.axvline(len(rows), color=pal["axis"], lw=0.9, zorder=0.5)

    for i, true_class in enumerate(rows):
        total = totals[true_class]
        for j, pred_class in enumerate(ccols):
            value = counts.get((true_class, pred_class), 0)
            share = value / total if total else 0.0
            rgb = to_rgb(cmap(share))
            corner = (j + PAD, i + PAD), 1 - 2 * PAD, 1 - 2 * PAD
            if value:                # an empty cell is left as bare surface
                ax.add_patch(FancyBboxPatch(*corner, **box, facecolor=rgb,
                                            edgecolor="none", zorder=1))
            if true_class == pred_class:             # find the diagonal fast
                ax.add_patch(FancyBboxPatch(*corner, **box, facecolor="none",
                                            edgecolor=pal["axis"], lw=0.9,
                                            zorder=3))
            if not value:
                # The diagonal keeps its ring whether or not anything landed
                # there, so an empty one says 0% rather than sitting blank --
                # the method never once named this class right.  `total` guards
                # the class no checkpoint carries at all: that is 0/0, not 0%.
                if true_class == pred_class and total:
                    ax.text(j + 0.5, i + 0.42, "0%", ha="center", va="center",
                            fontsize=16, color=pal["muted"], zorder=4)
                    ax.text(j + 0.5, i + 0.72, "0", ha="center", va="center",
                            fontsize=12, color=pal["muted"], alpha=0.70,
                            zorder=4)
                continue
            ink = _ink_on(rgb)
            # A share under half a point would round to a bare '0%', which
            # reads as 'never happened' on a cell that did happen.
            ax.text(j + 0.5, i + 0.42,
                    f"{share * 100:.0f}%" if share >= 0.005 else "<1%",
                    ha="center", va="center", fontsize=16, color=ink,
                    zorder=4)
            ax.text(j + 0.5, i + 0.72, f"{value}", ha="center", va="center",
                    fontsize=12, color=ink, alpha=0.70, zorder=4)

        ax.text(-0.22, i + 0.5, _class_label(true_class), ha="right",
                va="center", fontsize=14,
                color=pal["ink"] if true_class != "success" else pal["ink2"])

    for j, pred_class in enumerate(ccols):
        ax.text(j + 0.5, -0.20, _class_label(pred_class), ha="left",
                va="bottom", rotation=45, rotation_mode="anchor",
                fontsize=14, color=pal["ink2"])

    # ---- the colour scale, a rail down the right edge --------------------- #
    bar_x = m_l + row_w + len(ccols) * CELL + bar_gap
    bar = fig.add_axes([bar_x / fig_w, bot / fig_h,
                        bar_w / fig_w, grid_h / fig_h])
    bar.imshow(np.linspace(1, 0, 256).reshape(-1, 1), aspect="auto", cmap=cmap,
               extent=(0.0, 1.0, 0.0, 100.0))
    bar.set_xticks([])
    bar.set_yticks([0, 25, 50, 75, 100])
    bar.set_yticklabels(["0", "25", "50", "75", "100%"])
    bar.yaxis.tick_right()
    bar.tick_params(axis="y", length=0, pad=4.0, labelsize=13,
                    colors=pal["muted"])
    for spine in bar.spines.values():
        spine.set_visible(False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=pal["surface"])
    plt.close(fig)
    return path


def save_matrix_images(episodes, methods, path: Path, uncertain_is_failure=False,
                       theme="light", dpi=220) -> list[Path]:
    """One figure per method, all sharing a row order so they can be read as a set."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap, to_rgb
        from matplotlib.patches import FancyBboxPatch
    except ImportError as exc:
        print(f"\n  (--matrix-image needs matplotlib: {exc})")
        return []
    mpl = (np, plt, LinearSegmentedColormap, to_rgb, FancyBboxPatch)
    # Before anything is measured -- the caption is wrapped against the font
    # that will draw it, so the stack has to be in place first.
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": FONT_STACK})

    matrices = {}
    for method in methods:
        counts = class_matrix(episodes, method, uncertain_is_failure)
        if not counts:
            print(f"\n  (--matrix-image: {method} has no result in this table "
                  f"-- figure skipped)")
            continue
        # 'uncertain' is scored as a success call, so it is drawn as one: a
        # column of its own would split one behaviour across two places.
        matrices[method] = (counts if uncertain_is_failure
                            else _fold_uncertain(counts))
    if not matrices:
        print("\n  (--matrix-image: nothing to draw)")
        return []

    rows, cols = _matrix_axes_order(matrices)
    return [_draw_matrix(mpl, method, counts, rows, cols[method],
                         _image_path(path, method, len(matrices) == 1),
                         THEMES[theme], dpi)
            for method, counts in matrices.items()]


def print_matrix(title: str, counts: dict) -> None:
    rows = sorted({t for t, _ in counts}, key=lambda c: (c == "success", c))
    cols = sorted({p for _, p in counts}, key=lambda c: (c not in
                  ("success", "uncertain"), c))
    width = max([len(c) for c in cols] + [5]) + 2
    label = max([len(r) for r in rows] + [10]) + 2
    print(f"\n{title}")
    print("  " + "true \\ pred".ljust(label) + "".join(c.rjust(width) for c in cols)
          + "total".rjust(width))
    for row in rows:
        line = [counts.get((row, col), 0) for col in cols]
        print("  " + row.ljust(label)
              + "".join(str(v).rjust(width) for v in line)
              + str(sum(line)).rjust(width))


# Macro-averaged columns.  Read the accuracy with care: every per-type pool
# shares the same clean episodes, so a per-type accuracy is partly a measure of
# how many clean checkpoints came along with that type, and the macro mean
# inherits that.  Precision / recall / F1 are the sound comparisons here.
MACRO_KEYS = ("acc", "prec", "rec", "f1", "fpr", "type", "lat")


def macro_metrics(per_failure: dict, method: str, mask_blind: bool) -> dict:
    """Unweighted mean of one method's per-failure-type scores.

    Table 1 pools checkpoints, so the frequent failure types decide it -- 29
    transition episodes against 2 wrong-object ones.  The macro average weights
    every failure type equally instead, which is the number to read when the
    question is how broadly a method works rather than how it does on the
    average checkpoint.

    A type contributes only where the method has a defined score for it, and
    with mask_blind the types it is structurally blind to drop out as well.  The
    two conventions average over different type sets, so `n` reports how many
    types went into each method's mean -- without it the columns are not
    comparable.
    """
    used = []
    for failure, _ in FAILURE_ROWS:
        entry = per_failure.get(failure)
        if not entry or method not in entry:
            continue                      # no episode of this type in the table
        if mask_blind and entry[method]["blind"]:
            continue
        used.append(entry[method])
    out = {"n": len(used)}
    for key in MACRO_KEYS:
        values = [m[key] for m in used if m.get(key) is not None]
        out[key] = (sum(values) / len(values)) if values else None
    return out


def latency(episodes, frames: dict, method: str):
    """Mean control steps from injection to the first crediting alarm."""
    gaps = []
    for episode in episodes:
        if episode["failure"] is None:
            continue
        row = frames.get((episode["task"], episode["case"]))
        if not row:
            continue
        alarm = _int(row.get(f"{method}_alarm_frame"))
        inj = _int(row.get("injection_frame"))
        if alarm is None or inj is None:
            continue
        gaps.append(max(0, alarm - inj))
    return (sum(gaps) / len(gaps)) if gaps else None


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def fmt(value, blind=False) -> str:
    return "--" if blind or value is None else f"{value:.1f}"


def print_table(title: str, header: list[str], rows: list[list],
                groups: list[tuple[str, int]] | None = None) -> None:
    """`groups` spans a banner row over the columns, as [(label, n_cols), ...]."""
    body = [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]
    print(f"\n{title}")
    if groups:
        banner, first = [], 0
        for label, span in groups:
            width = sum(widths[first:first + span]) + 2 * (span - 1)
            banner.append(label.center(width))
            first += span
        print("  " + "  ".join(banner))
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for row in body:
        print("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--frames", type=Path, default=None,
                        help="latency sidecar (default: <samples>_frames.csv)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="where the metrics CSVs go")
    parser.add_argument("--unit", choices=("checkpoint", "episode"),
                        default="checkpoint")
    parser.add_argument("--window", choices=("to-injection", "full"),
                        default="to-injection",
                        help="'to-injection' (default) keeps waypoints 0..inj and "
                             "drops everything after the failure waypoint, so each "
                             "failure episode contributes exactly one positive; "
                             "'full' keeps every checkpoint to the end")
    parser.add_argument("--methods", nargs="+", default=list(METHODS),
                        metavar="M",
                        help=f"methods to report (any of {', '.join(METHODS)}, "
                             f"or 'all'). Default: all of them")
    parser.add_argument("--list-errors", dest="list_errors",
                        action="store_true", default=True,
                        help="list every misclassified checkpoint (default on)")
    parser.add_argument("--no-list-errors", dest="list_errors",
                        action="store_false",
                        help="suppress the misclassified-checkpoint listing")
    parser.add_argument("--matrix", action="store_true",
                        help="also print each method's binary and per-type "
                             "confusion matrix")
    parser.add_argument("--matrix-image", nargs="?", const="", default=None,
                        metavar="PATH",
                        help="save each method's per-type confusion matrix as "
                             "its own figure (default: B2 and A4, to "
                             "<out-dir>/confusion_B2.pdf and confusion_A4.pdf). "
                             "Give a path to put them elsewhere -- the method "
                             "goes into the filename; .pdf/.svg work too. "
                             "Independent of --matrix, which only prints them")
    parser.add_argument("--matrix-image-methods", nargs="+", default=["B2", "A4"],
                        metavar="M",
                        help="which methods get a figure, one file each "
                             "(default: B2 A4)")
    parser.add_argument("--matrix-image-theme", choices=("light", "dark"),
                        default="light",
                        help="figure palette (default: light)")
    parser.add_argument("--matrix-image-dpi", type=int, default=220,
                        help="figure resolution (default: 220)")
    parser.add_argument("--entropy", action="store_true",
                        help="with --matrix, also read each per-type matrix as a "
                             "Markov kernel and print its stationary and "
                             "transition entropies (bits)")
    parser.add_argument("--common-episodes", dest="common", action="store_true",
                        default=False,
                        help="restrict every method to the episodes all reported "
                             "methods cover, so the columns are computed on one "
                             "sample set")
    parser.add_argument("--no-common-episodes", dest="common",
                        action="store_false",
                        help="score each method on whatever it covers (default)")
    parser.add_argument("--uncertain-as", choices=("success", "failure"),
                        default="success",
                        help="whether the baseline's 'uncertain' verdict counts "
                             "as an alarm (default: it does not)")
    parser.add_argument("--no-write", action="store_true",
                        help="print only; do not write the metrics CSVs")
    parser.add_argument("--mask-blind", action="store_true",
                        help="print '--' where a method is structurally blind to "
                             "the failure type (A1 on collision/freezing, A2/A3 "
                             "on grasp/wrong object/sequence) instead of the "
                             "score it reaches through downstream effects")
    return parser.parse_args()


def main():
    args = parse_args()
    if any(m.lower() == "all" for m in args.methods):
        args.methods = list(METHODS)
    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        sys.exit(f"unknown method(s): {', '.join(unknown)}; "
                 f"choose from {', '.join(METHODS)}")
    selected = [m for m in METHODS if m in args.methods]   # keep paper order
    if args.matrix_image is not None:
        if any(m.lower() == "all" for m in args.matrix_image_methods):
            args.matrix_image_methods = list(METHODS)
        unknown = [m for m in args.matrix_image_methods if m not in METHODS]
        if unknown:
            sys.exit(f"--matrix-image-methods: unknown method(s): "
                     f"{', '.join(unknown)}; choose from {', '.join(METHODS)}")
    if selected != list(METHODS):
        print(f"Reporting {', '.join(selected)} only "
              f"(--methods all for {', '.join(METHODS)}). "
              f"The metrics CSVs get just these columns.")
    episodes = load_samples(args.samples)
    frames = load_frames(args.frames or
                         args.samples.with_name(args.samples.stem + "_frames.csv"))

    # Only the reported methods decide the common episode set.
    incomplete = [e for e in episodes
                  if any(e["preds"][m] is None for m in selected)]
    if incomplete and args.common:
        print_table(
            "Episodes with a method missing (excluded from all methods; "
            "--no-common-episodes keeps them)",
            ["task", "failure_condition", "missing"],
            [(e["task"], e["case"],
              ",".join(m for m in selected if e["preds"][m] is None))
             for e in incomplete])
    if args.common:
        episodes = [e for e in episodes if e not in incomplete]
    if not episodes:
        sys.exit("every episode was filtered out; try --no-common-episodes")

    episodes = apply_window(episodes, args.window)
    clean = [e for e in episodes if e["failure"] is None]
    n_cp = sum(len(e["gt"]) for e in episodes)
    note = (f"unit={args.unit}; window={args.window}; {len(episodes)} episodes "
            f"({len(clean)} clean), {n_cp} checkpoints, "
            f"from {args.samples.name}")

    # With gaps allowed, each column may rest on a different sample set -- say so.
    coverage = []
    for method in selected:
        covered = [e for e in episodes if e["preds"][method] is not None]
        cps = sum(len(e["gt"]) for e in covered)
        gaps = sorted({e["task"] for e in episodes
                       if e["preds"][method] is None})
        coverage.append([METHOD_LABELS[method], len(covered), cps,
                         ", ".join(gaps) if gaps else "-"])
    if any(row[1] != len(episodes) for row in coverage):
        print_table(
            "Per-method coverage (columns below rest on different sample sets)",
            ["Method", "episodes", "checkpoints", "tasks with gaps"], coverage)
    if not frames:
        print("\n  (no latency sidecar found -- Lat. rows will print '--')")

    # ---- per-failure pools ----------------------------------------------- #
    # Computed before table 1 is printed, because table 1's macro half is the
    # mean of these.
    unc = args.uncertain_as == "failure"
    overall = {m: metrics(confusion(pairs_for(episodes, m, args.unit, unc)))
               for m in selected}
    for method in selected:
        overall[method]["type"] = type_accuracy(episodes, method, unc)

    per_failure: dict[str, dict | None] = {}
    for failure, _ in FAILURE_ROWS:
        subset = [e for e in episodes if e["failure"] == failure]
        if not subset:
            per_failure[failure] = None
            continue
        pool = subset + clean
        entry = {}
        for method in selected:
            met = metrics(confusion(pairs_for(pool, method, args.unit, unc)))
            met["lat"] = latency(subset, frames, method)
            met["type"] = type_accuracy(subset, method, unc)
            met["blind"] = failure in BLIND.get(method, set())
            entry[method] = met
        per_failure[failure] = entry

    # ---- table 1: micro and macro side by side ---------------------------- #
    # Micro pools every checkpoint, so the failure types with the most episodes
    # decide it; macro weights each failure type equally.  A method that only
    # works on the common failures scores far better on the left than the right.
    macro = {m: macro_metrics(per_failure, m, args.mask_blind) for m in selected}
    scored = [label for failure, label in FAILURE_ROWS if per_failure.get(failure)]
    panel = ("acc", "prec", "rec", "f1", "fpr", "type")
    heads = ["Acc.", "Prec.", "Rec.", "F1", "FPR", "Type"]
    print_table(
        f"Table 1 -- overall binary detection ({note})",
        ["Method"] + heads + heads + ["TP", "FP", "FN", "TN"],
        [[METHOD_LABELS[m]]
         + [fmt(overall[m][k]) for k in panel]
         + [fmt(macro[m][k]) for k in panel]
         + [overall[m][k] for k in ("TP", "FP", "FN", "TN")] for m in selected],
        groups=[("", 1), ("micro (per checkpoint)", len(heads)),
                (f"macro ({len(scored)} failure types)", len(heads)),
                ("micro counts", 4)])
    print(f"   macro types: {', '.join(scored)}")
    if args.mask_blind and len({macro[m]["n"] for m in selected}) > 1:
        print("   --mask-blind drops each method's blind types, so the macro "
              "half averages over different type sets: "
              + ", ".join(f"{m}={macro[m]['n']}" for m in selected))

    # ---- table 2 ---------------------------------------------------------- #
    for failure, label in FAILURE_ROWS:
        entry = per_failure.get(failure)
        if entry is None:
            print(f"\n  (no {failure} episode in {args.samples.name} "
                  f"-- row left as '--')")
            continue
        subset = [e for e in episodes if e["failure"] == failure]
        print_table(
            f"Table 2 -- {label} ({failure}; {len(subset)} episodes "
            f"+ {len(clean)} clean)",
            ["Method", "Prec.", "Rec.", "FPR", "Lat.", "Type", "TP", "FP",
             "FN", "TN"],
            [[METHOD_LABELS[m]]
             + [fmt(entry[m][k], args.mask_blind and entry[m]["blind"])
                for k in ("prec", "rec", "fpr", "lat", "type")]
             + [entry[m][k] for k in ("TP", "FP", "FN", "TN")] for m in selected])

    if args.matrix:
        pos = sum(1 for e in episodes for v, _ in e["gt"] if v == "failure")
        print(f"\n{'=' * 72}\nConfusion matrices -- window={args.window}, "
              f"{n_cp} checkpoints ({pos} failure, {n_cp - pos} success)\n"
              f"{'=' * 72}")
        for method in selected:
            met = overall[method]
            print(f"\n{METHOD_LABELS[method]}")
            print(f"  binary        pred:failure  pred:success")
            print(f"  true:failure  {met['TP']:>12}  {met['FN']:>12}")
            print(f"  true:success  {met['FP']:>12}  {met['TN']:>12}")
            counts = class_matrix(episodes, method, unc)
            print_matrix(f"  per type (rows sum to the true class's checkpoints)",
                         counts)
            if args.entropy:
                print_entropies(matrix_entropies(counts))

    if args.matrix_image is not None:
        image_methods = [m for m in METHODS if m in args.matrix_image_methods]
        out = (Path(args.matrix_image) if args.matrix_image else
               args.out_dir / "confusion.pdf")
        for written in save_matrix_images(
                episodes, image_methods, out, uncertain_is_failure=unc,
                theme=args.matrix_image_theme, dpi=args.matrix_image_dpi):
            print(f"\nwrote {written}")

    if args.list_errors:
        for method in selected:
            errors = misclassified(episodes, method, unc)
            tally = Counter(kind for kind, *_ in errors)
            summary = ", ".join(f"{tally[k]} {k}" for k in
                                ("missed failure", "false alarm", "wrong type")
                                if tally[k])
            print_table(
                f"Misclassified checkpoints -- {METHOD_LABELS[method]} "
                f"({len(errors)} of {n_cp}: {summary or 'none'})",
                ["kind", "task", "failure_condition", "wp", "truth", "predicted"],
                [list(row) for row in errors])

    if args.no_write:
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with (args.out_dir / "metrics_overall.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        keys = ("acc", "prec", "rec", "f1", "fpr", "type",
                "TP", "FP", "FN", "TN")
        writer.writerow(["averaging", "method", *keys, "n"])
        for method in selected:
            met = overall[method]
            writer.writerow(["micro", method] + [met[k] for k in keys]
                            + [met["TP"] + met["FP"] + met["FN"] + met["TN"]])
        for method in selected:
            met = macro[method]
            # n is the number of failure types averaged, not a checkpoint count.
            writer.writerow(["macro", method, ""]
                            + [met[k] for k in ("prec", "rec", "f1", "fpr",
                                                "type")]
                            + ["", "", "", ""] + [met["n"]])

    with (args.out_dir / "metrics_per_failure.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["failure", "method", "applicable", "prec", "rec", "fpr",
                         "latency_steps", "type_accuracy",
                         "TP", "FP", "FN", "TN", "n"])
        for failure, _ in FAILURE_ROWS:
            entry = per_failure.get(failure)
            if not entry:
                continue
            for method in selected:
                met = entry[method]
                writer.writerow(
                    [failure, method, not met["blind"]]
                    + [met[k] for k in ("prec", "rec", "fpr", "lat", "type",
                                        "TP", "FP", "FN", "TN")]
                    + [met["TP"] + met["FP"] + met["FN"] + met["TN"]])

    print(f"\nwrote {args.out_dir}/"
          "{metrics_overall.csv,metrics_per_failure.csv}")


if __name__ == "__main__":
    main()
