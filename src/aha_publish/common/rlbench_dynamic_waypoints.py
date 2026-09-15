#!/usr/bin/env python3
"""Which waypoints an RLBench task repositions at RUN TIME.

Why this exists
---------------
The waypoint chain (waypoint_chain.py) composes a waypoint's world pose once per
episode from parent-local offsets that inspect_ttm.py read out of the simulator.
That is only valid for a waypoint whose pose really IS a fixed offset from its
parent.  RLBench tasks routinely break that assumption: ``init_episode`` and the
``register_waypoint_ability_start`` hooks call ``set_pose`` / ``set_position`` /
``set_orientation`` on a waypoint dummy, so the pose is chosen at run time from
whichever object the episode's variation selected.

    open_jar.init_episode:
        w0 = Dummy('waypoint0')
        w0.set_position([0, 0, 0.1], relative_to=self.lids[index % 2])

A chain that grounds such a waypoint composes a pose that is wrong by a fixed
transform for the whole episode, and every waypoint parented below it inherits
the same error.  The transition and orientation detectors compare the arm's
arrival pose against that reference, so they fire at every waypoint of a CLEAN
run -- which is exactly what stack_blocks, open_jar, light_bulb_in and
place_shape_in_shape_sorter do today.

The chain code already has a concept for this (``dynamic_waypoints``), but
nothing ever populated it: ``spec_from_inspection_report`` did not emit the key,
so both the runner's warning and check_waypoint_chain's warning were unreachable.
This module supplies the missing input.

How
---
Statically, from the task's own source with ``ast``.  No simulator, no import of
the task module (importing pulls in PyRep).  Three rules, each of which alone is
enough to mark a waypoint dynamic:

  1. A waypoint dummy that a setter is called on.  Bindings are tracked from
     ``<name> = Dummy('waypointN')`` -- both locals (``w2 = ...``) and attributes
     (``self.waypoint1 = ...``), since place_shape_in_shape_sorter binds in
     ``init_task`` and sets in a hook called much later.

  2. An ability hook whose body writes to the waypoint it was registered for,
     via its own parameter: ``waypoint.set_pose(...)`` or
     ``waypoint.get_waypoint_object().set_position(...)``.  The index comes from
     the ``register_waypoint_ability_start(N, self._fn)`` call.

  3. Any waypoint of a task that calls ``register_waypoints_should_repeat``.
     The whole sequence runs several times per episode with different poses each
     pass (stack_blocks stacks 2-4 blocks), so a single per-waypoint reference
     cannot describe the episode even where the offsets themselves are static.

Rule 1 is deliberately name-based rather than a full dataflow analysis: RLBench
tasks address waypoints by literal name, and a missed binding costs a false
NEGATIVE (a waypoint stays grounded that should not be), which is the same
failure we have today -- never a false positive that would silently drop a
usable reference.

Usage
-----
    python3 rlbench_dynamic_waypoints.py                    # every task
    python3 rlbench_dynamic_waypoints.py stack_blocks       # one task, verbose
    python3 rlbench_dynamic_waypoints.py --stamp            # write the reports

``--stamp`` writes the ``dynamic_waypoints`` key into
ttm_inspection_reports/*.llm_context.json, which is where waypoint_chain.py
reads it from.
"""

from __future__ import annotations

from aha_publish import paths

import argparse
import ast
import glob
import json
import os
import re
import sys
from pathlib import Path

HERE = (paths.SOURCE_DIR)

# The task sources. RLBench is used from its own checkout, not vendored under
# aha/Data_Generation (which carries only backend/, not tasks/).
DEFAULT_TASKS_DIR = Path(
    os.getenv("AHA_RLBENCH_TASKS_DIR")
    or (paths.RLBENCH_ROOT / "rlbench/tasks"))

DEFAULT_REPORT_DIR = (paths.TTM_CONTEXT_DIR)

WAYPOINT_NAME = re.compile(r"^waypoint(\d+)$")

# Pose writes. set_parent is included: re-parenting invalidates a stored
# parent-local offset just as thoroughly as moving the object does.
SETTERS = frozenset({
    "set_pose", "set_position", "set_orientation", "set_quaternion",
    "set_matrix", "set_parent",
})

# The hook registrars. *_start and *_end both run while the episode is live.
ABILITY_REGISTRARS = frozenset({
    "register_waypoint_ability_start", "register_waypoint_ability_end",
})

REPEAT_REGISTRAR = "register_waypoints_should_repeat"


def _attr_path(node):
    """Dotted source text of a Name/Attribute chain, or None.

    ``self.waypoint1`` -> 'self.waypoint1', ``w2`` -> 'w2'. Anything else
    (subscripts, calls) is not a stable binding name and returns None.
    """
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _dummy_waypoint_index(node):
    """Index behind ``Dummy('waypointN')``, else None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
    if name != "Dummy" or not node.args:
        return None
    arg = node.args[0]
    if not isinstance(arg, ast.Constant) or not isinstance(arg.value, str):
        return None
    match = WAYPOINT_NAME.match(arg.value)
    return int(match.group(1)) if match else None


def _setter_target(node):
    """For a call like ``X.set_pose(...)``, the dotted receiver X, else None.

    ``X.get_waypoint_object().set_position(...)`` unwraps to X as well: the
    ability hooks reach the dummy through that accessor.
    """
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return None
    if node.func.attr not in SETTERS:
        return None
    receiver = node.func.value
    if (isinstance(receiver, ast.Call)
            and isinstance(receiver.func, ast.Attribute)
            and receiver.func.attr == "get_waypoint_object"):
        receiver = receiver.func.value
    return _attr_path(receiver)


class _TaskAnalyzer(ast.NodeVisitor):
    """Collects waypoint bindings, setter writes and hook registrations."""

    def __init__(self):
        # dotted name -> waypoint index, for `self.x = Dummy('waypointN')`.
        # Attributes outlive the function that bound them.
        self.attr_bindings: dict[str, int] = {}
        # (function name, local name) -> index, for `w2 = Dummy('waypointN')`.
        self.local_bindings: dict[tuple[str, str], int] = {}
        # (function name, receiver) for every setter call seen.
        self.setter_calls: list[tuple[str, str]] = []
        # function name -> waypoint index it is registered as a hook for.
        self.hook_index: dict[str, int] = {}
        # hook functions that write through their own `waypoint` parameter.
        self.hooks_writing_self: set[str] = set()
        self.repeats = False
        self._func: list[str] = []

    # -- structure ---------------------------------------------------------
    def visit_FunctionDef(self, node):
        self._func.append(node.name)
        # The hook parameter is whatever the function's first non-self arg is
        # called; RLBench uses `waypoint` or `_`.
        args = [a.arg for a in node.args.args if a.arg != "self"]
        param = args[0] if args else None
        for sub in ast.walk(node):
            target = _setter_target(sub)
            if target is not None and param and target == param:
                self.hooks_writing_self.add(node.name)
        self.generic_visit(node)
        self._func.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    # -- bindings ----------------------------------------------------------
    def visit_Assign(self, node):
        index = _dummy_waypoint_index(node.value)
        if index is not None:
            for target in node.targets:
                name = _attr_path(target)
                if name is None:
                    continue
                if "." in name:
                    self.attr_bindings[name] = index
                elif self._func:
                    self.local_bindings[(self._func[-1], name)] = index
        self.generic_visit(node)

    # -- writes and registrations -----------------------------------------
    def visit_Call(self, node):
        target = _setter_target(node)
        if target is not None:
            self.setter_calls.append((self._func[-1] if self._func else "", target))

        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in ABILITY_REGISTRARS and len(node.args) >= 2:
            index_arg, fn_arg = node.args[0], node.args[1]
            if isinstance(index_arg, ast.Constant) and isinstance(index_arg.value, int):
                fn_name = fn_arg.attr if isinstance(fn_arg, ast.Attribute) else getattr(
                    fn_arg, "id", None)
                if fn_name:
                    self.hook_index[fn_name] = index_arg.value
        elif name == REPEAT_REGISTRAR:
            self.repeats = True
        self.generic_visit(node)


def analyze_source(source: str, waypoint_count: int | None = None) -> dict:
    """`{'dynamic': [idx...], 'repeats': bool, 'reasons': {idx: reason}}`.

    `waypoint_count` is only needed to expand the repeat rule, which taints
    every waypoint rather than a particular one.
    """
    tree = ast.parse(source)
    analyzer = _TaskAnalyzer()
    analyzer.visit(tree)

    reasons: dict[int, str] = {}
    runtime_dynamic = set()

    # Rule 1 -- a setter called on something bound to a waypoint dummy.
    for func, receiver in analyzer.setter_calls:
        index = analyzer.attr_bindings.get(receiver)
        if index is None:
            index = analyzer.local_bindings.get((func, receiver))
        if index is None:
            continue
        # Only direct initialization writes can be explained by anchors inferred
        # across resets. Helpers/hooks may also run during execution: abstain.
        if func not in ('init_task', 'init_episode'):
            runtime_dynamic.add(index)
        reasons.setdefault(
            index, f"{func or '<module>'}() calls a setter on {receiver}")

    # Rule 2 -- an ability hook writing through its own waypoint parameter.
    for fn_name in analyzer.hooks_writing_self:
        index = analyzer.hook_index.get(fn_name)
        if index is not None:
            runtime_dynamic.add(index)
            reasons.setdefault(
                index, f"ability hook {fn_name}() writes its own waypoint")

    # Rule 3 -- a repeating waypoint sequence taints the whole sequence.
    if analyzer.repeats and waypoint_count:
        for index in range(waypoint_count):
            runtime_dynamic.add(index)
            reasons.setdefault(
                index, f"{REPEAT_REGISTRAR}: the sequence repeats with new poses")

    return {
        "dynamic": sorted(reasons),
        "runtime_dynamic": sorted(runtime_dynamic),
        "repeats": analyzer.repeats,
        "reasons": reasons,
    }


def analyze_task(task_name: str, tasks_dir=None, waypoint_count=None) -> dict | None:
    """Analyze one task by name, or None when its source is not on disk."""
    base = Path(tasks_dir or DEFAULT_TASKS_DIR)
    path = base / f"{task_name}.py"
    if not path.is_file():
        return None
    result = analyze_source(path.read_text(encoding="utf-8"), waypoint_count)
    result["source"] = str(path)
    return result


# --------------------------------------------------------------------------- #
# report stamping
# --------------------------------------------------------------------------- #
def _report_paths(report_dir):
    return sorted(glob.glob(os.path.join(str(report_dir), "*.llm_context.json")))


def _report_task_name(path, report):
    """Prefer the report's own task_name; fall back to the filename."""
    name = report.get("task_name")
    if name:
        return name
    stem = Path(path).name[: -len(".llm_context.json")]
    head, _, rest = stem.partition("_")
    return rest if head.isdigit() and rest else stem


def _reported_waypoint_count(task_name, report_dir=None):
    """How many real waypoints the task's report lists, or None if there is none."""
    report_dir = Path(report_dir or DEFAULT_REPORT_DIR)
    for path in _report_paths(report_dir):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        if _report_task_name(path, report) != task_name:
            continue
        return len([w for w in report.get("waypoints") or []
                    if WAYPOINT_NAME.match(w.get("name", ""))])
    return None


def unresolved_dynamic_waypoints(result, report):
    """Validated reset anchors explain initialization writes, never live hooks."""
    validated = {int(WAYPOINT_NAME.match(w['name']).group(1))
                 for w in report.get('waypoints', [])
                 if WAYPOINT_NAME.match(w.get('name', ''))
                 and w.get('anchor_validation_episodes', 0) >= 3
                 and w.get('position_parent') and w.get('orientation_parent')
                 and not w.get('reference_unavailable')}
    return sorted((set(result['dynamic']) - validated)
                  | set(result.get('runtime_dynamic', result['dynamic'])))


def stamp_reports(report_dir=None, tasks_dir=None, dry_run=False):
    """Write `dynamic_waypoints` into every inspection report. Returns a summary."""
    report_dir = Path(report_dir or DEFAULT_REPORT_DIR)
    changed, skipped, missing = [], [], []
    for path in _report_paths(report_dir):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        task = _report_task_name(path, report)
        count = len([w for w in report.get("waypoints") or []
                     if WAYPOINT_NAME.match(w.get("name", ""))])
        result = analyze_task(task, tasks_dir, waypoint_count=count)
        if result is None:
            missing.append(task)
            continue
        # Only waypoints the report actually carries; a rule-3 expansion over a
        # count we guessed wrong should not invent indices.
        listed = {int(WAYPOINT_NAME.match(w["name"]).group(1))
                  for w in report.get("waypoints") or []
                  if WAYPOINT_NAME.match(w.get("name", ""))}
        dynamic = unresolved_dynamic_waypoints(result, report)
        if listed:
            dynamic = sorted(set(dynamic) & listed)
        if report.get("dynamic_waypoints") == dynamic:
            skipped.append(task)
            continue
        report["dynamic_waypoints"] = dynamic
        report["dynamic_waypoint_reasons"] = {
            str(i): result["reasons"][i] for i in dynamic}
        if not dry_run:
            # Match how the reports are already written (indent=2, no trailing
            # newline) so stamping shows up as the added keys and not as a
            # reformat of every tracked file.
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(report, indent=2))
        changed.append((task, dynamic))
    return {"changed": changed, "unchanged": skipped, "no_source": missing}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", nargs="?", help="one task (default: every report)")
    parser.add_argument("--tasks-dir", default=None,
                        help=f"RLBench tasks/ (default: {DEFAULT_TASKS_DIR})")
    parser.add_argument("--report-dir", default=None,
                        help=f"inspection reports (default: {DEFAULT_REPORT_DIR})")
    parser.add_argument("--stamp", action="store_true",
                        help="write dynamic_waypoints into the reports")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --stamp, show what would change and write nothing")
    args = parser.parse_args()

    if args.task:
        # The repeat rule needs the waypoint count, which only the report knows.
        result = analyze_task(
            args.task, args.tasks_dir,
            waypoint_count=_reported_waypoint_count(args.task, args.report_dir))
        if result is None:
            print(f"{args.task}: no source under "
                  f"{args.tasks_dir or DEFAULT_TASKS_DIR}")
            return 1
        print(f"{args.task}  ({result['source']})")
        print(f"  repeats: {result['repeats']}")
        if not result["dynamic"]:
            print("  dynamic waypoints: none -- every waypoint is a static "
                  "offset from its parent")
        for index in result["dynamic"]:
            print(f"  waypoint{index}: {result['reasons'][index]}")
        return 0

    if args.stamp:
        summary = stamp_reports(args.report_dir, args.tasks_dir, args.dry_run)
        verb = "would update" if args.dry_run else "updated"
        for task, dynamic in summary["changed"]:
            print(f"  {verb} {task}: dynamic_waypoints={dynamic}")
        print(f"\n{len(summary['changed'])} {verb}, "
              f"{len(summary['unchanged'])} already correct, "
              f"{len(summary['no_source'])} without task source")
        if summary["no_source"]:
            print("  no source: " + ", ".join(sorted(summary["no_source"])))
        return 0

    report_dir = Path(args.report_dir or DEFAULT_REPORT_DIR)
    rows = []
    for path in _report_paths(report_dir):
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        task = _report_task_name(path, report)
        count = len([w for w in report.get("waypoints") or []
                     if WAYPOINT_NAME.match(w.get("name", ""))])
        result = analyze_task(task, args.tasks_dir, waypoint_count=count)
        if result is None:
            rows.append((task, None, count, "no task source"))
            continue
        listed = {int(WAYPOINT_NAME.match(w["name"]).group(1))
                  for w in report.get("waypoints") or []
                  if WAYPOINT_NAME.match(w.get("name", ""))}
        dynamic = sorted(set(result["dynamic"]) & listed) if listed else result["dynamic"]
        note = "; ".join(dict.fromkeys(
            result["reasons"][i] for i in dynamic)) if dynamic else ""
        rows.append((task, dynamic, count, note))

    affected = [r for r in rows if r[1]]
    print(f"{'task':<38}{'wps':>4}  dynamic waypoints")
    print("-" * 100)
    for task, dynamic, count, note in sorted(
            rows, key=lambda r: (-len(r[1] or []), r[0])):
        if not dynamic:
            continue
        print(f"{task:<38}{count:>4}  {dynamic}  <- {note}")
    print(f"\n{len(affected)} of {len(rows)} tasks reposition at least one "
          f"waypoint at run time; their chain references are invalid.")
    no_source = [r[0] for r in rows if r[1] is None]
    if no_source:
        print(f"{len(no_source)} report(s) have no task source and were not "
              f"analyzed: {', '.join(sorted(no_source))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
