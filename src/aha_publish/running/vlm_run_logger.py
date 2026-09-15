"""Per-run VLM event logger for the interactive BT runner.

Records ONE CSV row per VLM event during a single `waypoints_interactive_bt_conditions`
run -- both channels:
  * condition checks  (pre / post / prepost / arrival) -- the BT checkpoint VLM
  * detector fires    (slip / collision / orientation / transition / freezing)

and saves the exact camera montage sent to each VLM into one per-run folder.

The whole module is a no-op until `start(...)` is called (only the interactive
runner, when given --vlm-run-log-dir, does so), so importing it costs nothing and
existing runs are unaffected.

Token/cost attribution is channel-agnostic: `start()` monkeypatches the OpenAI
`Responses.create` call to accumulate usage, and each `log()` charges the delta
since the previous event to that event. Sequential (no concurrent VLM calls in
the interactive runner), so the delta is exactly this event's usage.
"""

from aha_publish import paths
import csv
import os
import sys
import time
from pathlib import Path

_LOGGER = None


class _Tee:
    """Write to several streams at once (real stdout + the run.log file).

    The first stream is the "primary" (the real stdout); any attribute this proxy
    does not define itself (encoding, isatty, fileno, buffer, ...) is delegated to
    it, so libraries that introspect sys.stdout (e.g. py_trees reading
    sys.stdout.encoding at import) keep working."""

    def __init__(self, primary, *others):
        self._primary = primary
        self._streams = (primary,) + others

    def write(self, s):
        for st in self._streams:
            try:
                st.write(s)
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def __getattr__(self, name):
        # Only reached for attributes not found on the instance/class above.
        return getattr(self._primary, name)


def active():
    """The live RunLogger, or None when logging is off (the common case)."""
    return _LOGGER


CSV_COLUMNS = [
    "task", "failure", "injection_waypoint", "injection_frame",
    "channel", "event_kind", "waypoint", "detection_frame",
    "verdict", "failure_detected", "explanation",
    "api_calls", "input_tokens", "output_tokens", "reasoning_tokens",
    "latency_s", "cost_usd", "image_path",
]

# failure-type -> responsible detector. Prefer the canonical map the eval code
# uses; fall back to prefix rules so a new failure type still resolves.
_PREFIX_RULES = (
    ("rotation", "orientation"),
    ("no_rotation", "orientation"),
    ("orientation", "orientation"),
    ("translation", "transition"),
    ("transition", "transition"),
    ("slip", "slip"),
    ("collision", "collision"),
    ("freez", "freezing"),
)


def responsible_detector_for(failtype):
    """Detector that owns `failtype` (e.g. slip->slip, rotation_z->orientation)."""
    if not failtype:
        return None
    ft = str(failtype).strip().lower()
    try:
        import aha_publish.running.eval_detectors_vlm as _ev
        mapping = {f: det for det, fts in _ev.DETECTOR_FAILTYPES.items() for f in fts}
        if ft in mapping:
            return mapping[ft]
    except Exception:
        pass
    for prefix, det in _PREFIX_RULES:
        if ft.startswith(prefix):
            return det
    return None


# USD per 1M tokens, OpenAI standard processing -- same table the eval/roll-up
# scripts use. Reasoning ("thinking") tokens are billed at the output rate and
# are already counted inside output_tokens, so charging output covers them.
OPENAI_PRICE_PER_1M = {
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.5": (5.00, 30.00),
}


def _prices(model=None):
    """(input_$per_1M, output_$per_1M) from env, else the price table."""
    for in_env, out_env in (("AHA_INPUT_PRICE_PER_1M", "AHA_OUTPUT_PRICE_PER_1M"),
                            ("AHA_VLM_INPUT_PRICE_PER_1M",
                             "AHA_VLM_OUTPUT_PRICE_PER_1M")):
        try:
            in_p = float(os.getenv(in_env, "") or "nan")
            out_p = float(os.getenv(out_env, "") or "nan")
            if in_p == in_p and out_p == out_p:      # both finite
                return in_p, out_p
        except (TypeError, ValueError):
            pass
    key = (model or "").strip().lower()
    if key in OPENAI_PRICE_PER_1M:
        return OPENAI_PRICE_PER_1M[key]
    # Dated / suffixed ids (gpt-5.4-2026-01-01) fall back to the longest
    # matching base name, so "-mini"/"-nano" still win over plain "gpt-5.4".
    matches = [k for k in OPENAI_PRICE_PER_1M if key.startswith(k)]
    if matches:
        return OPENAI_PRICE_PER_1M[max(matches, key=len)]
    return None


class RunLogger:
    def __init__(self, out_dir, task, failure, failure_waypoint, model):
        self.out_dir = Path(out_dir)
        self.images_dir = self.out_dir / "vlm_images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.out_dir / "vlm_events.csv"
        self.task = task
        self.failure = failure or "none"
        self.injection_waypoint = failure_waypoint
        self.injection_frame = ""
        self.model = model
        # cumulative usage since start; the delta since the previous event is
        # charged to each event.
        self._usage = {"calls": 0, "in": 0, "out": 0, "reasoning": 0,
                       "latency": 0.0}
        self._charged = dict(self._usage)
        self._price = _prices(model)
        self._rows = 0

        # Route each fire-verifier's saved montage into our one folder.
        os.environ["AHA_SLIP_VLM_SAVE_GRID"] = str(self.images_dir)
        os.environ["AHA_COLLISION_VLM_SAVE_GRID"] = str(self.images_dir)

        # Have the live detectors write per-step drift CSVs into the run folder,
        # so we can render the same trace plots the batch runner produces. These
        # env vars are read when each detector is constructed (later than now).
        self.drift = {d: self.out_dir / f"{d}_drift.csv"
                      for d in ("orientation", "transition", "slip", "collision")}
        os.environ["AHA_ORIENTATION_DRIFT_LOG"] = str(self.drift["orientation"])
        os.environ["AHA_TRANSITION_DRIFT_LOG"] = str(self.drift["transition"])
        os.environ["AHA_SLIP_DRIFT_LOG"] = str(self.drift["slip"])
        os.environ["AHA_COLLISION_DRIFT_LOG"] = str(self.drift["collision"])
        os.environ.setdefault("AHA_COLLISION_METHOD", "3")  # momentum observer (De Luca)

        # Tee stdout to run.log so the plot renderer can recover fire steps for
        # the orientation/transition traces (their fire markers come from stdout).
        self.run_log = self.out_dir / "run.log"
        # Explicit UTF-8: VLM explanations routinely contain non-ASCII (em dashes,
        # x, superscripts). Under a non-UTF-8 subprocess locale a default-encoding
        # open() would raise UnicodeEncodeError on those writes.
        self._logfile = open(self.run_log, "w", encoding="utf-8")
        self._orig_stdout = sys.stdout
        sys.stdout = _Tee(self._orig_stdout, self._logfile)

        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_COLUMNS)
        self._install_usage_hook()

    # -- usage accounting ---------------------------------------------------
    def _install_usage_hook(self):
        try:
            from openai.resources.responses import Responses
        except Exception:
            return
        if getattr(Responses.create, "_aha_wrapped", False):
            return
        orig = Responses.create
        usage = self._usage

        def wrapped(self_inner, *a, **kw):
            started = time.monotonic()
            try:
                resp = orig(self_inner, *a, **kw)
            finally:
                # Charge wall-clock time even for a call that raised, so a slow
                # failure is still visible in the latency column.
                usage["latency"] += time.monotonic() - started
            try:
                u = getattr(resp, "usage", None)
                if u is not None:
                    usage["calls"] += 1
                    usage["in"] += int(getattr(u, "input_tokens", 0) or 0)
                    usage["out"] += int(getattr(u, "output_tokens", 0) or 0)
                    details = getattr(u, "output_tokens_details", None)
                    if details is not None:
                        usage["reasoning"] += int(
                            getattr(details, "reasoning_tokens", 0) or 0)
            except Exception:
                pass
            return resp

        wrapped._aha_wrapped = True
        Responses.create = wrapped

    def _charge_delta(self):
        d = {k: self._usage[k] - self._charged[k] for k in self._usage}
        self._charged = dict(self._usage)
        cost = ""
        if self._price is not None:
            cost = round(d["in"] / 1e6 * self._price[0]
                         + d["out"] / 1e6 * self._price[1], 6)
        return d, cost

    # -- frame stamps -------------------------------------------------------
    def set_injection_frame(self, frame):
        if frame is not None and self.injection_frame == "":
            self.injection_frame = frame

    # -- event rows ---------------------------------------------------------
    def log(self, *, channel, event_kind, waypoint, verdict, explanation="",
            image_path="", detection_frame=None):
        """Append one CSV row. `verdict`: for conditions the bool
        checkpoint_passed (True=pass/no-failure, False=fail); for detectors the
        bool failure-confirmed (True=failure). None -> UNKNOWN. failure_detected
        is normalized so it means the same thing across channels."""
        if channel == "condition":
            vtext = ("PASS" if verdict is True else
                     "FAIL" if verdict is False else "UNKNOWN")
            failure_detected = (verdict is False)
        elif channel == "diagnosis":
            # Diagnosis rows carry the classified failure-type string directly
            # (e.g. 'slip', 'no_grasp'); write it verbatim instead of coercing to
            # a bool verdict. 'unknown'/empty means no decisive classification.
            vtext = str(verdict) if verdict else "unknown"
            failure_detected = vtext not in ("", "unknown")
        else:
            vtext = ("YES" if verdict is True else
                     "NO" if verdict is False else "UNKNOWN")
            failure_detected = (verdict is True)
        d, cost = self._charge_delta()
        row = {
            "task": self.task, "failure": self.failure,
            "injection_waypoint": ("" if self.injection_waypoint is None
                                   else self.injection_waypoint),
            "injection_frame": self.injection_frame,
            "channel": channel, "event_kind": event_kind,
            "waypoint": "" if waypoint is None else waypoint,
            "detection_frame": "" if detection_frame is None else detection_frame,
            "verdict": vtext, "failure_detected": failure_detected,
            "explanation": (explanation or "").replace("\n", " ").strip(),
            "api_calls": d["calls"], "input_tokens": d["in"],
            "output_tokens": d["out"], "reasoning_tokens": d["reasoning"],
            "latency_s": round(d["latency"], 3),
            "cost_usd": cost, "image_path": image_path or "",
        }
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)
        self._rows += 1

    def finalize(self, task, failure, waypoint):
        """Restore stdout and render the per-detector trace plots (same helpers
        the batch runner uses) from the drift CSVs written during the run."""
        try:
            sys.stdout = self._orig_stdout
            self._logfile.flush()
            self._logfile.close()
        except Exception:
            pass
        try:
            text = self.run_log.read_text(errors="ignore")
        except Exception:
            text = ""
        self.plots = {}
        try:
            import aha_publish.running.run_all_tasks_all_failures as R
            drift = {d: str(self.drift[d]) for d in self.drift}
            info = {d: R.EV.parse_run(text, d) for d in R.DETECTORS}
            label = ("clean" if not failure or failure == "none"
                     else f"{failure}@wp{waypoint}")
            self.plots = R.render_plots(
                str(self.out_dir), task, label, drift, info,
                stdout=text, failure=failure or "none", waypoint=waypoint)
        except Exception as exc:
            print(f"[vlm-log] detector plot render failed: {exc}")

    def summary_line(self):
        pngs = [str(p) for p in (getattr(self, "plots", {}) or {}).values()
                if isinstance(p, str) and p.endswith(".png")]
        line = (f"[vlm-log] {self._rows} VLM event(s) -> {self.csv_path}\n"
                f"[vlm-log] images -> {self.images_dir}")
        if pngs:
            line += f"\n[vlm-log] detector plots -> {self.out_dir} ({len(pngs)} png)"
        return line


def start(out_dir, task, failure, failure_waypoint, model):
    """Begin logging into out_dir. Returns the RunLogger (also stored globally)."""
    global _LOGGER
    _LOGGER = RunLogger(out_dir, task, failure, failure_waypoint, model)
    return _LOGGER


def default_out_dir(root, task, failure, failure_waypoint):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "clean" if not failure or failure == "none" else f"{failure}_wp{failure_waypoint}"
    return os.path.join(root, "aha_output", "bt_runs", f"{task}__{tag}__{stamp}")
