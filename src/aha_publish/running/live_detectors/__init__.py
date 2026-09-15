"""Live runtime detectors embedded in the bt_gui BT run.

Each detector lives in its OWN module here (collision.py, freezing.py, slip.py,
orientation.py, transition.py) as a faithful copy of that detector's standalone
``interactive.py`` real-time detection + live plot + VLM-confirm-and-pause,
adapted to be fed observations by the BT runner instead of driving its own demo.

This package's job is only orchestration: build the detectors the BT's hold
conditions request, feed each one per simulation step (gated per waypoint by the
BT), and summarize. All detection/plot/VLM behaviour lives in the per-detector
modules.
"""

from aha_publish import paths

# name -> submodule providing a LiveDetector class
_DETECTOR_MODULES = {
    "collision": "collision",
    "freezing": "freezing",
    "slip": "slip",
    "orientation": "orientation",
    "transition": "transition",
}

import os as _os

_ALL_LIVE_DETECTORS = (
    "collision", "freezing", "slip", "orientation", "transition",
)
# Optional allow-list to run only specific detectors, e.g.
# AHA_LIVE_DETECTORS=orientation (comma-separated). Empty -> all.
_env_sel = _os.getenv("AHA_LIVE_DETECTORS", "").strip()
if _env_sel:
    DEFAULT_LIVE_DETECTORS = tuple(
        n.strip() for n in _env_sel.split(",")
        if n.strip() in _ALL_LIVE_DETECTORS)
else:
    DEFAULT_LIVE_DETECTORS = _ALL_LIVE_DETECTORS

_DETECTOR_NAMES = set(_DETECTOR_MODULES)

# Detectors whose firing is taken as final — no VLM confirmation step. Only the
# GEOMETRIC detectors (orientation, transition) skip confirmation by default:
# they are pure pose signals where a VLM look-and-confirm adds latency and can
# falsely retract a real detection. Slip and collision ARE VLM-confirmed by
# default — their physical signal (grip force / torque) is noisier, so a visual
# check (taken a few frames after the trigger; see AHA_VLM_CONFIRM_DELAY_FRAMES)
# guards against false positives. Override with
# AHA_VLM_CONFIRM_DETECTORS=orientation,transition,slip,collision,... to choose
# exactly which detectors are confirmed (empty value -> none confirmed).
_skip_env = _os.getenv("AHA_VLM_CONFIRM_DETECTORS")
if _skip_env is None:
    NO_VLM_CONFIRM_DETECTORS = {"orientation", "transition"}
else:
    _confirm = {n.strip() for n in _skip_env.split(",") if n.strip()}
    NO_VLM_CONFIRM_DETECTORS = _DETECTOR_NAMES - _confirm

# Fallback mapping from a hold-condition predicate to its detector.
_HOLD_PREDICATE_TO_DETECTOR = {
    "no_collision": "collision",
    "not_frozen": "freezing",
    "maintains_grasp": "slip",
    "orientation_maintained": "orientation",
    "reaches_waypoint": "transition",
}


def detector_for_hold_block(block):
    """Resolve which runtime detector a BT hold-condition block maps to."""
    if not isinstance(block, dict):
        return None
    for key in ("detector", "failure"):
        val = str(block.get(key) or "").strip().lower()
        if val in _DETECTOR_NAMES:
            return val
    for link in block.get("failure_links") or []:
        if isinstance(link, dict):
            val = str(link.get("failure") or "").strip().lower()
            if val in _DETECTOR_NAMES:
                return val
    cond = str(block.get("condition") or block.get("original_condition") or "")
    for predicate, name in _HOLD_PREDICATE_TO_DETECTOR.items():
        if predicate in cond:
            return name
    return None


# Predicates whose FIRST argument names the object the stage manipulates. Ordered
# by how directly each one identifies that object, so the first match wins.
_TARGET_OBJECT_PREDICATES = (
    "object_in_gripper",
    "gripper_released",
    "selected_object",
    "object_for_press",
    "end_effector_aligned_with",
    "gripper_oriented_for",
    "closed",
    "inside",
    "on",
    "next_to",
)


def target_object_for_stage(stage):
    """The object a BT stage acts on, read off its own conditions.

    Contact with this object is INTENDED at this waypoint; the collision VLM
    verifier is told so, which is what separates a normal grasp/place from a
    collision with the table, a fixture, or some other object."""
    import re

    if not isinstance(stage, dict):
        return None
    texts = []
    for section in ("preconditions", "postconditions", "hold_conditions"):
        for block in stage.get(section) or []:
            if not isinstance(block, dict):
                continue
            text = block.get("condition") or block.get("original_condition") or ""
            # Object_found(a, b, ...) lists every scene object, not this stage's
            # target, so it would resolve to an arbitrary name.
            if text and not text.lower().startswith("object_found("):
                texts.append(text)
    for predicate in _TARGET_OBJECT_PREDICATES:
        for text in texts:
            match = re.search(rf"\b{predicate}\(\s*(\w+)", text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if name and name.lower() not in ("true", "false", "none"):
                    return name
    return None


def _bt_scene_objects(stages):
    """Every object name the BT mentions, incl. the Object_found(...) roster."""
    import re

    names = []
    for stage in stages or []:
        if not isinstance(stage, dict):
            continue
        for section in ("preconditions", "postconditions", "hold_conditions"):
            for block in stage.get(section) or []:
                if not isinstance(block, dict):
                    continue
                text = block.get("condition") or block.get("original_condition") or ""
                for match in re.finditer(r"\b\w+\(([^)]*)\)", text):
                    for arg in match.group(1).split(","):
                        arg = arg.strip()
                        if (arg and re.fullmatch(r"\w+", arg)
                                and arg.lower() not in ("true", "false", "none")
                                and arg not in names):
                            names.append(arg)
    return names


def _target_object_from_stage_name(stage, candidates):
    """Object a stage's NAME refers to ("approach above the remote" -> remote).

    Some BTs (press/open-door tasks) carry no object-bearing predicate at all —
    only gripper_condition — so the stage name is the only BT-side signal left.
    Longest candidate first, so 'microwave_door' wins over 'door'."""
    name = str((stage or {}).get("name") or "").lower()
    if not name:
        return None
    for candidate in sorted(candidates, key=len, reverse=True):
        needle = candidate.lower()
        if needle in name or needle.replace("_", " ") in name:
            return candidate
    return None


def target_object_map_for_stages(stages, n_waypoints, stage_lookup=None):
    """{waypoint index -> target object} for a whole BT.

    Per stage the object comes from that stage's own conditions; failing that,
    from its name. Stages that still name nothing (a bare approach, a lift, a
    retreat) inherit the manipulated object from the nearest stage that does:
    the previous one, or — for stages before the first grasp — the next one.
    That keeps the collision verifier from being told 'unknown' on exactly the
    approach waypoints where knowing the target matters most."""
    if stage_lookup is None:
        def stage_lookup(stages_, idx):
            return stages_[idx] if idx < len(stages_) else None

    candidates = _bt_scene_objects(stages)
    raw = []
    for idx in range(n_waypoints):
        stage = stage_lookup(stages, idx)
        name = target_object_for_stage(stage)
        if not name:
            name = _target_object_from_stage_name(stage, candidates)
        raw.append(name)

    filled = list(raw)
    last = None
    for i, name in enumerate(filled):
        if name:
            last = name
        elif last:
            filled[i] = last
    nxt = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i]:
            nxt = filled[i]
        elif nxt:
            filled[i] = nxt
    # Last resort: a BT whose stages are unnamed and predicate-free (only
    # gripper_condition) still names its object once, in Object_found. Use it
    # only when it is the BT's single object, so there is nothing to confuse.
    if not any(filled) and len(candidates) == 1:
        filled = [candidates[0]] * n_waypoints
    return {i: filled[i] for i in range(n_waypoints)}


def enabled_detectors_for_stage(stage):
    """Detectors the BT requests for one stage, from its (selected) hold blocks."""
    names = set()
    if isinstance(stage, dict):
        for block in stage.get("hold_conditions") or []:
            name = detector_for_hold_block(block)
            if name:
                names.add(name)
    return names


def _load_live_detector_class(name):
    import importlib
    module = importlib.import_module(f"{__name__}.{_DETECTOR_MODULES[name]}")
    return module.LiveDetector


def _reduce_obs(obs):
    """Serialize the LOW-DIM fields of an RLBench Observation to plain JSON so a
    step's detector inputs can be replayed offline. Images / point clouds (ndim>=3
    or large) are dropped -- no live detector reads them. Small arrays become
    lists; None/bool/number pass through. Replay reconstructs a stand-in obs from
    this dict, so it must contain every field any detector's obs_to_row reads."""
    import numpy as np

    out = {}
    for key, val in vars(obs).items():
        if val is None:
            out[key] = None
            continue
        if isinstance(val, (bool, int, float)):
            out[key] = val
            continue
        try:
            arr = np.asarray(val)
        except Exception:
            continue
        if arr.dtype == object or arr.ndim >= 3 or arr.size > 256:
            continue                      # image/pointcloud/misc -> skip
        out[key] = arr.astype(float).tolist()
    return out


def _pose_to_list(pose):
    if pose is None:
        return None
    try:
        import numpy as np
        return np.asarray(pose, dtype=float).tolist()
    except Exception:
        return None


def stage_context_for_stage(stage):
    """The BT stage's contract: its primitive plus its pre/postconditions.

    This is what the collision VLM verifier is told about the current step. It
    is deliberately the BT's structured predicates and NOT the description
    JSON's prose robot_action — narration like "the robot releases the ball"
    led the verifier to count "I cannot see a clean release" as evidence of
    collision."""
    if not isinstance(stage, dict):
        return None

    def _conditions(section):
        out = []
        for block in stage.get(section) or []:
            if not isinstance(block, dict):
                continue
            text = str(block.get("condition")
                       or block.get("original_condition") or "").strip()
            if text and text not in out:
                out.append(text)
        return out

    return {
        "primitive": str(stage.get("primitive") or "").strip(),
        "preconditions": _conditions("preconditions"),
        "postconditions": _conditions("postconditions"),
    }


def stage_context_map_for_stages(stages, n_waypoints, stage_lookup=None):
    """{waypoint index -> stage contract} for a whole BT."""
    if stage_lookup is None:
        def stage_lookup(stages_, idx):
            return stages_[idx] if idx < len(stages_) else None

    out = {}
    for idx in range(n_waypoints):
        context = stage_context_for_stage(stage_lookup(stages, idx))
        if context:
            out[idx] = context
    return out


class LiveDetectorMonitor:
    """Feeds every requested detector per step; each detector self-reports."""

    def __init__(self, task_name, *, waypoint_pose_fn=None, original_pose_fn=None,
                 enabled_map=None, target_object_map=None, stage_context_map=None,
                 env_wrapper=None, vlm_enabled=True, vlm_model=None,
                 vlm_trace=False, show_plots=True, n_waypoints=0,
                 failure_waypoint=None):
        self.task_name = task_name
        self.waypoint_pose_fn = waypoint_pose_fn
        self.original_pose_fn = original_pose_fn
        self.enabled_map = enabled_map
        self.target_object_map = target_object_map or {}
        self.stage_context_map = stage_context_map or {}
        self.detectors = {}     # name -> LiveDetector
        self.skipped = []       # (name, reason)
        self._obs_checked = False
        self._none_obs = 0
        self._pose_measurements = {}
        self._pose_measurements_total = {n: 0 for n in ('orientation', 'transition')}


        # Optional per-episode INPUT CAPTURE: when AHA_DETECTOR_CAPTURE points to
        # a file, every on_step's detector inputs (reduced obs, waypoint, poses,
        # phase, per-detector report gate) are streamed as JSONL. That stream can
        # later be replayed through the REAL detectors with new thresholds, so a
        # threshold change never needs a re-sim. See replay_captured.py.
        self._capture = None
        self._capture_i = 0
        self._shuffle_announced = False
        cap_path = _os.getenv("AHA_DETECTOR_CAPTURE", "").strip()
        if cap_path:
            try:
                self._capture = open(cap_path, "w")
                import json as _json
                self._capture.write(_json.dumps({
                    "_meta": True, "task": task_name,
                    "n_waypoints": n_waypoints,
                    "failure_waypoint": failure_waypoint,
                    "detectors": list(DEFAULT_LIVE_DETECTORS),
                }) + "\n")
            except Exception as exc:
                print(f"  [capture] disabled — {exc}")
                self._capture = None

        requested = None
        if enabled_map is not None:
            requested = set()
            for names in enabled_map.values():
                requested |= set(names)
            if not requested:  # older BT with no hold conditions -> run all
                requested = None
                self.enabled_map = None

        for name in DEFAULT_LIVE_DETECTORS:
            if requested is not None and name not in requested:
                self.skipped.append((name, "not requested by this BT"))
                continue
            try:
                cls = _load_live_detector_class(name)
            except Exception as exc:
                self.skipped.append((name, f"import failed: {exc}"))
                continue
            if getattr(cls, "NEEDS_WAYPOINT_POSE", False) and waypoint_pose_fn is None:
                self.skipped.append((name, "no waypoint-pose source available"))
                continue
            # Orientation/transition are geometric — skip their VLM confirmation.
            det_vlm = vlm_enabled and name not in NO_VLM_CONFIRM_DETECTORS
            try:
                det = cls(
                    task_name, env_wrapper=env_wrapper, vlm_enabled=det_vlm,
                    vlm_model=vlm_model, vlm_trace=vlm_trace, show_plot=show_plots,
                    n_waypoints=n_waypoints, failure_waypoint=failure_waypoint)
            except Exception as exc:
                self.skipped.append((name, f"init failed: {exc}"))
                continue
            # BT-supplied per-waypoint target object, for detectors whose VLM
            # verifier needs to know which contact is intended (collision).
            if hasattr(det, "bt_target_objects"):
                det.bt_target_objects = self.target_object_map
            # BT stage contract (primitive + pre/postconditions) for the same
            # verifier: what the stage is FOR, in predicates rather than prose.
            if hasattr(det, "bt_stage_contexts"):
                det.bt_stage_contexts = self.stage_context_map
            self.detectors[name] = det

    # ---- BT gating --------------------------------------------------------
    def _enabled_at(self, waypoint_index):
        if self.enabled_map is None:
            return None
        return self.enabled_map.get(waypoint_index, set())

    def _original_waypoint_at(self, waypoint_index):
        """Index of the waypoint that physically sits in this slot.

        Identity for every run except wrong_sequence_v2, which rewrites
        task._waypoints in place and records the slot -> original mapping in
        fail_sequence_v2._shuffle_permutation. The module global is set in the
        failure's on_start(), which runs after this monitor is built, so it is
        read per step rather than cached at construction.
        """
        perm = None
        try:
            from failgen import fail_sequence_v2
            perm = fail_sequence_v2._shuffle_permutation
        except Exception:
            perm = None
        if not perm:
            return waypoint_index
        mapped = perm.get(waypoint_index, waypoint_index)
        if mapped != waypoint_index and not self._shuffle_announced:
            self._shuffle_announced = True
            print(f"  [detectors] wrong_sequence_v2 shuffle active; "
                  f"orientation/transition follow it (slot -> waypoint: "
                  f"{ {k: v for k, v in sorted(perm.items())} })")
        return mapped

    def active_at(self, waypoint_index):
        names = {n for n, d in self.detectors.items() if d.disabled_reason is None}
        enabled = self._enabled_at(waypoint_index)
        if enabled is None:
            return names
        return names & set(enabled)

    def begin_waypoint(self, waypoint_index):
        """Re-arm geometric checks for each execution, including repetitions."""
        ref_index = self._original_waypoint_at(waypoint_index)
        for name in ('orientation', 'transition'):
            self._pose_measurements[(name, waypoint_index)] = 0
            det = self.detectors.get(name)
            if det is not None:
                det._cur_wp = None
                det._fired_this_execution.clear()
                det._vlm_rejected.discard(ref_index)

    def unevaluated_at(self, waypoint_index):
        return {name for name in self.active_at(waypoint_index)
                if name in ('orientation', 'transition')
                and not self._pose_measurements.get((name, waypoint_index), 0)}

    def fired_at(self, waypoint_index):
        ref_index = self._original_waypoint_at(waypoint_index)
        return {n for n, d in self.detectors.items()
                if (ref_index in d._fired_this_execution
                    if n in ('orientation', 'transition')
                    else waypoint_index in d.fired_waypoints)}

    def current_step(self):
        """Best-effort global frame index: the max step any live detector has
        been fed. Detectors are fed together each sim frame, so this is a
        consistent run-wide frame counter for the VLM event log."""
        best = None
        for d in self.detectors.values():
            for attr in ("step_counter", "fed", "step"):
                v = getattr(d, attr, None)
                if isinstance(v, int):
                    best = v if best is None else max(best, v)
                    break
        return best

    def flush_pending_vlm(self):
        """Resolve any detector's deferred VLM confirmation now. Called right
        before the BT consumes fired_at(), so a delayed verifier's verdict (and
        any retraction) is applied before the hold check reads fired_waypoints."""
        for det in self.detectors.values():
            fn = getattr(det, "flush_pending_vlm", None)
            if fn is None:
                continue
            try:
                fn()
            except Exception:
                pass

    def active_names(self):
        return list(self.detectors)

    def note_gripper_open_command(self):
        """Tell the slip detector that an open_gripper command was just issued,
        so it tags the next telemetry frame as the open-command frame."""
        det = self.detectors.get("slip")
        if det is not None and hasattr(det, "mark_open_command"):
            det.mark_open_command()

    def note_gripper_close_command(self):
        """Tell the slip detector that a close_gripper command was just issued
        (starts a grasp window / resets the held-peak baseline)."""
        det = self.detectors.get("slip")
        if det is not None and hasattr(det, "mark_close_command"):
            det.mark_close_command()

    # ---- reporting --------------------------------------------------------
    def print_banner(self):
        active = ", ".join(self.active_names()) or "(none)"
        print(f"\nLive detectors active: {active}")
        if self.enabled_map is not None:
            for name in self.active_names():
                wps = sorted(i for i, s in self.enabled_map.items() if name in s)
                print(f"  {name:<11} BT-checked at waypoints: {wps if wps else 'none'}")
        for name, reason in self.skipped:
            print(f"  [detector:{name}] skipped — {reason}")
        print()

    def _report_obs_fields(self, obs):
        import numpy as np

        def stat(field):
            value = getattr(obs, field, None)
            if value is None:
                return f"{field}=MISSING"
            try:
                arr = np.asarray(value, dtype=float)
                return f"{field}=ok(norm={float(np.linalg.norm(arr)):.3f})"
            except Exception:
                return f"{field}=present"

        fields = ("joint_forces", "gripper_touch_forces", "joint_velocities",
                  "gripper_pose", "gripper_joint_positions", "gripper_open")
        print("  [detectors] first obs fields: " + ", ".join(stat(f) for f in fields),
              flush=True)

    # ---- per-step ---------------------------------------------------------
    def on_step(self, obs, waypoint_index, path_done, phase="move"):
        if obs is None:
            self._none_obs += 1
            return
        if not self._obs_checked:
            self._obs_checked = True
            try:
                self._report_obs_fields(obs)
            except Exception:
                pass
        enabled = self._enabled_at(waypoint_index)
        # wrong_sequence_v2 scatters the waypoints across slots, so the waypoint
        # physically sitting in slot `waypoint_index` is the one that was
        # originally at perm[waypoint_index]. The geometric detectors judge a
        # motion against a per-waypoint reference pose AND per-waypoint clean
        # baselines, both keyed by the ORIGINAL index, so they have to follow the
        # shuffle -- otherwise every slot is scored against another waypoint's
        # target and the run fires at every waypoint.
        #
        # BT stage gating (`enabled`, above) deliberately stays on the slot: that
        # is the tree's own ordering, which the shuffle does not touch.
        ref_index = self._original_waypoint_at(waypoint_index)
        waypoint_pose = None
        if self.waypoint_pose_fn is not None:
            try:
                waypoint_pose = self.waypoint_pose_fn(waypoint_index)
            except Exception:
                waypoint_pose = None
        original_pose = ref_pose = None
        if self.original_pose_fn is not None:
            try:
                original_pose = self.original_pose_fn(waypoint_index)
            except Exception:
                original_pose = None
            if ref_index == waypoint_index:
                ref_pose = original_pose
            else:
                try:
                    ref_pose = self.original_pose_fn(ref_index)
                except Exception:
                    ref_pose = None
        freezing_already_reported = (
            "freezing" in self.detectors
            and waypoint_index in self.detectors["freezing"].fired_waypoints
        )
        # Per-detector report gate (BT gating + freezing pre-emption + gripper
        # phase). Computed once so the capture stream and the live feed agree.
        reports = {}
        for name in self.detectors:
            rep = enabled is None or name in enabled
            if freezing_already_reported and name != "freezing":
                rep = False
            if phase == "gripper" and name == "collision":
                rep = False
            reports[name] = rep

        if self._capture is not None:
            self._capture_step(obs, waypoint_index, path_done, phase,
                               waypoint_pose, original_pose, reports)

        for name, det in self.detectors.items():
            report = reports[name]
            extra = {}
            geometric = name in {"orientation", "transition"}
            if geometric and report and phase == 'move':
                import numpy as np
                target = np.asarray(ref_pose, dtype=float).reshape(-1)
                actual = np.asarray(getattr(obs, 'gripper_pose', None), dtype=float).reshape(-1)
                if (target.size >= 7 and actual.size >= 7
                        and np.isfinite(target[:7]).all() and np.isfinite(actual[:7]).all()):
                    key = (name, waypoint_index)
                    self._pose_measurements[key] = self._pose_measurements.get(key, 0) + 1
                    self._pose_measurements_total[name] += 1
            if getattr(det, "NEEDS_ORIGINAL_POSE", False):
                # Only the geometric pair follows the shuffle. collision and slip
                # also take original_pose but key nothing per-waypoint off it, so
                # they keep the slot and their behaviour is unchanged.
                extra["original_pose"] = ref_pose if geometric else original_pose
            if geometric:
                extra["phase"] = phase
            try:
                det.step(obs,
                         waypoint=ref_index if geometric else waypoint_index,
                         path_done=path_done,
                         waypoint_pose=waypoint_pose, report=report, **extra)
            except Exception as exc:
                if det.disabled_reason is None:
                    det.disabled_reason = str(exc)
                    print(f"  [detector:{name}] disabled mid-run — {exc}")

    def print_summary(self):
        # Resolve any still-deferred VLM confirmation so the summary reflects the
        # final verdict (normally every waypoint's hold check already flushed).
        self.flush_pending_vlm()
        print("\n" + "=" * 70)
        print("  Live detector summary")
        print("=" * 70)
        if not self._obs_checked:
            print("  WARNING: no valid observations were ever received "
                  f"({self._none_obs} None obs) — detectors had nothing to read.")
        for name, det in self.detectors.items():
            if det.disabled_reason is not None:
                print(f"  {name:<11} DISABLED — {det.disabled_reason}")
            elif name in self._pose_measurements_total and not self._pose_measurements_total[name]:
                print(f"  {name:<11} NOT EVALUATED — no valid target measurements")
            elif det.fire_count == 0:
                print(f"  {name:<11} ok — no {name} detected (checked {det.checked} obs)")
            else:
                step, reason, wp = det.first_fire
                print(f"  {name:<11} DETECTED {name} x{det.fire_count} "
                      f"(first: step {step}, waypoint {wp}, {reason})")
            try:
                print(det.diagnostics_line())
            except Exception:
                pass
        for name, reason in self.skipped:
            print(f"  {name:<11} skipped — {reason}")
        print("=" * 70 + "\n")

    def _capture_step(self, obs, waypoint_index, path_done, phase,
                      waypoint_pose, original_pose, reports):
        import json as _json
        try:
            rec = {
                "i": self._capture_i,
                "wp": waypoint_index,
                "path_done": bool(path_done),
                "phase": phase,
                "wp_pose": _pose_to_list(waypoint_pose),
                "orig_pose": _pose_to_list(original_pose),
                "reports": reports,
                "obs": _reduce_obs(obs),
            }
            self._capture.write(_json.dumps(rec) + "\n")
        except Exception as exc:
            print(f"  [capture] step {self._capture_i} skipped — {exc}")
        self._capture_i += 1

    def close(self):
        if self._capture is not None:
            try:
                self._capture.close()
            except Exception:
                pass
            self._capture = None
        for det in self.detectors.values():
            try:
                det.close()
            except Exception:
                pass
