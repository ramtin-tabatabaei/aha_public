"""Real behavior tree (py_trees) for the interactive BT condition runner.

This module turns the per-waypoint episode into an explicit `py_trees` behavior
tree. It owns no simulation, VLM, or detector logic of its own — every heavy
operation is delegated back to callables on the runner module
(`waypoints_interactive_bt_conditions`), which is handed in via `BTContext.runner`.
That keeps this module import-light (only `py_trees` + stdlib at the top level)
and avoids a duplicate import of the runner when it is executed as `__main__`.

Tree shape (per waypoint `i`):

    episode = Sequence(memory)[ wp[0], wp[1], ... ]
    wp[i]   = Sequence(memory)[
                  PreCondition(i),
                  act[i] = Parallel(SuccessOnAll)[
                      motion[i] = Sequence(memory)[ MovePath(i), GripperAction(i) ],
                      HoldMonitor(i),
                  ],
                  PostCondition(i),
              ]

Two run modes (selected via `--mode`, read from `ctx.args.mode`):
  * manual — at each checkpoint the runner asks whether to verify with the VLM;
    answering "no" accepts the conditions as satisfied (today's `--vlm-checks ask`).
  * auto   — pre/post verified by the VLM and holds by the detectors, no prompts.

Leaf actions run to completion inside a single `update()` (they never return
RUNNING). This is deliberate: the original procedural loop always ran a
waypoint's path to completion and surfaced a detector firing *after* the motion,
so a fired hold becomes a tree FAILURE at `HoldMonitor` rather than preempting
mid-path. On any FAILURE the root Sequence stops at that waypoint (detect →
report → explain → stop); no recovery branches.
"""

from __future__ import annotations

from aha_publish import paths

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import py_trees
from py_trees.common import Status


# Extra waypoints (beyond the injection waypoint's own post[N]+pre[N+1] boundary)
# that a slip run is allowed to keep running so the VLM boundary checks get more
# chances to catch the slip before the deferred detector fire finalizes the run.
# 1 -> the VLM also gets to check pre[N+2]. Override with the env var.
def _slip_detect_extra_waypoints() -> int:
    try:
        return max(0, int(os.environ.get('AHA_SLIP_DETECT_EXTRA_WAYPOINTS', '1')))
    except (TypeError, ValueError):
        return 1


# Waypoints AFTER the injection waypoint N that still get their VLM pre/post
# condition checks under --limited / --limited-v2. 1 -> the check window is
# exactly {N, N+1}: the failure is looked for at N (pre[N] and the post[N] +
# pre[N+1] boundary) and, if it was not caught there, once more at N+1; from N+2
# on the run is photo-only. Same rule for slip, collision, transition and
# orientation.
LIMITED_EXTRA_WAYPOINTS = 1


# ---------------------------------------------------------------------------
# Shared run context (blackboard)
# ---------------------------------------------------------------------------

@dataclass
class BTContext:
    """Everything the tree nodes need, mirroring the locals main() builds."""

    runner: Any                      # the runner module (its functions are reused)
    args: Any
    scene: Any
    env_wrapper: Any
    robot: Any
    waypoints: Any
    n: int
    bt_stages: Any
    bt_source_name: str
    task_name: str
    chosen_cam: str
    side_camera: Any
    side_camera_window: Any
    live_camera: Any
    monitor: Any
    yaml_descs: Any
    depth_units: str
    vlm_enabled: bool
    sequence_predicates: Any
    failure_active: bool
    failure_waypoint: Optional[int]
    report_wrist: bool
    visualization_mode: str
    report_wrist_values: Callable
    # detector that owns the injected failure; when set, only its confirmed fire
    # aborts the run (other detectors' fires are recorded but not fatal).
    responsible_detector: Optional[str] = None
    # mutable run state
    motion_buffer: Any = None
    # per-step failure-waypoint frame dump target + cameras (set by bt_run_move,
    # reused by bt_run_gripper); None on clean waypoints or when dumping is off.
    frame_dump_dir: Any = None
    frame_dump_cams: Any = None
    boundary_cache: dict = field(default_factory=dict)
    failures: list = field(default_factory=list)
    deferred_hold_fires: dict = field(default_factory=dict)
    # Chronological transcript of every VLM condition check run this episode, fed
    # to the post-failure failure-type diagnosis (see run_failure_diagnosis).
    check_history: list = field(default_factory=list)
    # Filled by run_failure_diagnosis: {'with_detectors': {...}, 'no_detectors': {...}}.
    diagnosis: Optional[dict] = None
    # Remaining "extra" waypoints a slip run may keep running after the injection
    # waypoint so the VLM boundary checks get another chance to catch the slip
    # before the deferred detector fire is finalized. None -> not yet initialized
    # (lazily seeded from AHA_SLIP_DETECT_EXTRA_WAYPOINTS on first deferred fire).
    detect_grace_remaining: Optional[int] = None
    # Latched True the moment the failure is first recorded (under any no-abort
    # mode). Under --limited-v2 it also turns the rest of the run photo-only: no
    # more VLM condition checks and the live detectors muted (see
    # _detectors_muted in the runner and the failure_detected gate in
    # _limited_skip_waypoint). Under --no-abort the checks keep running.
    failure_detected: bool = False


@dataclass
class FailureReport:
    waypoint: int
    phase: str                       # 'precondition' | 'postcondition' | 'hold'
    node: str
    failed_conditions: list
    failure_modes: list
    vlm_explanation: Any = None
    vlm_trace_path: Any = None


# ---------------------------------------------------------------------------
# Verdict mapping (three-valued VLM result -> BT Status)
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return ''.join((text or '').split()).lower().replace('==true', '=true').replace('==false', '=false')


def _failure_modes_for(blocks, failed_conditions):
    """Failure-mode names linked to the conditions that did not hold."""
    failed_keys = {_norm(c) for c in failed_conditions}
    modes = []
    for block in blocks or []:
        cond = block.get('condition') or block.get('original_condition') or ''
        if failed_keys and _norm(cond) not in failed_keys:
            continue
        for link in block.get('failure_links', []) or []:
            failure = link.get('failure')
            if failure and failure not in modes:
                modes.append(failure)
    return modes


def _record_failure(ctx, result, phase, i, blocks, node_name, fired=None):
    if fired:
        failed_conditions = sorted(fired)
        failure_modes = sorted(fired)
        explanation = None
        trace = None
    else:
        not_ok = [
            item.get('condition', '')
            for item in (result or {}).get('conditions', [])
            if item.get('status') == 'not_satisfied'
        ]
        failed_conditions = not_ok or [
            (block.get('condition') or block.get('original_condition') or '')
            for block in (blocks or [])
        ]
        failure_modes = _failure_modes_for(blocks, failed_conditions)
        explanation = (result or {}).get('explanation')
        trace = (result or {}).get('trace_path')
    ctx.failures.append(FailureReport(
        waypoint=i,
        phase=phase,
        node=node_name,
        failed_conditions=failed_conditions,
        failure_modes=failure_modes,
        vlm_explanation=explanation,
        vlm_trace_path=trace,
    ))


def _record_history(ctx, phase, i, result):
    """Append one VLM condition-check result to ``ctx.check_history`` so the
    end-of-run failure diagnosis can reason over the whole task's checks. No-op
    when nothing was checked (``result is None``)."""
    if result is None:
        return
    conditions = [
        {
            'condition': item.get('condition', ''),
            'status': item.get('status', ''),
            'evidence': item.get('evidence', ''),
        }
        for item in (result.get('conditions', []) or [])
    ]
    ctx.check_history.append({
        'waypoint': i,
        'phase': phase,
        'checkpoint_passed': result.get('checkpoint_passed'),
        'conditions': conditions,
        'explanation': result.get('explanation', ''),
    })


def _no_abort(ctx) -> bool:
    """True under --limited-v2 and --no-abort: a failed VLM check or a fired
    detector is still recorded, but the tree must NOT stop — every waypoint runs
    to the end (only the wall-clock --timeout ends the episode) so the dumped
    frames cover the whole task from the first waypoint to the last.

    The two differ in what happens AFTER the first failure: --limited-v2 goes
    photo-only (see _photo_only_after_detect), while --no-abort keeps running
    every VLM condition check and every live detector to the last waypoint."""
    return bool(getattr(ctx.args, 'limited_v2', False)
                or getattr(ctx.args, 'no_abort', False))


def _photo_only_after_detect(ctx) -> bool:
    """True under --limited-v2 only: once the failure is recorded the rest of the
    run takes pictures but runs no VLM condition check (and the runner mutes the
    live detectors). --no-abort deliberately keeps checking."""
    return bool(getattr(ctx.args, 'limited_v2', False))


def _fail_or_continue(ctx):
    """Status to return once a failure has been recorded: SUCCESS to keep the
    episode running to the end under --limited-v2 / --no-abort, FAILURE to stop at
    this waypoint otherwise. It also latches ctx.failure_detected, which under
    --limited-v2 turns the rest of the run photo-only — no more VLM condition
    checks and the live detectors muted (see _detectors_muted in the runner)."""
    if _no_abort(ctx):
        ctx.failure_detected = True
        return Status.SUCCESS
    return Status.FAILURE


def verdict_to_status(ctx, result, phase, i, blocks, node_name, strict_override=None):
    """Map a VLM checkpoint result to a BT Status.

    `result is None` means no check ran (disabled / declined in manual mode / no
    conditions) -> the checkpoint is accepted as satisfied (SUCCESS).

    `strict_override` lets a specific checkpoint force strict handling of an
    uncertain verdict (treat it as a FAILURE) independent of the global --strict
    flag. The arrival/orientation check uses this so orientation defaults to
    strict while alignment/position stay lenient. None -> use the global flag.
    """
    if result is None:
        return Status.SUCCESS
    passed = result.get('checkpoint_passed')
    if passed is True:
        return Status.SUCCESS
    if passed is False:
        _record_failure(ctx, result, phase, i, blocks, node_name)
        return _fail_or_continue(ctx)
    # passed is None -> the VLM was uncertain.
    strict = (
        getattr(ctx.args, 'strict', False)
        if strict_override is None else strict_override
    )
    if strict:
        _record_failure(ctx, result, phase, i, blocks, node_name)
        return _fail_or_continue(ctx)
    if ctx.args.mode == 'manual':
        ok = ctx.runner.ask_yes_no(
            f"  [uncertain] VLM could not decide the {phase} for waypoint {i}; "
            f"treat it as satisfied?",
            default=True,
        )
        if ok:
            return Status.SUCCESS
        _record_failure(ctx, result, phase, i, blocks, node_name)
        return _fail_or_continue(ctx)
    # auto mode, not strict
    print(
        f"  [uncertain] VLM could not decide the {phase} for waypoint {i}; "
        f"treating as satisfied (use --strict to fail)."
    )
    return Status.SUCCESS


# ---------------------------------------------------------------------------
# Behaviors (leaves)
# ---------------------------------------------------------------------------

def _limited_skip_waypoint(ctx, i) -> bool:
    """True when --limited says to skip waypoint ``i``'s VLM pre/post condition
    checks.

    Under --limited, a detector-owned failure injected at waypoint N is only
    checked inside the window N .. N+LIMITED_EXTRA_WAYPOINTS (i.e. N and N+1):
    pre[N] and post[N] (the combined post[N]+pre[N+1] boundary) get the first
    look, and if nothing is caught there the run gets one more look at waypoint
    N+1. Every waypoint BEFORE N is skipped (the failure has not been injected
    yet) and every waypoint AFTER the window is skipped too (the failure
    manifests as the arm executes N, so a check three waypoints later is a wasted
    VLM call). Runs with no responsible detector -- clean, wrong_sequence,
    wrong_object -- are never limited: they rely on the pre/post checks to catch
    the failure, so every waypoint is checked.

    Exception: freezing is fully owned by the freezing detector, so its VLM
    pre/post condition checks add nothing. Under --limited a freezing run skips
    those checks at EVERY waypoint (not just those before N).

    --limited-v2 behaves exactly like --limited UNTIL the failure is detected
    (the runner turns --limited on for it). From the moment ctx.failure_detected
    latches, every remaining checkpoint is skipped (photo-only, no VLM) so the run
    just moves to the end taking pictures. --no-abort is NOT limited at all: it
    checks every waypoint, before and after the failure."""
    if _photo_only_after_detect(ctx) and getattr(ctx, 'failure_detected', False):
        return True
    if not getattr(ctx.args, 'limited', False):
        return False
    if not getattr(ctx, 'failure_active', False):
        return False
    responsible = getattr(ctx, 'responsible_detector', None)
    if not responsible:
        return False
    if responsible == 'freezing':
        return True
    n = getattr(ctx, 'failure_waypoint', None)
    if n is None or n < 0:
        return False
    return i < n or i > n + LIMITED_EXTRA_WAYPOINTS


def _limited_skip_reason(ctx, i) -> str:
    """Human-readable reason for a --limited VLM-check skip (for the run log)."""
    if getattr(ctx, 'responsible_detector', None) == 'freezing':
        return "freezing is detector-owned; VLM pre/post checks off under --limited"
    if _photo_only_after_detect(ctx) and getattr(ctx, 'failure_detected', False):
        return "failure already detected; photo-only for the rest of the run"
    n = getattr(ctx, 'failure_waypoint', None)
    if n is not None and i is not None and i > n:
        return (f"past the check window {n}..{n + LIMITED_EXTRA_WAYPOINTS} "
                f"for injection waypoint {n}")
    return f"before injection waypoint {ctx.failure_waypoint}"


class _BTLeaf(py_trees.behaviour.Behaviour):
    def __init__(self, name, ctx, i, wp, stage):
        super().__init__(name=name)
        self.ctx = ctx
        self.i = i
        self.wp = wp
        self.stage = stage


class PreCondition(_BTLeaf):
    """Verify the waypoint's preconditions (or reuse the prior boundary verdict)."""

    def update(self):
        ctx, i, stage = self.ctx, self.i, self.stage
        runner = ctx.runner
        # If this pre was already evaluated as part of the previous waypoint's
        # combined boundary check, reuse that verdict — its header/preconditions
        # were printed there too (mirrors the old `move_already_confirmed`).
        cached = ctx.boundary_cache.pop(i, None)
        if cached is not None:
            # That boundary check also photographed pre[i], unless this waypoint
            # is now outside the --limited check window (or the failure latched
            # right after the boundary check) — then take the photo here so the
            # pre/post image coverage stays complete to the last waypoint.
            if _limited_skip_waypoint(ctx, i):
                runner.capture_condition_frames(
                    scene=ctx.scene,
                    side_camera=ctx.side_camera,
                    env_wrapper=ctx.env_wrapper,
                    task_name=ctx.task_name,
                    waypoint_index=i,
                    checkpoint_kind='pre',
                    conditions=runner.condition_strings(stage['preconditions']),
                )
            return cached
        runner.print_waypoint_header(self.wp, i, ctx.n - 1, stage, ctx.yaml_descs)
        runner.print_preconditions(stage, i)
        if _limited_skip_waypoint(ctx, i):
            print(
                f"  [limited] skipping precondition VLM check for waypoint {i} "
                f"({_limited_skip_reason(ctx, i)})"
            )
            # VLM check is skipped, but still photograph the precondition so every
            # pre/post condition has an image from the first waypoint to the last.
            runner.capture_condition_frames(
                scene=ctx.scene,
                side_camera=ctx.side_camera,
                env_wrapper=ctx.env_wrapper,
                task_name=ctx.task_name,
                waypoint_index=i,
                checkpoint_kind='pre',
                conditions=runner.condition_strings(stage['preconditions']),
            )
            return Status.SUCCESS
        result = runner.maybe_run_vlm_condition_check(
            args=ctx.args,
            enabled=ctx.vlm_enabled,
            checkpoint_kind='pre',
            scene=ctx.scene,
            side_camera=ctx.side_camera,
            env_wrapper=ctx.env_wrapper,
            task_name=ctx.task_name,
            waypoint_index=i,
            stage=stage,
            conditions=runner.condition_strings(stage['preconditions']),
        )
        _record_history(ctx, 'precondition', i, result)
        return verdict_to_status(
            ctx, result, 'precondition', i, stage['preconditions'], self.name
        )


class MovePath(_BTLeaf):
    """Run the waypoint's path to completion (verbatim port)."""

    def update(self):
        ctx = self.ctx
        # Stamp the global frame at which the injected waypoint's motion begins,
        # so the VLM event log can compare injection vs. detection frame.
        if (ctx.failure_active and ctx.failure_waypoint == self.i
                and ctx.monitor is not None):
            try:
                import aha_publish.running.vlm_run_logger as vlm_run_logger
                lg = vlm_run_logger.active()
                if lg is not None:
                    lg.set_injection_frame(ctx.monitor.current_step())
            except Exception:
                pass
        ctx.runner.bt_run_move(ctx, self.i, self.wp, self.stage)
        return Status.SUCCESS


class GripperAction(_BTLeaf):
    """Run the waypoint's gripper open/close action, or no-op when absent."""

    def update(self):
        self.ctx.runner.bt_run_gripper(self.ctx, self.i, self.wp)
        return Status.SUCCESS


class HoldMonitor(_BTLeaf):
    """Record fired live detectors, deferring fatal stop until after post/pre."""

    def update(self):
        ctx, i, stage = self.ctx, self.i, self.stage
        monitor = ctx.monitor
        if monitor is not None:
            # Force any delayed VLM confirmation to resolve before we read the
            # fired set, so a deferred verifier's retraction lands in time.
            monitor.flush_pending_vlm()
            ctx.runner.print_hold_results(stage, i, monitor)
        fired = monitor.fired_at(i) if monitor is not None else set()
        if fired:
            # Clean run (no failure injected): a fire here is a false positive.
            # Do NOT abort -- let the BT run every remaining waypoint to the end
            # so the clean trace and plots cover the whole task instead of
            # stopping short. The run log still contains the detector fire.
            if not ctx.failure_active:
                return Status.SUCCESS
            responsible = getattr(ctx, 'responsible_detector', None)
            # No live detector owns this failure type (wrong_sequence /
            # wrong_object are meant to be caught by the VLM pre/post sequence
            # checks, not a detector). Record the cross-fire but never abort, so
            # the task runs to completion and every pre/post checkpoint gets its
            # chance to catch the failure.
            if not responsible:
                return Status.SUCCESS
            # Stop only on the responsible detector for the injected failure
            # (e.g. a slip-injection run stops on slip, a collision run on
            # collision). A cross-fire from any other detector must NOT abort, so
            # the run reaches the injected waypoint and the responsible detector
            # gets its chance to catch the real failure.
            if responsible not in fired:
                return Status.SUCCESS
            # slip/collision fire BEFORE the injection waypoint: the injected
            # failure has not been applied yet (it is injected at
            # failure_waypoint), so a confirmed slip/collision here is a
            # pre-injection false positive. Record it (it stays in the detector's
            # fired_waypoints and the run log) but do NOT abort — let the run
            # reach the injection waypoint where the real failure manifests.
            fw = getattr(ctx, 'failure_waypoint', None)
            if (responsible in ('slip', 'collision')
                    and fw is not None and fw >= 0 and i < fw):
                print(
                    f"  [{responsible}] fired+confirmed at waypoint {i}, before "
                    f"the injection waypoint {fw}; recorded as a pre-injection "
                    f"false positive, not aborting."
                )
                return Status.SUCCESS
            # Do not fail the waypoint here. Let PostCondition run the normal
            # post[i] + pre[i+1] boundary VLM check, then stop before waypoint
            # i+1 executes.
            ctx.deferred_hold_fires[i] = set(fired)
            return Status.SUCCESS
        return Status.SUCCESS


class ArrivalCondition(_BTLeaf):
    """Verify arrival-checked predicates at the pose the gripper actually reached.

    Runs between MovePath and GripperAction — after the move reaches the (possibly
    failure-perturbed) waypoint pose but before the gripper acts. This is the moment
    a predicate like gripper_oriented_for is genuinely judgeable: the gripper is at
    the grasp pose, so a wrong orientation is visible (unlike the approach boundary,
    where the gripper is still hovering above the object). Predicates are routed here
    via --vlm-arrival-predicates and removed from the pre/post checkpoints. A no-op
    (SUCCESS) when no arrival predicates are configured."""

    def update(self):
        ctx, i, stage = self.ctx, self.i, self.stage
        if self.wp.skip:
            return Status.SUCCESS
        arrival = getattr(ctx.args, 'vlm_arrival_predicate_filter', None)
        if not arrival:
            return Status.SUCCESS
        runner = ctx.runner
        if _limited_skip_waypoint(ctx, i):
            # Photo-only: capture the arrival checkpoint even though its VLM check
            # is skipped, so the pre/post image coverage is complete.
            runner.capture_condition_frames(
                scene=ctx.scene,
                side_camera=ctx.side_camera,
                env_wrapper=ctx.env_wrapper,
                task_name=ctx.task_name,
                waypoint_index=i,
                checkpoint_kind='arrival',
                conditions=runner.condition_strings(stage['preconditions']),
            )
            return Status.SUCCESS
        result = runner.maybe_run_vlm_condition_check(
            args=ctx.args,
            enabled=ctx.vlm_enabled,
            checkpoint_kind='arrival',
            scene=ctx.scene,
            side_camera=ctx.side_camera,
            env_wrapper=ctx.env_wrapper,
            task_name=ctx.task_name,
            waypoint_index=i,
            stage=stage,
            conditions=runner.condition_strings(stage['preconditions']),
            predicate_allow=arrival,
            predicate_deny=None,
        )
        _record_history(ctx, 'arrival', i, result)
        # Orientation (the arrival-checked predicate) is strict by default: an
        # uncertain verdict fails the checkpoint even without the global --strict,
        # because the VLM rarely commits to not_satisfied on a tilt. Toggle with
        # --no-vlm-arrival-strict.
        return verdict_to_status(
            ctx, result, 'arrival', i, stage['preconditions'], self.name,
            strict_override=getattr(ctx.args, 'vlm_arrival_strict', True),
        )


class PostCondition(_BTLeaf):
    """Verify postconditions; for non-final waypoints this is the combined
    post[i] + pre[i+1] boundary check (one VLM call), and the pre[i+1] verdict is
    cached so PreCondition(i+1) does not re-check."""

    def update(self):
        ctx, i, stage = self.ctx, self.i, self.stage
        runner = ctx.runner
        n = ctx.n

        def maybe_stop_for_deferred_hold(status):
            if status != Status.SUCCESS:
                # The VLM boundary check already flagged the failure at this
                # waypoint -> stop and let that verdict stand.
                return status
            fired = ctx.deferred_hold_fires.pop(i, None)
            if not fired:
                return Status.SUCCESS
            # The responsible detector fired but the VLM boundary check here
            # (post[i] + pre[i+1]) did NOT catch the failure. For slip, give the
            # VLM another opportunity: carry the deferred fire to i+1 so
            # PostCondition(i+1) also runs its post[i+1] + pre[i+2] boundary
            # check. Only slip, only while grace budget remains, and only when a
            # next waypoint exists.
            if ctx.detect_grace_remaining is None:
                ctx.detect_grace_remaining = (
                    _slip_detect_extra_waypoints()
                    if getattr(ctx, 'responsible_detector', None) == 'slip'
                    else 0
                )
            # Only worth carrying if waypoint i+1 still gets a VLM boundary
            # check; past the --limited window it would be a silent deferral.
            if (ctx.detect_grace_remaining > 0 and (i + 1) < n
                    and getattr(ctx, 'responsible_detector', None) == 'slip'
                    and not _limited_skip_waypoint(ctx, i + 1)):
                ctx.detect_grace_remaining -= 1
                ctx.deferred_hold_fires[i + 1] = fired
                print(
                    f"  [slip] not caught at waypoint {i} boundary (pre[{i + 1}]); "
                    f"giving the VLM another chance at waypoint {i + 1} "
                    f"(pre[{i + 2}])"
                )
                return Status.SUCCESS
            _record_failure(
                ctx, result=None, phase='hold', i=i,
                blocks=stage.get('hold_conditions', []),
                node_name=f"hold[{i}]", fired=fired,
            )
            return _fail_or_continue(ctx)

        runner.print_postconditions(stage, i, n - 1)
        if _limited_skip_waypoint(ctx, i):
            # Outside the check window: skip post[i] (and the post[i]+pre[i+1]
            # boundary) VLM check. Deliberately do NOT cache pre[i+1] -- when the
            # next waypoint is the injection one, PreCondition(i+1) must run its own
            # pre-check rather than reuse a skipped boundary verdict. The next
            # waypoint header is left for PreCondition(i+1) to print.
            print(
                f"  [limited] skipping postcondition VLM check for waypoint {i} "
                f"({_limited_skip_reason(ctx, i)})"
            )
            # VLM check is skipped, but still photograph the postcondition so every
            # pre/post condition has an image from the first waypoint to the last.
            runner.capture_condition_frames(
                scene=ctx.scene,
                side_camera=ctx.side_camera,
                env_wrapper=ctx.env_wrapper,
                task_name=ctx.task_name,
                waypoint_index=i,
                checkpoint_kind='post',
                conditions=runner.condition_strings(stage['postconditions']),
            )
            return maybe_stop_for_deferred_hold(Status.SUCCESS)
        if i < n - 1:
            next_stage = runner.stage_for_waypoint(ctx.bt_stages, i + 1)
            runner.print_waypoint_header(
                ctx.waypoints[i + 1], i + 1, n - 1, next_stage, ctx.yaml_descs
            )
            runner.print_preconditions(next_stage, i + 1)
            result = runner.maybe_run_vlm_boundary_check(
                args=ctx.args,
                enabled=ctx.vlm_enabled,
                scene=ctx.scene,
                side_camera=ctx.side_camera,
                env_wrapper=ctx.env_wrapper,
                task_name=ctx.task_name,
                waypoint_index=i,
                stage=stage,
                next_waypoint_index=i + 1,
                next_stage=next_stage,
                motion_buffer=ctx.motion_buffer,
                sequence_predicates=ctx.sequence_predicates,
            )
            _record_history(ctx, f'postcondition+pre[{i + 1}]', i, result)
            status = verdict_to_status(
                ctx, result, 'postcondition', i,
                stage['postconditions'] + next_stage['preconditions'], self.name,
            )
            if status == Status.SUCCESS:
                # The combined check covered pre[i+1]; let PreCondition(i+1) reuse it.
                ctx.boundary_cache[i + 1] = Status.SUCCESS
            return maybe_stop_for_deferred_hold(status)
        # Final waypoint: post-only check.
        result = runner.maybe_run_vlm_condition_check(
            args=ctx.args,
            enabled=ctx.vlm_enabled,
            checkpoint_kind='post',
            scene=ctx.scene,
            side_camera=ctx.side_camera,
            env_wrapper=ctx.env_wrapper,
            task_name=ctx.task_name,
            waypoint_index=i,
            stage=stage,
            conditions=runner.condition_strings(stage['postconditions']),
            motion_buffer=ctx.motion_buffer,
            sequence_predicates=ctx.sequence_predicates,
        )
        _record_history(ctx, 'postcondition', i, result)
        status = verdict_to_status(
            ctx, result, 'postcondition', i, stage['postconditions'], self.name
        )
        return maybe_stop_for_deferred_hold(status)


# ---------------------------------------------------------------------------
# Tree builder + driver
# ---------------------------------------------------------------------------

def build_tree(ctx):
    Sequence = py_trees.composites.Sequence
    Parallel = py_trees.composites.Parallel
    policy = py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)

    wp_subtrees = []
    for i, wp in enumerate(ctx.waypoints):
        stage = ctx.runner.stage_for_waypoint(ctx.bt_stages, i)
        motion = Sequence(name=f"motion[{i}]", memory=True, children=[
            MovePath(f"move[{i}]", ctx, i, wp, stage),
            ArrivalCondition(f"arrive[{i}]", ctx, i, wp, stage),
            GripperAction(f"grip[{i}]", ctx, i, wp, stage),
        ])
        action = Parallel(name=f"act[{i}]", policy=policy, children=[
            motion,
            HoldMonitor(f"hold[{i}]", ctx, i, wp, stage),
        ])
        wp_subtrees.append(Sequence(name=f"wp[{i}]", memory=True, children=[
            PreCondition(f"pre[{i}]", ctx, i, wp, stage),
            action,
            PostCondition(f"post[{i}]", ctx, i, wp, stage),
        ]))
    return Sequence(name="episode", memory=True, children=wp_subtrees)


def _print_failure_summary(ctx):
    print("\n" + "=" * 70)
    if not ctx.failures:
        print("BT VERDICT: all checkpoints satisfied — no failures detected.")
        print("=" * 70)
        return
    if _no_abort(ctx):
        mode = '--limited-v2' if _photo_only_after_detect(ctx) else '--no-abort'
        print(f"BT VERDICT: {len(ctx.failures)} checkpoint(s) flagged a failure — "
              f"recorded; run continued to the end ({mode}, no abort).")
    else:
        print(f"BT VERDICT: {len(ctx.failures)} failure(s) detected — stopped at the first.")
    for report in ctx.failures:
        print("-" * 70)
        print(f"  Waypoint {report.waypoint}  [{report.phase}]  (node {report.node})")
        if report.failed_conditions:
            print("  Failed conditions:")
            for cond in report.failed_conditions:
                print(f"    - {cond}")
        if report.failure_modes:
            print(f"  Linked failure mode(s): {', '.join(report.failure_modes)}")
        if report.vlm_explanation:
            print(f"  VLM explanation: {report.vlm_explanation}")
        if report.vlm_trace_path:
            print(f"  VLM trace: {report.vlm_trace_path}")
    print("=" * 70)


def run_tree(ctx):
    """Build, display, and tick the behavior tree to completion. Returns the
    root Status (SUCCESS = all waypoints passed, FAILURE = stopped early)."""
    root = build_tree(ctx)
    print("\nBehavior tree:")
    print(py_trees.display.unicode_tree(root))

    # Leaves run to completion, so a single tick cascades the whole episode; the
    # loop is a safety net in case a node ever returns RUNNING.
    root.tick_once()
    while root.status == Status.RUNNING:
        root.tick_once()

    # Honour the same task callback as RLBench's demo loop. Each pass prepares
    # fresh targets through start_of_path; reference sampling and detector state
    # are reset at those boundaries, never from a failure-perturbed target.
    # OFF by default: the 8 repeating tasks (stack_blocks, place_cups, ...) run
    # the same waypoint indices once per item, but an injected failure is
    # one-shot (fail_collision latches IDLE->APPLIED), so every pass after the
    # first is a clean run whose detector fires can only be false positives.
    # --repeat-waypoints restores the full multi-item episode.
    allow_repeats = bool(getattr(ctx.args, 'repeat_waypoints', False))
    repeats = 0
    while root.status == Status.SUCCESS:
        repeat = ctx.scene.task.should_repeat_waypoints()
        if not repeat or ctx.scene.task.success()[0]:
            break
        if not allow_repeats:
            print("[waypoints] task asks to repeat the waypoint cycle; "
                  "stopping after the first pass "
                  "(pass --repeat-waypoints to run every repetition)")
            break
        repeats += 1
        if repeats >= 100:
            raise RuntimeError('task requested more than 100 waypoint repetitions')
        print(f"[waypoints] starting repetition {repeats + 1}")
        ctx.boundary_cache.clear()
        ctx.deferred_hold_fires.clear()
        root = build_tree(ctx)
        root.tick_once()
        while root.status == Status.RUNNING:
            root.tick_once()

    print("\n" + "=" * 70)
    print(f"BEHAVIOR TREE RESULT: {root.status.name}")
    print("=" * 70)
    print(py_trees.display.unicode_tree(root, show_status=True))
    _print_failure_summary(ctx)
    # After a run-ending failure, classify what kind of failure it was (twice:
    # with live-detector/grasp evidence and blind). Delegated to the runner, which
    # owns all VLM/LLM access.
    if (ctx.failures and getattr(ctx, 'vlm_enabled', False)
            and getattr(ctx.args, 'vlm_diagnose_failure', False)):
        try:
            ctx.runner.run_failure_diagnosis(ctx)
        except Exception as exc:
            print(f"  [diagnosis] failed: {exc}")
    return root.status
