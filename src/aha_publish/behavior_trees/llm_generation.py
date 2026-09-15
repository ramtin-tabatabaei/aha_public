"""LLM client calls, response normalization, and generation orchestration."""

from aha_publish import paths

import json
import os
import re
import sys
from pathlib import Path

# Import directly from the source modules. This module used to receive these
# names transitively via `from .local_generation import *`, but that chain funneled
# through prompt_building's `from .cleanup import *`; once that became a selective
# import the names stopped propagating, so we depend on the real sources here.
from .io_context import *          # also re-exports prompts + config names
from .semantics import *
from .conditions import *
from .cleanup import *
from .failure_definitions import *
from .local_generation import *

# --- token-usage accounting -------------------------------------------------
# The streaming calls below used to discard the API-reported usage. We now record
# it per agent so a full BT build reports exactly what it cost (input vs output
# tokens, and a per-waypoint breakdown). Reset at the start of generate_conditions.
BT_TOKEN_USAGE: list[dict] = []


def _reset_token_usage() -> None:
    BT_TOKEN_USAGE.clear()


def _record_usage(agent: str, model: str, usage) -> None:
    """Append one {agent, model, input_tokens, output_tokens} record. `usage` is
    the Anthropic/OpenAI usage object; missing fields are stored as None."""
    if usage is None:
        return
    BT_TOKEN_USAGE.append({
        "agent": agent,
        "model": model,
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    })


def format_bt_token_usage(n_waypoints: int | None = None) -> str:
    """Human-readable summary of the tokens spent in the current BT build."""
    if not BT_TOKEN_USAGE:
        return "[BT token usage: none recorded]"
    lines = ["[BT token usage]"]
    tot_in = tot_out = 0
    for rec in BT_TOKEN_USAGE:
        i = rec.get("input_tokens") or 0
        o = rec.get("output_tokens") or 0
        tot_in += i
        tot_out += o
        lines.append(f"  {rec['agent']:<8s} ({rec['model']}): input={i}  output={o}")
    lines.append(f"  TOTAL: input={tot_in}  output={tot_out}")
    if n_waypoints:
        lines.append(
            f"  per-waypoint ({n_waypoints} wps): "
            f"input={tot_in / n_waypoints:.0f}  output={tot_out / n_waypoints:.0f}"
        )
    return "\n".join(lines)


def build_client(provider: str):
    if provider == "claude":
        if importlib.util.find_spec("anthropic") is None:
            raise RuntimeError(
                f"Missing Python package 'anthropic'. Install it with: "
                f"{sys.executable} -m pip install anthropic"
            )
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY in your environment.")
        return anthropic.Anthropic(api_key=api_key)
    if provider == "openai":
        missing = [
            package
            for package in OPENAI_REQUIRED_PACKAGES
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            packages = " ".join(missing)
            raise RuntimeError(
                f"Missing Python package(s): {packages}. Install with: "
                f"{sys.executable} -m pip install {packages}"
            )
        try:
            import openai
        except ModuleNotFoundError as e:
            package = e.name or "the missing OpenAI dependency"
            raise RuntimeError(
                f"Missing Python package '{package}'. Install with: "
                f"{sys.executable} -m pip install {package}"
            ) from e
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY.")
        os.environ["OPENAI_API_KEY"] = api_key
        return openai.OpenAI(api_key=api_key)
    raise ValueError(f"Unknown provider '{provider}'. Choose 'claude' or 'openai'.")

def generate_conditions_with_claude(client, user_message: str, system_prompt: str | None = None) -> dict:
    if system_prompt is None:
        system_prompt = get_system_prompt()
    full_text = ""
    print("[Agent 1: generating BT conditions...]\n", flush=True)
    with client.messages.stream(
        model=GENERATOR_CLAUDE_MODEL,
        max_tokens=CONDITION_MAX_OUTPUT_TOKENS,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text
        _record_usage("Agent1", GENERATOR_CLAUDE_MODEL, stream.get_final_message().usage)
    print("\n", flush=True)
    return parse_json_response(full_text)

def generate_conditions_with_openai(client, user_message: str, system_prompt: str | None = None) -> dict:
    if system_prompt is None:
        system_prompt = get_system_prompt()
    full_text = ""
    print("[Agent 1: generating BT conditions...]\n", flush=True)
    stream = client.responses.create(
        model=GENERATOR_OPENAI_MODEL,
        max_output_tokens=CONDITION_MAX_OUTPUT_TOKENS,
        stream=True,
        text={
            "format": {
                "type": "json_schema",
                "name": "bt_conditions",
                "schema": CONDITION_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    for event in stream:
        if hasattr(event, "type") and event.type == "response.output_text.delta":
            chunk = getattr(event, "delta", "") or ""
            print(chunk, end="", flush=True)
            full_text += chunk
        elif getattr(event, "type", "") == "response.completed":
            _record_usage("Agent1", GENERATOR_OPENAI_MODEL, getattr(event.response, "usage", None))
    print("\n", flush=True)
    return parse_json_response(full_text)

def _reviewer_enabled(reviewer: bool | None) -> bool:
    """Resolve whether Agent 2 runs: explicit flag wins, else BT_USE_REVIEWER env
    (default ON). Set BT_USE_REVIEWER=0 to disable globally."""
    if reviewer is not None:
        return reviewer
    return os.environ.get("BT_USE_REVIEWER", "1").strip().lower() not in ("0", "false", "no", "")


def build_reviewer_message(draft_stages: list, task_context: str | dict) -> str:
    """Build Agent 2's user message: a compact per-waypoint gripper/held summary
    (the authoritative ground truth) plus Agent 1's draft stages to audit."""
    try:
        ctx = json.loads(task_context) if isinstance(task_context, str) else (task_context or {})
    except Exception:
        ctx = {}
    gripper_lines = []
    for wp in (ctx.get("waypoints") or []):
        gripper_lines.append(
            f"  waypoint {wp.get('waypoint')}: gripper_state={wp.get('gripper_state')!r}, "
            f"held_object={wp.get('held_object')!r}"
        )
    objects = sorted({
        obj.get("name")
        for obj in (ctx.get("key_scene_objects") or [])
        if obj.get("name")
    })
    return (
        "Authoritative per-waypoint gripper / held-object ground truth:\n"
        + ("\n".join(gripper_lines) if gripper_lines else "  (none provided)")
        + "\n\nTask scene objects (only these names are valid):\n  "
        + (", ".join(objects) if objects else "(none listed)")
        + "\n\nAgent 1 draft stages to audit and correct:\n"
        + json.dumps({"stages": draft_stages}, indent=2)
    )


def review_conditions_with_claude(client, user_message: str) -> dict:
    full_text = ""
    print("[Agent 2: reviewing BT conditions...]\n", flush=True)
    with client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=CONDITION_MAX_OUTPUT_TOKENS,
        system=REVIEWER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_text += text
        _record_usage("Agent2", CLAUDE_MODEL, stream.get_final_message().usage)
    print("\n", flush=True)
    return parse_json_response(full_text)


def review_conditions_with_openai(client, user_message: str) -> dict:
    full_text = ""
    print("[Agent 2: reviewing BT conditions...]\n", flush=True)
    stream = client.responses.create(
        model=OPENAI_MODEL,
        max_output_tokens=CONDITION_MAX_OUTPUT_TOKENS,
        stream=True,
        text={
            "format": {
                "type": "json_schema",
                "name": "bt_review",
                "schema": REVIEWER_RESPONSE_SCHEMA,
                "strict": True,
            }
        },
        input=[
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    for event in stream:
        if hasattr(event, "type") and event.type == "response.output_text.delta":
            chunk = getattr(event, "delta", "") or ""
            print(chunk, end="", flush=True)
            full_text += chunk
        elif getattr(event, "type", "") == "response.completed":
            _record_usage("Agent2", OPENAI_MODEL, getattr(event.response, "usage", None))
    print("\n", flush=True)
    return parse_json_response(full_text)


def _condition_block(text: str, failure: str, reason: str) -> dict:
    return {
        "condition": text,
        "alternatives": [],
        "failure_links": [{"failure": failure, "reason": reason}],
    }


def _predicate_head(template: str | None) -> str | None:
    """'object_in_gripper(<object>) == True' -> 'object_in_gripper';
    'gripper_condition = Open' -> 'gripper_condition'."""
    if not template:
        return None
    return template.split("(")[0].split("=")[0].strip() or None


def _instantiate_predicate(template: str, obj: str) -> str:
    """Bind the first <placeholder> in a catalogue template to a real object name."""
    return re.sub(r"<[^>]+>", obj, template, count=1)


def _deterministic_vocab() -> dict:
    """Resolve every predicate the deterministic pass touches from the catalogue and
    the gripper-state predicate list — no predicate name is hard-coded here, so a
    rename in generated_condition_rules.json is picked up automatically."""
    from . import prompts

    rules = prompts.load_condition_rules()
    held_tmpl, held_fail = prompts.predicate_for_placement("grasp", "postcondition", rules)
    rel_tmpl, rel_fail = prompts.predicate_for_placement("release", "postcondition", rules)

    grippers = list(prompts.gripper_state_predicates())
    gripper_head = _predicate_head(grippers[0]) if grippers else None
    open_pred = next((g for g in grippers if "open" in g.lower()), None)
    closed_pred = next((g for g in grippers if "closed" in g.lower()), None)
    open_state = open_pred.split("=")[-1].strip() if open_pred else "Open"
    closed_state = closed_pred.split("=")[-1].strip() if closed_pred else "Closed"

    return {
        "held_head": _predicate_head(held_tmpl),
        "held_tmpl": held_tmpl,
        "held_fail": held_fail,
        "rel_head": _predicate_head(rel_tmpl),
        "rel_tmpl": rel_tmpl,
        "rel_fail": rel_fail,
        "gripper_head": gripper_head,
        "open_state": open_state,
        "closed_state": closed_state,
        "gripper_re": (
            re.compile(rf"{re.escape(gripper_head)}\s*=\s*(\w+)") if gripper_head else None
        ),
    }


def _effective_gripper_state(conditions, vocab: dict) -> str | None:
    """The open/closed gripper state a stage ends in, inferred from its conditions:
    the release predicate -> open state, the held predicate -> closed state, else the
    gripper_condition value. All predicate identities come from `vocab`."""
    fallback = None
    for c in conditions or []:
        text = c.get("condition") or ""
        if vocab["rel_head"] and vocab["rel_head"] in text:
            return vocab["open_state"]
        if vocab["held_head"] and vocab["held_head"] in text:
            fallback = vocab["closed_state"]
            continue
        if vocab["gripper_re"]:
            match = vocab["gripper_re"].search(text)
            if match and fallback is None:
                fallback = match.group(1)
    return fallback


def _infer_held_object(stages: list, index: int, task_context, held_head: str | None) -> str | None:
    """Object held going into `index`, from the nearest preceding held-predicate
    condition, falling back to the task context's held_object."""
    if held_head:
        pattern = re.compile(rf"{re.escape(held_head)}\((\w+)\)")
        for j in range(index - 1, -1, -1):
            for c in (stages[j].get("postconditions") or []) + (stages[j].get("preconditions") or []):
                m = pattern.search(c.get("condition") or "")
                if m:
                    return m.group(1)
    try:
        ctx = json.loads(task_context) if isinstance(task_context, str) else (task_context or {})
        by_wp = {w.get("waypoint"): w for w in ctx.get("waypoints", [])}
        for k in (index - 1, index):
            held = (by_wp.get(k) or {}).get("held_object")
            if held:
                return held
    except Exception:
        pass
    return None


def apply_deterministic_review_fixes(stages: list, task_context) -> tuple[list, list]:
    """Deterministic normalization the LLM reviewer cannot be relied on to perform.
    Every predicate it reads or writes is resolved from the catalogue (see
    `_deterministic_vocab`), so no predicate name is hard-coded.

    (1) A release waypoint must hold the object at its start and be released at its
        end. When the reviewer reclassifies a place->release and strips the now
        invalid held-object conditions, it can leave the waypoint empty (as happened
        to change_channel stage 4); refill it here.
    (2) Gripper continuity: each waypoint's gripper_condition precondition must equal
        the previous waypoint's end-of-waypoint gripper state.
    """
    fixes: list[dict] = []
    try:
        vocab = _deterministic_vocab()
    except Exception:
        return stages, fixes

    # (1) Repair release waypoints, using the catalogue's held/release predicates.
    if vocab["held_tmpl"] and vocab["rel_tmpl"]:
        for i, stage in enumerate(stages):
            if (stage.get("primitive") or "").lower() != "release":
                continue
            pre = stage.get("preconditions") or []
            post = stage.get("postconditions") or []
            obj = _infer_held_object(stages, i, task_context, vocab["held_head"])
            if not obj:
                continue
            held_cond = _instantiate_predicate(vocab["held_tmpl"], obj)
            rel_cond = _instantiate_predicate(vocab["rel_tmpl"], obj)
            if not any(vocab["held_head"] in (c.get("condition") or "") for c in pre):
                pre = [
                    _condition_block(
                        held_cond,
                        vocab["held_fail"] or "ExecutionSequenceMismatch",
                        "The object must be held at the start of the release.",
                    )
                ] + pre
                fixes.append({
                    "stage": stage.get("stage", i), "rule": "DET-release",
                    "detail": "Release waypoint had no holding precondition.",
                    "fix": f"Added {held_cond} precondition.",
                })
            if not any(vocab["rel_head"] in (c.get("condition") or "") for c in post):
                post = post + [
                    _condition_block(
                        rel_cond,
                        vocab["rel_fail"] or "ExecutionSequenceMismatch",
                        "Confirms the object was released by this waypoint.",
                    )
                ]
                fixes.append({
                    "stage": stage.get("stage", i), "rule": "DET-release",
                    "detail": "Release waypoint had no release postcondition.",
                    "fix": f"Added {rel_cond} postcondition.",
                })
            stage["preconditions"] = pre
            stage["postconditions"] = post

    # (2) Gripper precondition continuity: pre(i) = end state of (i-1).
    if vocab["gripper_re"]:
        for i in range(1, len(stages)):
            prev_state = _effective_gripper_state(stages[i - 1].get("postconditions"), vocab)
            if prev_state is None:
                continue
            for c in stages[i].get("preconditions") or []:
                match = vocab["gripper_re"].search(c.get("condition") or "")
                if match and match.group(1) != prev_state:
                    old = c["condition"]
                    c["condition"] = f"{vocab['gripper_head']} = {prev_state}"
                    fixes.append({
                        "stage": stages[i].get("stage", i), "rule": "DET-continuity",
                        "detail": (
                            f"Precondition {old!r} did not match the previous waypoint's "
                            f"end gripper state ({prev_state})."
                        ),
                        "fix": f"Set precondition to {vocab['gripper_head']} = {prev_state}.",
                    })
    return stages, fixes


def _deterministic_fixes_enabled() -> bool:
    """Whether the deterministic post-review pass runs. Default OFF — Agent 2 (the
    LLM reviewer) is instructed to refill the held-object precondition and the
    release postcondition itself whenever it reclassifies a waypoint to release
    (see rules R3/R5 in the reviewer prompt). Set BT_DETERMINISTIC_FIXES=1 to also
    run the deterministic safety net as a redundant backstop."""
    return os.environ.get("BT_DETERMINISTIC_FIXES", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def review_conditions(draft_stages: list, task_context, client, provider: str) -> dict:
    """Run Agent 2 over Agent 1's raw stages. Returns {"violations": [...],
    "stages": [...corrected...]}. On any failure, returns the draft unchanged with
    an empty violation list so generation never hard-fails on the reviewer.

    The deterministic post-review pass is OFF by default (BT_DETERMINISTIC_FIXES=1
    to re-enable); Agent 2 is the sole fixer otherwise."""
    if not draft_stages:
        return {"violations": [], "stages": draft_stages}
    user_message = build_reviewer_message(draft_stages, task_context)
    if provider == "claude":
        reviewed = review_conditions_with_claude(client, user_message)
    elif provider == "openai":
        reviewed = review_conditions_with_openai(client, user_message)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose 'claude' or 'openai'.")
    stages = reviewed.get("stages") or draft_stages
    det_fixes: list[dict] = []
    if _deterministic_fixes_enabled():
        stages, det_fixes = apply_deterministic_review_fixes(stages, task_context)
    violations = list(reviewed.get("violations", [])) + det_fixes
    return {"violations": violations, "stages": stages}


def _print_review_report(violations: list) -> None:
    """Human-readable summary of Agent 2's findings."""
    if not violations:
        print("[Agent 2: no structural violations found]\n", flush=True)
        return
    print(f"[Agent 2: {len(violations)} violation(s) found and fixed]", flush=True)
    for v in violations:
        print(
            f"  - stage {v.get('stage')} [{v.get('rule')}]: {v.get('detail')} "
            f"-> {v.get('fix')}",
            flush=True,
        )
    print("", flush=True)


def _normalize_primitive(value) -> str | None:
    """Return the Step-1 waypoint primitive if it is one of the six allowed names."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v if v in WAYPOINT_PRIMITIVES else None


def normalize_generated_conditions(result: dict, task_context: str | dict | None = None) -> dict:
    stage_contexts, _scene_roles = parse_task_context_for_semantics(task_context)
    normalized_stages = []
    for index, stage in enumerate(result.get("stages", [])):
        stage_number = stage.get("stage", stage.get("stage_number", index))
        stage_for_rules = stage_with_context(
            {"stage": stage_number, **stage},
            stage_contexts,
        )
        preconditions = merge_condition_blocks(expand_and_conditions(stage.get("preconditions", [])))
        postconditions = merge_condition_blocks(expand_and_conditions(stage.get("postconditions", [])))
        normalized_stages.append(
            {
                "stage": stage_number,
                "name": stage.get("name") or stage.get("summary") or f"Stage {index}",
                "primitive": _normalize_primitive(stage.get("primitive")),
                "preconditions": filter_stage_condition_blocks(stage_for_rules, preconditions, "preconditions"),
                "postconditions": filter_stage_condition_blocks(stage_for_rules, postconditions, "postconditions"),
            }
        )
    normalized_stages = apply_condition_cleanup_pipeline(normalized_stages)
    attach_hold_conditions(normalized_stages)
    return {"source": "api_generation", "stages": normalized_stages}

def normalize_saved_condition(saved_condition: dict) -> dict:
    normalized = normalize_condition(saved_condition)
    normalized["selected"] = saved_condition.get("selected", True)
    normalized["original_condition"] = saved_condition.get(
        "original_condition",
        normalized["condition"],
    )
    return normalized

def saved_review_to_conditions(saved_data: dict) -> dict | None:
    review = saved_data.get("review") or saved_data.get("generated") or {}
    saved_stages = review.get("stages") or []
    if not saved_stages:
        return None

    stages = []
    hold_by_stage: dict = {}
    primitive_by_stage: dict = {}
    for index, stage in enumerate(saved_stages):
        stage_number = stage.get("stage", index)
        preconditions = merge_condition_blocks(
            [
                normalize_saved_condition(condition)
                for condition in stage.get("preconditions", [])
            ]
        )
        postconditions = merge_condition_blocks(
            [
                normalize_saved_condition(condition)
                for condition in stage.get("postconditions", [])
            ]
        )
        hold_by_stage[stage_number] = hold_conditions_for_saved_stage(stage)
        primitive_by_stage[stage_number] = _normalize_primitive(stage.get("primitive"))
        stages.append(
            {
                "stage": stage_number,
                "name": stage.get("name") or f"Stage {stage_number}",
                "summary": stage.get("summary", ""),
                "missing_description": stage.get("missing_description", ""),
                "preconditions": filter_stage_condition_blocks(stage, preconditions, "preconditions"),
                "postconditions": filter_stage_condition_blocks(stage, postconditions, "postconditions"),
            }
        )

    stages = apply_condition_cleanup_pipeline(stages)
    stages = fix_press_stage_conditions(stages)
    # Reattach the curated hold conditions and the Step-1 primitive after the
    # pipeline (which may rebuild stage dicts), keyed by stage number, so user
    # removals and the primitive label persist on reload.
    for index, stage in enumerate(stages):
        stage["hold_conditions"] = hold_by_stage.get(
            stage.get("stage", index), build_hold_condition_blocks()
        )
        stage["primitive"] = primitive_by_stage.get(stage.get("stage", index))
    return {
        "source": "saved_review",
        "saved_at": saved_data.get("saved_at"),
        "stages": stages,
    }

def _press_target_from_stage(stage: dict) -> str | None:
    """Return the target chosen by the agent via object_for_press(...)."""
    for section in ("postconditions", "preconditions"):
        for c in stage.get(section, []) or []:
            m = re.search(r"\bobject_for_press\((\w+)\)", c.get("condition", ""), re.IGNORECASE)
            if m:
                return m.group(1)
    return None


def _is_gripper_condition(cond: dict) -> bool:
    return bool(re.search(r"gripper_condition\s*=\s*(?:Open|Closed)",
                          cond.get("condition", ""), re.IGNORECASE))


def _press_contact_waypoint_indices_from_agent(stages: list[dict]) -> set[int]:
    """Waypoint indices the agent explicitly marked with object_for_press(...)."""
    return {
        stage.get("stage")
        for stage in stages
        if stage.get("stage") is not None
        and any(
            re.search(r"\bobject_for_press\(", c.get("condition", ""), re.IGNORECASE)
            for section in ("preconditions", "postconditions")
            for c in (stage.get(section) or [])
        )
    }


def fix_press_stage_conditions(stages: list[dict]) -> list[dict]:
    """Normalize press-contact waypoints chosen by the agent.

    The LLM decides which waypoints are press-contact waypoints by emitting
    object_for_press(<target>). This pass only cleans up those chosen waypoints:
    no object_in_gripper, no gripper_condition beside it, and stable
    carry-forward of object_for_press across adjacent press-contact waypoints.
    """
    press_wps = _press_contact_waypoint_indices_from_agent(stages)
    if not press_wps:
        return stages

    # A single task-level fallback target (some later press frames carry no object name).
    fallback_target = None
    for stage in stages:
        if stage.get("stage") in press_wps:
            fallback_target = _press_target_from_stage(stage)
            if fallback_target:
                break

    def _ofp_block(target: str):
        return normalize_condition({
            "condition": f"object_for_press({target}) == True",
            "failure_links": [{
                "failure": "ExecutionSequenceMismatch",
                "reason": "Gripper closes in pressing contact with the target without grasping it.",
            }],
        })

    for stage in stages:
        idx = stage.get("stage")
        if idx not in press_wps:
            continue
        prev_is_press = (idx - 1) in press_wps
        target = _press_target_from_stage(stage) or fallback_target
        if not target:
            continue

        # Postconditions: convert object_in_gripper -> object_for_press, ensure the
        # predicate is present, and drop any redundant gripper_condition.
        post = stage.get("postconditions", []) or []
        for c in post:
            c["condition"] = re.sub(r"\bobject_in_gripper\((\w+)\)",
                                    r"object_for_press(\1)", c.get("condition", ""))
            if target:
                c["condition"] = re.sub(
                    r"\bobject_for_press\((\w+)\)",
                    f"object_for_press({target})",
                    c["condition"],
                )
        has_ofp = any(re.search(r"\bobject_for_press\(", c.get("condition", "")) for c in post)
        if not has_ofp:
            post = [c for c in post if not _is_gripper_condition(c)]
            post.append(_ofp_block(target))
        else:
            post = [c for c in post if not _is_gripper_condition(c)]
        stage["postconditions"] = merge_condition_blocks(post)

        # Preconditions: drop any carried object_in_gripper. When the previous
        # waypoint was also press-contact, the gripper is already pressing on entry,
        # so gripper_condition = Closed becomes object_for_press(<target>).
        pre = [c for c in (stage.get("preconditions", []) or [])
               if not re.search(r"\bobject_in_gripper\(", c.get("condition", ""))]
        if prev_is_press:
            new_pre, replaced = [], False
            for c in pre:
                if _is_gripper_condition(c):
                    if not replaced:
                        new_pre.append(_ofp_block(target))
                        replaced = True
                else:
                    new_pre.append(c)
            pre = new_pre
        elif any(re.search(r"\bobject_for_press\(", c.get("condition", ""), re.IGNORECASE) for c in pre):
            pre = [c for c in pre if not _is_gripper_condition(c)]
        for c in pre:
            c["condition"] = re.sub(
                r"\bobject_for_press\((\w+)\)",
                f"object_for_press({target})",
                c.get("condition", ""),
            )
        stage["preconditions"] = merge_condition_blocks(pre)
    return stages


def _authoritative_gripper_end_states(task_context):
    """{waypoint_index: ('open'|'closed'|'holding', object_or_None)} from the
    description's stamped gripper fields, else None."""
    try:
        obj = json.loads(task_context) if isinstance(task_context, str) else task_context
    except Exception:
        return None
    if not has_stamped_gripper_fields(obj):
        return None
    states = {}
    for wp in obj.get("waypoints") or []:
        idx = wp.get("waypoint")
        if idx is None:
            continue
        gs = str(wp.get("gripper_state", "")).lower()
        ho = wp.get("held_object")
        if ho:
            states[idx] = ("holding", ho)
        elif any(token in gs for token in ("closed", "closing", "holding")):
            states[idx] = ("closed", None)
        else:
            states[idx] = ("open", None)
    return states or None


def _gripper_state_condition_text(state) -> str:
    kind, obj = state
    if kind == "holding":
        return f"object_in_gripper({obj}) == True"
    if kind == "closed":
        return "gripper_condition = Closed"
    return "gripper_condition = Open"


def _is_gripper_state_condition(cond) -> bool:
    t = condition_text(cond)
    return has_gripper_state_text(t) or bool(re.match(r"\s*object_in_gripper\s*\(", t))


_ALIGNMENT_PREDICATE_RE = re.compile(
    r"(end_effector_aligned_with|gripper_oriented_for)\s*\(", re.IGNORECASE)


def strip_alignment_when_holding(stages: list[dict]) -> list[dict]:
    """Remove pre-grasp alignment/orientation preconditions from any waypoint
    whose gripper precondition is already Closed or holding an object — you
    cannot be "aligning to grasp" a target the gripper is already clamped on.
    Fallback used only when the description has no simulator-ground-truth gripper
    sequence; otherwise place_grasp_alignment handles relocation deterministically.
    """
    for stage in stages:
        pre = stage.get("preconditions") or []
        holding = any(
            ("object_in_gripper(" in condition_text(c).lower())
            or re.search(r"gripper_condition\s*=\s*closed", condition_text(c), re.IGNORECASE)
            for c in pre
        )
        if holding:
            stage["preconditions"] = [
                c for c in pre
                if not _ALIGNMENT_PREDICATE_RE.search(condition_text(c))
            ]
    return stages


def _stage_pre_post_states(stages, states):
    """Replicate enforce_gripper_states' pre/post mapping: post(pos)=end-state(wp),
    pre(pos)=end-state(prev wp), pre(pos 0)=Open. Returns (pre_states, post_states)."""
    pre_states, post_states = [], []
    for pos, stage in enumerate(stages):
        wp_idx = stage.get("stage", pos)
        post_states.append(states.get(wp_idx))
        if pos == 0:
            pre_states.append(("open", None))
        else:
            prev_idx = stages[pos - 1].get("stage", pos - 1)
            pre_states.append(states.get(prev_idx))
    return pre_states, post_states


def _insert_alignment_blocks(pre, blocks):
    """Insert alignment/orientation blocks before the first gripper-state
    precondition (so the order reads selected_object, align, orient, gripper)."""
    if not blocks:
        return pre
    for i, c in enumerate(pre):
        if _is_gripper_state_condition(c):
            return pre[:i] + blocks + pre[i:]
    return pre + blocks


_GRASP_ROLE_TERMS = ("grasp", "pick", "lift", "carry", "move", "held", "reloc",
                     "transport", "relocat")
# An object is NOT the grasped object if its OWN NAME marks it as fixture-like, or
# its role marks it a distractor/receptacle. Match against the name (not the whole
# role) so a grasped object whose role merely mentions a receptacle ("moved into
# the hoop") is not wrongly excluded.
_NON_GRASP_NAME_TERMS = ("gripper", "robot", "manipulator", "hoop", "hook",
                         "stand", "holder", "table", "wall",
                         "receptacle", "rack")
_NON_GRASP_ROLE_TERMS = ("distractor", "receptacle")


def _scene_object_roles(task_context):
    """{scene_object_name: role_text_lower} from key_scene_objects, else {}."""
    try:
        d = json.loads(task_context) if isinstance(task_context, str) else task_context
    except Exception:
        return {}
    if not isinstance(d, dict):
        return {}
    roles = {}
    for o in (d.get("key_scene_objects") or []):
        if isinstance(o, dict) and o.get("name"):
            roles[o["name"]] = str(o.get("role") or o.get("description") or "").lower()
    return roles


def _canonical_grasped_name(held_handle, selected_name, scene_roles):
    """Choose one canonical name for the grasped object.

    The simulator's held_object is ground truth for WHICH instance is grasped; we
    prefer a descriptive scene-description name for that instance:
      - if the held handle is itself a scene-object name, trust it (this also
        overrides a mis-selected instance — e.g. open_jar resolves to jar_lid0
        even though the description labels jar_lid1 the 'target');
      - else if the LLM's selected_object name is a scene-object name, use it
        (e.g. handle 'ball' -> scene name 'basketball');
      - else if exactly one scene object reads as the grasped/manipulated object,
        use that; otherwise fall back to the raw handle.
    """
    if held_handle in scene_roles:
        return held_handle
    if selected_name and selected_name in scene_roles:
        return selected_name
    cands = [
        n for n, role in scene_roles.items()
        if any(t in role for t in _GRASP_ROLE_TERMS)
        and not any(x in n.lower() for x in _NON_GRASP_NAME_TERMS)
        and not any(x in role for x in _NON_GRASP_ROLE_TERMS)
    ]
    return cands[0] if len(cands) == 1 else held_handle


def _rename_tokens(text, mapping):
    for old, new in mapping.items():
        text = re.sub(r"\b" + re.escape(old) + r"\b", new, text)
    return text


def _dedupe_object_found_args(text):
    def repl(m):
        seen, kept = set(), []
        for a in (x.strip() for x in m.group(1).split(",")):
            if a and a not in seen:
                seen.add(a)
                kept.append(a)
        return "Object_found(" + ", ".join(kept) + ")"
    return re.sub(r"Object_found\(([^)]*)\)", repl, text)


def _selected_object_args(stages):
    out = []
    for st in stages:
        for c in (st.get("preconditions") or []):
            m = re.search(r"selected_object\((\w+)\)", condition_text(c))
            if m:
                out.append(m.group(1))
    return out


def normalize_grasped_object_identity(stages: list[dict], task_context) -> list[dict]:
    """Force one canonical name per grasped object across the whole BT.

    The LLM names the grasped object from the task description, while
    enforce_gripper_states / place_grasp_alignment name it from the simulator's
    held_object handle. When those disagree (e.g. selected_object(basketball) but
    object_in_gripper(ball), or a mis-selected instance like jar_lid1 vs the
    actually-held jar_lid0) the BT is internally inconsistent. This rewrites every
    reference to the grasped object — selected_object, object_in_gripper,
    gripper_released, end_effector_aligned_with, gripper_oriented_for,
    on/inside/next_to, Object_found, and stage name/summary text — to one
    canonical name. Run AFTER place_grasp_alignment so freshly added alignment
    blocks are normalized too.
    """
    states = _authoritative_gripper_end_states(task_context)
    if not states:
        return stages
    held = []
    seen = set()
    for _, (kind, obj) in sorted(states.items()):
        if kind == "holding" and obj not in seen:
            seen.add(obj)
            held.append(obj)
    if not held:
        return stages
    scene_roles = _scene_object_roles(task_context)
    sel_args = _selected_object_args(stages)

    rename = {}
    if len(held) == 1:
        # Common single-grasp case: reconcile the one held handle with the one
        # selected_object name so a synonym/instance mismatch is fixed.
        H = held[0]
        S = sel_args[0] if len(sel_args) == 1 else None
        C = _canonical_grasped_name(H, S, scene_roles)
        for alias in {H, S} - {None, C}:
            rename[alias] = C
    else:
        # Multi-grasp: canonicalize each handle alone — do not risk mis-pairing
        # selected_object names across distinct grasped objects.
        for H in held:
            C = _canonical_grasped_name(H, None, scene_roles)
            if C != H:
                rename[H] = C

    if not rename:
        return stages
    for st in stages:
        for key in ("name", "summary"):
            if isinstance(st.get(key), str):
                st[key] = _rename_tokens(st[key], rename)
        for section in ("preconditions", "postconditions", "hold_conditions"):
            for c in (st.get(section) or []):
                if isinstance(c, dict) and "condition" in c:
                    c["condition"] = _dedupe_object_found_args(
                        _rename_tokens(c["condition"], rename))
    return stages


def normalize_scene_object_names(stages: list[dict], task_context) -> list[dict]:
    """Rewrite raw simulator handles to the descriptive key_scene_objects name.

    The BT must speak the description's ``name`` vocabulary (basketball, hoop),
    not the simulator's ``original_name`` handles (ball,
    basket_ball_hoop_respondable), since the conditions are checked visually.
    The system prompt asks for this; this pass enforces it deterministically on
    whatever the LLM emitted.

    Only predicate arguments are rewritten. Stage name/summary and the
    failure_links reasons are free text, where a handle like "lid", "target" or
    "stand" is usually an ordinary English word, so rewriting them would garble
    the prose.
    """
    rename = scene_name_aliases(task_context)
    if not rename:
        return stages
    for st in stages:
        if not isinstance(st, dict):
            continue
        for section in ("preconditions", "postconditions", "hold_conditions"):
            for c in (st.get(section) or []):
                if isinstance(c, dict) and isinstance(c.get("condition"), str):
                    c["condition"] = _dedupe_object_found_args(
                        _rename_tokens(c["condition"], rename))
                if isinstance(c, dict) and isinstance(c.get("alternatives"), list):
                    c["alternatives"] = [
                        _dedupe_object_found_args(_rename_tokens(a, rename))
                        if isinstance(a, str) else a
                        for a in c["alternatives"]
                    ]
    return stages


def fix_object_in_gripper_from_selected_object(stages: list[dict], task_context=None) -> list[dict]:
    """Use selected_object as the oracle for object_in_gripper.

    selected_object(X) at the engage waypoint is confirmed correct by the LLM.
    We propagate X into every object_in_gripper / gripper_released condition until
    the gripper opens (gripper_condition = Open postcondition).

    When the task context provides held_object ground truth, it is used to:
      - Validate that selected_object(X) is not premature (oracle confirms X is
        held within the next ORACLE_LOOKAHEAD waypoints).  If not confirmed, the
        condition is stripped so it cannot cascade into wrong object_in_gripper.
      - Directly set current_held at waypoints where the oracle is authoritative
        (held_object is non-null), covering cases where selected_object is absent.
    When the task context has no non-null held_object entries (e.g. tasks where
    the held object is determined dynamically), the function falls back to
    selected_object-only mode.
    """
    ORACLE_LOOKAHEAD = 2

    # Build oracle: waypoint_index → held object name (only non-null entries).
    held_oracle: dict[int, str] = {}
    if task_context:
        try:
            ctx = json.loads(task_context) if isinstance(task_context, str) else task_context
            for wp in (ctx.get("waypoints") or []):
                idx = wp.get("waypoint")
                if idx is not None and wp.get("held_object"):
                    held_oracle[idx] = wp["held_object"]
        except Exception:
            pass
    has_oracle = bool(held_oracle)

    # Ground-truth veto: when the description carries stamped gripper states and
    # NO waypoint ever holds an object, the selected_object -> object_in_gripper
    # inference is invalid. That would otherwise turn a contact-only closed
    # gripper into a false held-object state.
    # Strip any hallucinated object_in_gripper and skip the inference entirely.
    gt_never_holds = False
    if task_context:
        try:
            _ctx = json.loads(task_context) if isinstance(task_context, str) else task_context
            if has_stamped_gripper_fields(_ctx):
                _wps = _ctx.get("waypoints") or []
                if _wps and not any(w.get("held_object") for w in _wps):
                    gt_never_holds = True
        except Exception:
            pass
    if gt_never_holds:
        for stage in stages:
            for key in ("preconditions", "postconditions"):
                section = stage.get(key) or []
                if section_has_object_in_gripper(section):
                    stage[key] = merge_condition_blocks([
                        c for c in section
                        if not re.search(r"\bobject_in_gripper\(", condition_text(c))
                    ])
        return stages

    def _oracle_confirms(stage_pos: int, obj: str) -> bool:
        if not has_oracle:
            return True
        for k in range(ORACLE_LOOKAHEAD + 1):
            if held_oracle.get(stage_pos + k) == obj:
                return True
        return False

    current_held: str | None = None

    for stage in stages:
        pos = stage.get("stage", 0)

        # Oracle takes priority: directly sets current_held for confirmed hold stages.
        if has_oracle and held_oracle.get(pos):
            current_held = held_oracle[pos]
        else:
            # Validate and apply selected_object from LLM preconditions.
            new_pre = []
            for c in stage.get("preconditions", []):
                m = re.search(r"\bselected_object\((\w+)\)", condition_text(c))
                if m:
                    candidate = m.group(1)
                    if _oracle_confirms(pos, candidate):
                        current_held = candidate
                        new_pre.append(c)
                    # else: premature selected_object — drop it silently
                else:
                    new_pre.append(c)
            stage["preconditions"] = new_pre

        pre = stage.get("preconditions", [])
        post = stage.get("postconditions", [])

        if current_held:
            for section in (pre, post):
                for c in section:
                    txt = condition_text(c)
                    if re.search(r"\bobject_in_gripper\(", txt):
                        c["condition"] = re.sub(
                            r"\bobject_in_gripper\((\w+)\)",
                            f"object_in_gripper({current_held})",
                            txt,
                        )
                    elif re.search(r"\bgripper_released\(", txt):
                        c["condition"] = re.sub(
                            r"\bgripper_released\((\w+)\)",
                            f"gripper_released({current_held})",
                            txt,
                        )
            # If the LLM wrote gripper_condition = Closed but no object_in_gripper,
            # inject object_in_gripper so enforce_gripper_condition_continuity can
            # propagate it correctly instead of falling back to gripper_condition = Closed.
            for section_key in ("preconditions", "postconditions"):
                section = stage.get(section_key, [])
                if (section_gripper_condition(section) == "Closed"
                        and not section_has_object_in_gripper(section)):
                    stage[section_key] = remove_gripper_condition_blocks(section)
                    stage[section_key].append(
                        object_in_gripper_block(
                            current_held,
                            "Injected from selected_object: replaces gripper_condition = Closed.",
                        )
                    )
                    stage[section_key] = merge_condition_blocks(stage[section_key])
        else:
            # No object committed yet — strip any object_in_gripper the LLM
            # hallucinated before a valid commit was seen.
            for section_key in ("preconditions", "postconditions"):
                section = stage.get(section_key, [])
                if section_has_object_in_gripper(section):
                    stage[section_key] = merge_condition_blocks(
                        [c for c in section
                         if not re.search(r"\bobject_in_gripper\(", condition_text(c))]
                    )

        # If post opens the gripper, the object is no longer held.
        for c in stage.get("postconditions", []):
            txt = condition_text(c)
            if "gripper_condition = Open" in txt or re.search(r"\bgripper_released\(", txt):
                current_held = None
                break

    return stages


def place_grasp_alignment(stages: list[dict], task_context) -> list[dict]:
    """Put end_effector_aligned_with / gripper_oriented_for on the actual grasp
    waypoint and nowhere else.

    The description encodes a grasp across waypoints: the gripper transitions
    Open -> Closed at the engage waypoint, but the object only registers as
    'holding' on the following lift waypoint. The grasp — where alignment and
    orientation physically matter — is the Open->Closed engage, while the lift
    already has a closed gripper. Keying alignment to the object_in_gripper
    postcondition (the lift) puts it one waypoint too late, on a closed gripper.

    So: find each engaged run (Closed/holding bounded by Open). If the run ends
    up holding an object, its FIRST waypoint (the Open->Closed engage) is the
    grasp of that object — alignment + orientation belong there as preconditions.
    Runs that never hold anything get none. Strip these predicates from every
    other waypoint. Runs after enforce_gripper_states.
    """
    states = _authoritative_gripper_end_states(task_context)
    if not states:
        return strip_alignment_when_holding(stages)

    pre_states, post_states = _stage_pre_post_states(stages, states)
    n = len(stages)

    # Identify each grasp: an engaged run (Closed/holding bounded by Open) that
    # ends up holding an object. (engage_pos, run_end_exclusive, held_obj)
    grasps = []
    pos = 0
    while pos < n:
        pre, post = pre_states[pos], post_states[pos]
        if pre and pre[0] == "open" and post and post[0] in ("closed", "holding"):
            obj, run = None, pos
            while run < n and post_states[run] and post_states[run][0] in ("closed", "holding"):
                if post_states[run][0] == "holding" and obj is None:
                    obj = post_states[run][1]
                run += 1
            if obj:
                grasps.append((pos, run, obj))
        pos += 1

    # Alignment/orientation only ever belong on the grasp waypoint: clear them
    # everywhere first, then add them back on each grasp's pickup waypoint.
    for stage in stages:
        for section in ("preconditions", "postconditions"):
            stage[section] = [
                c for c in (stage.get(section) or [])
                if not _ALIGNMENT_PREDICATE_RE.search(condition_text(c))
            ]

    for engage_pos, run_end, held_obj in grasps:
        # The pickup waypoint is the one inside the run that commits to the object
        # (carries selected_object) — that is where the robot actually grasps and
        # where alignment belongs. The simulator sometimes records the gripper
        # closed one waypoint early (a pre-grasp pose), so keying off the Open->
        # Closed transition alone can land a waypoint too early; selected_object
        # tracks the real pickup. Fall back to the engage waypoint if none.
        target_pos, name = engage_pos, held_obj
        for p in range(engage_pos, run_end):
            sel = None
            for c in (stages[p].get("preconditions") or []):
                m = re.search(r"selected_object\((\w+)\)", condition_text(c))
                if m:
                    sel = m.group(1)
                    break
            if sel:
                target_pos, name = p, sel
                break

        pre = stages[target_pos].get("preconditions") or []
        # Never add alignment to a section that already holds this object.
        if any(re.search(rf"object_in_gripper\(\s*{re.escape(name)}\s*\)", condition_text(c))
               for c in pre):
            continue
        blocks = [
            normalize_condition({
                "condition": f"end_effector_aligned_with({name}) == true",
                "failure_links": [{
                    "failure": "WrongPosition",
                    "reason": "First engagement must be aligned before the gripper closes.",
                }],
            }),
            normalize_condition({
                "condition": f"gripper_oriented_for({name}) == True",
                "failure_links": [{
                    "failure": "WrongOrientation",
                    "reason": "Grasp orientation must match the object.",
                }],
            }),
        ]
        idx = next((i for i, c in enumerate(pre)
                    if "selected_object(" in condition_text(c)), None)
        if idx is None:
            stages[target_pos]["preconditions"] = _insert_alignment_blocks(pre, blocks)
        else:
            stages[target_pos]["preconditions"] = pre[:idx + 1] + blocks + pre[idx + 1:]
    return stages


def enforce_gripper_states(stages: list[dict], task_context) -> list[dict]:
    """Make the gripper pre/postcondition of every stage match simulator ground
    truth: post(stage_i) = end-state(wp_i); pre(stage_i) = end-state(wp_{i-1})
    (pre of the first stage = Open). The language model does not reliably keep
    gripper continuity across waypoints, so this enforces it deterministically.
    """
    states = _authoritative_gripper_end_states(task_context)
    if not states:
        return stages
    for pos, stage in enumerate(stages):
        wp_idx = stage.get("stage", pos)
        post_state = states.get(wp_idx)
        if pos == 0:
            pre_state = ("open", None)
        else:
            prev_idx = stages[pos - 1].get("stage", pos - 1)
            pre_state = states.get(prev_idx)
        for section, want in (("preconditions", pre_state),
                              ("postconditions", post_state)):
            if want is None:
                continue
            conds = [c for c in (stage.get(section) or [])
                     if not _is_gripper_state_condition(c)]
            conds.append(normalize_condition({
                "condition": _gripper_state_condition_text(want),
                "failure_links": [{
                    "failure": "ExecutionSequenceMismatch",
                    "reason": "Gripper state from simulator ground truth.",
                }],
            }))
            stage[section] = conds
    return stages


def _filter_object_found_stage0(stages: list[dict], task_context) -> list[dict]:
    """Remove objects from Object_found at stage 0 that are not among stage 0's
    waypoint objects (held object + approx_distance_to_objects keys).

    The LLM sometimes includes objects that are hidden inside closed containers
    at task start (e.g. a cube inside a closed drawer) because multi-object
    guidance asks it to list all grasped objects across all phases. The objects
    stage 0 actually records a distance to (plus anything held) are what is
    visible at task start; anything else is hidden and must not appear in
    Object_found.
    """
    if not stages:
        return stages
    try:
        ctx = json.loads(task_context) if isinstance(task_context, str) else task_context
        stage0_wp = next(
            (wp for wp in (ctx.get("waypoints") or []) if wp.get("waypoint") == 0),
            None,
        )
        stage0_relevant = set(waypoint_object_names(stage0_wp)) if stage0_wp else set()
    except Exception:
        return stages
    if not stage0_relevant:
        return stages

    new_pre = []
    for c in (stages[0].get("preconditions") or []):
        text = condition_text(c)
        m = re.match(r"Object_found\(([^)]+)\)\s*==\s*True", text, re.IGNORECASE)
        if m:
            args = [a.strip() for a in m.group(1).split(",")]
            filtered = [a for a in args if a in stage0_relevant]
            if not filtered:
                new_pre.append(c)
                continue
            c = dict(c)
            c["condition"] = f"Object_found({', '.join(filtered)}) == True"
        new_pre.append(c)
    stages[0]["preconditions"] = new_pre
    return stages


def _finalize_stages(stages: list, task_context, task_name: str) -> list:
    """Deterministic normalization applied after generation.

    Runs the cleanup pipeline, stamps simulator ground truth (gripper states,
    grasped-object identity, press/alignment fixes), and attaches hold conditions.
    """
    stages = apply_condition_cleanup_pipeline(stages)
    # Multi-object tasks have misaligned simulator waypoint indices vs video-analysis
    # stages, so enforce_gripper_states / normalize_grasped_object_identity would stamp
    # the wrong object name (e.g. "bottle" on fridge-door waypoints); those tasks get
    # explicit per-waypoint guidance instead. Pure-press tasks (e.g. lamp_off/lamp_on)
    # keep aligned indices and no held object, so they MUST still get ground-truth
    # gripper enforcement — otherwise a press waypoint that closes the gripper is left
    # as gripper_condition = Open. tv_on is covered via is_two_object_interaction.
    _skip_sim_override = (
        is_two_object_interaction(task_name)
        or is_three_object_interaction(task_name)
        or is_five_interaction(task_name)
    )
    if not _skip_sim_override:
        stages = enforce_gripper_states(stages, task_context)
        stages = normalize_grasped_object_identity(stages, task_context)
    stages = fix_object_in_gripper_from_selected_object(stages, task_context)
    stages = enforce_gripper_condition_continuity(stages)
    stages = fix_press_stage_conditions(stages)
    stages = place_grasp_alignment(stages, task_context)
    stages = enforce_pre_grasp_conditions(stages)
    stages = fix_press_stage_conditions(stages)
    stages = _filter_object_found_stage0(stages, task_context)
    # Last: every object reference (including the ones the passes above stamped
    # from simulator handles) is rewritten to the descriptive scene name.
    stages = normalize_scene_object_names(stages, task_context)
    attach_hold_conditions(stages)
    return stages


def generate_conditions(
    failure_definitions: dict,
    task_context: str,
    client,
    provider: str,
    reviewer_guidance: str = "",
    task_context_path: Path | None = None,
    postprocess: bool = False,  # default OFF: return the raw LLM result; pass True to run the deterministic cleanup pipeline
    reviewer: bool | None = None,  # None -> BT_USE_REVIEWER env (default ON); True/False forces Agent 2 on/off
) -> dict:
    _reset_token_usage()
    _task_name = (
        task_name_from_context_path(task_context_path)
        if task_context_path is not None
        else ""
    )
    user_message = build_user_message(
        task_context,
        failure_definitions,
        reviewer_guidance,
        task_name=_task_name,
    )
    _system_prompt = get_system_prompt(_task_name)

    # Agent 1: generate draft
    if provider == "claude":
        draft_raw = generate_conditions_with_claude(client, user_message, _system_prompt)
    elif provider == "openai":
        draft_raw = generate_conditions_with_openai(client, user_message, _system_prompt)
    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose 'claude' or 'openai'.")

    # Agent 2: structural review + auto-fix of the raw draft. Runs before both the
    # raw return and the deterministic pipeline so its corrected stages feed either
    # path. Never hard-fails generation — a reviewer error leaves the draft as-is.
    review_report = None
    if _reviewer_enabled(reviewer):
        try:
            review = review_conditions(
                draft_raw.get("stages", []), task_context, client, provider
            )
            draft_raw["stages"] = review["stages"]
            review_report = {"violations": review["violations"]}
            _print_review_report(review["violations"])
        except Exception as e:
            print(f"[Agent 2 reviewer skipped: {e}]\n", flush=True)

    # Token accounting: summarize what this build actually spent (per agent and
    # per waypoint) and carry the raw records on the result for programmatic use.
    try:
        _nwp = len(draft_raw.get("stages", []))
        print(format_bt_token_usage(_nwp), flush=True)
    except Exception:
        pass
    token_usage = {"records": list(BT_TOKEN_USAGE)}

    # Raw mode: return what the LLM produced (as corrected by Agent 2) — no cleanup
    # pipeline and no ground-truth stamping. Hold conditions are deterministic (the
    # runtime detectors that must hold on every waypoint), so they are still attached
    # here regardless of postprocess.
    if not postprocess:
        stages = draft_raw.get("stages", [])
        stages = normalize_scene_object_names(stages, task_context)
        attach_hold_conditions(stages)
        out = {"source": "api_generation_raw", "stages": stages}
        if review_report is not None:
            out["review"] = review_report
        out["token_usage"] = token_usage
        return out

    try:
        draft = normalize_generated_conditions(draft_raw, task_context)
    except Exception as e:
        print(f"[Agent 1 output could not be parsed: {e} — output may be truncated, try increasing BT_CONDITION_MAX_TOKENS]\n", flush=True)
        draft = {"stages": []}

    stages = draft.get("stages", [])

    # Remember each waypoint's Step-1 primitive, keyed by stage number, before the
    # cleanup passes rebuild the stage dicts (which would drop the key).
    primitive_by_stage = {
        s.get("stage"): s.get("primitive")
        for s in stages
        if s.get("primitive") is not None
    }

    # Deterministic normalization of the Agent 1 draft.
    stages = _finalize_stages(stages, task_context, _task_name)

    # Re-stamp the primitive the pipeline stripped (stage numbers are preserved).
    for index, stage in enumerate(stages):
        stage["primitive"] = primitive_by_stage.get(stage.get("stage", index))
    result = {"source": "api_generation", "stages": stages}
    if review_report is not None:
        result["review"] = review_report
    result["token_usage"] = token_usage
    return result
