"""Predicate-level condition filtering helpers."""

from aha_publish import paths

from ..semantics import *


_COLOR_QUALIFIERS: frozenset[str] = frozenset({
    'red', 'blue', 'green', 'yellow', 'gray', 'grey', 'purple',
    'orange', 'white', 'black', 'pink', 'brown', 'cyan', 'magenta',
})


def apply_allowed_predicates_filter(stages: list[dict]) -> list[dict]:
    """Remove condition parts that are not in the allowed predicate list from the JSON."""
    for stage in stages:
        for section in ("preconditions", "postconditions"):
            new_conditions = []
            for cond in stage.get(section, []):
                filtered = filter_condition_to_allowed_predicates(cond)
                if filtered is not None:
                    new_conditions.append(filtered)
            stage[section] = new_conditions
    return stages


def strip_color_qualifiers_from_object_found(stages: list[dict]) -> list[dict]:
    """Strip leading color tokens from Object_found arguments.

    RLBench randomizes object colors between episodes so Object_found(red_button)
    produces a false positive when the scene spawns a gray button. Stripping the
    color prefix to Object_found(button) makes the check color-agnostic.
    Only Object_found is affected; selected_object keeps color qualifiers because
    those are intentional per-instance identity checks.
    """
    def _strip_color(arg: str) -> str:
        tokens = arg.strip().split('_')
        while len(tokens) > 1 and tokens[0].lower() in _COLOR_QUALIFIERS:
            tokens.pop(0)
        result = '_'.join(tokens)
        return result if result and not is_invalid_condition_argument(result) else arg.strip()

    for stage in stages:
        for section in ('preconditions', 'postconditions'):
            for cond in stage.get(section, []):
                text = cond.get('condition', '')
                def _replace(m: re.Match, _sc: type = _strip_color) -> str:
                    stripped = [_sc(a) for a in m.group(1).split(',')]
                    seen: set[str] = set()
                    deduped = [a for a in stripped if not (a in seen or seen.add(a))]  # type: ignore[func-returns-value]
                    return f"Object_found({', '.join(deduped)}) == True"
                new_text = re.sub(
                    r'\bObject_found\(([^)]+)\)\s*==\s*True',
                    _replace,
                    text,
                    flags=re.IGNORECASE,
                )
                if new_text != text:
                    cond['condition'] = new_text
    return stages
