"""General file, image, JSON, token-usage, and geometry helpers."""

from aha_publish import paths

from .prompts import *

def as_path(value: str | None) -> Path | None:
    if not value:
        return None
    return Path(value).expanduser()


def env_float(name: str) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def usage_from_openai(response) -> Usage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage(calls=1)
    return Usage(
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        calls=1,
    )


def usage_from_claude(message) -> Usage:
    usage = getattr(message, "usage", None)
    if usage is None:
        return Usage(calls=1)
    return Usage(
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        calls=1,
    )


def read_text_file(path: Path, max_chars: int) -> str:
    if not path.exists():
        return f"[MISSING FILE: {path}]"
    if path.is_dir():
        files = [
            item
            for item in sorted(path.rglob("*"))
            if item.is_file()
            and "__pycache__" not in item.parts
            and ".git" not in item.parts
        ]
        lines = [
            f"[DIRECTORY: {path}]",
            f"file_count: {len(files)}",
            "files:",
        ]
        for item in files[:80]:
            lines.append(f"- {item.relative_to(path)}")
        if len(files) > 80:
            lines.append(f"[TRUNCATED LIST: showing first 80 of {len(files)} files]")

        text = "\n".join(lines) + "\n"
        for item in files:
            if item.suffix.lower() not in TEXT_CONTEXT_EXTENSIONS:
                continue
            remaining = max_chars - len(text)
            if remaining <= 0:
                break
            file_text = item.read_text(errors="replace")
            section = "\n".join([
                "",
                f"## {item.relative_to(path)}",
                file_text[:remaining],
            ])
            text += section
            if len(file_text) > remaining:
                text += f"\n\n[TRUNCATED: {item} exceeded remaining directory budget.]"
                break
        if len(text) > max_chars:
            return text[:max_chars] + f"\n\n[TRUNCATED: directory context exceeded {max_chars} chars.]"
        return text
    text = path.read_text(errors="replace")
    if len(text) > max_chars:
        return (
            text[:max_chars]
            + f"\n\n[TRUNCATED: {path} has {len(text)} chars; sent first {max_chars}.]"
        )
    return text


# inspect_ttm.py writes two audiences into one .llm_context.json. The description
# prompt gets the world-frame poses and the scene objects that are defined today;
# the parent-frame chain geometry below exists for the transition/orientation
# detectors and is stripped here, so adding it to the report cannot change what the
# model sees as the scene.
# The waypoint_chain / local_orientation_rpy_rad / root_object entries are no longer
# written, but reports produced before that still carry them, so they stay listed.
DETECTOR_ONLY_REPORT_FIELDS = ("task_boundary_root", "waypoint_chain", "reference_variation_index",
                               "hook_reference_rules", "rule_resolved_waypoints",
                               "reference_state", "reference_objects")
DETECTOR_ONLY_WAYPOINT_FIELDS = (
    "local_position_xyz_m",
    "local_quaternion_xyzw",
    "local_orientation_rpy_rad",
    "root_object",
    "local_offset_unavailable",
    "position_parent",
    "orientation_parent",
    "anchor_validation_episodes",
    "reference_unavailable",
)


def strip_detector_geometry(report: dict[str, Any]) -> dict[str, Any]:
    trimmed = {k: v for k, v in report.items() if k not in DETECTOR_ONLY_REPORT_FIELDS}
    waypoints = trimmed.get("waypoints")
    if isinstance(waypoints, list):
        trimmed["waypoints"] = [
            {k: v for k, v in wp.items() if k not in DETECTOR_ONLY_WAYPOINT_FIELDS}
            if isinstance(wp, dict) else wp
            for wp in waypoints
        ]
    return trimmed


def read_context_file(path: Path, max_chars: int) -> str:
    """read_text_file, minus the detector-only geometry in a TTM inspection report."""
    if path.is_file() and path.name.endswith(".llm_context.json"):
        try:
            report = json.loads(path.read_text(errors="replace"))
        except json.JSONDecodeError:
            return read_text_file(path, max_chars)
        if isinstance(report, dict):
            text = json.dumps(strip_detector_geometry(report), indent=2)
            if len(text) > max_chars:
                return (
                    text[:max_chars]
                    + f"\n\n[TRUNCATED: {path} has {len(text)} chars; sent first {max_chars}.]"
                )
            return text
    return read_text_file(path, max_chars)


def load_image_as_base64(image_path: Path) -> tuple[str, str]:
    suffix = image_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Grid image not found: {image_path}")
    with open(image_path, "rb") as f:
        encoded = base64.standard_b64encode(f.read()).decode("utf-8")
    return encoded, SUPPORTED_EXTENSIONS[suffix]


def parse_json_response(raw: str) -> dict[str, Any]:
    clean = raw.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        ).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Some models wrap the JSON in prose; extract the first JSON object.
        start = clean.find("{")
        if start == -1:
            raise
        obj, _ = json.JSONDecoder().raw_decode(clean[start:])
        return obj


def extract_task_from_grid_image(path: Path) -> str:
    name = path.stem
    suffix = "_ALL_WAYPOINTS_COMBINED"
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return name


def find_grid_image(task_name: str, photo_dir: Path) -> Path:
    preferred = photo_dir / f"{task_name}_ALL_WAYPOINTS_COMBINED.png"
    if preferred.exists():
        return preferred
    matches = sorted(photo_dir.glob(f"{task_name}_ALL_WAYPOINTS_COMBINED.*"))
    for match in matches:
        if match.suffix.lower() in SUPPORTED_EXTENSIONS:
            return match
    return preferred


def find_all_grid_images(photo_dir: Path) -> list[Path]:
    if not photo_dir.exists():
        raise FileNotFoundError(f"Photo directory not found: {photo_dir}")
    return [
        path
        for path in sorted(photo_dir.glob("*_ALL_WAYPOINTS_COMBINED.*"))
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def find_ttm_context(task_name: str) -> Path | None:
    # Prefer the lean JSON emitted by inspect_ttm.py; fall back to the legacy .md.
    for ext in ("json", "md"):
        exact = DEFAULT_TTM_CONTEXT_DIR / f"{task_name}.llm_context.{ext}"
        if exact.exists():
            return exact
        matches = sorted(DEFAULT_TTM_CONTEXT_DIR.glob(f"*_{task_name}.llm_context.{ext}"))
        if matches:
            return matches[0]
    return DEFAULT_TTM_CONTEXT_DIR / f"{task_name}.llm_context.json"
