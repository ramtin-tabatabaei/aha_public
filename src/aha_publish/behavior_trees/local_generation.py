"""Task-stage extraction for the GUI bootstrap payload.

The star-import of prompt_building is what propagates ``build_user_message``
(and the cleanup names behind it) into ``llm_generation``; it is load-bearing,
not a leftover.
"""

from aha_publish import paths

from .prompt_building import *

def extract_task_stages(task_context: dict) -> list[dict]:
    stages = []
    source_stages = task_context.get("stages")
    if source_stages is None:
        source_stages = task_context.get("waypoints", [])

    for index, stage in enumerate(source_stages):
        stage_number = stage.get("stage", stage.get("stage_number", stage.get("waypoint", index)))
        summary_parts = [
            stage.get("summary", ""),
            stage.get("visual_summary", ""),
            stage.get("robot_action", ""),
        ]
        stages.append(
            {
                "stage": stage_number,
                "summary": " ".join(part for part in summary_parts if part)
                or stage.get("name")
                or f"Waypoint {stage_number}",
            }
        )
    return stages
