
from aha_publish import paths
import os
import re
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

REPO_ROOT = (paths.PROJECT_ROOT)
BASE_FOLDER = str(paths.OUTPUT_DIR / 'waypoint_screenshots')
# Write combined grids straight into waypoints_description, the canonical home
# for every task's *_ALL_WAYPOINTS_COMBINED.png (where the description generator
# and BT pipeline read them from).
OUTPUT_FOLDER = str(paths.DESCRIPTION_DIR)

# Pattern captures wp like 'wp1', 'wp1.5', 'wp-1'
pattern = re.compile(r"(.+?)_(wp\d+(?:\.\d+)?)_(.+)\.png")

# Only these camera views are combined, in this left-to-right order. Any other
# view (e.g. stale left_shoulder/right_shoulder/overhead shots from older runs)
# is ignored.
CAMERA_ORDER = ["front", "side", "wrist"]


def camera_of(label):
    """Camera name for a frame label ('front' or 'gripper_open_front' -> 'front')."""
    return label.rsplit("_", 1)[-1]


def camera_rank(label):
    """Sort key putting front, then side, then wrist; unknown cameras last."""
    cam = camera_of(label)
    return CAMERA_ORDER.index(cam) if cam in CAMERA_ORDER else len(CAMERA_ORDER)


# Label (camera name) and waypoint-tag font sizes for the combined grid.
LABEL_FONT_SIZE = 34
WP_FONT_SIZE = 52
# Prefer a real TrueType font so the sizes above take effect. arial.ttf is usually
# absent on Linux; DejaVuSans (shipped with Pillow / most distros) is a safe default.
_FONT_CANDIDATES = ["DejaVuSans.ttf", "arial.ttf", "LiberationSans-Regular.ttf"]
_BOLD_FONT_CANDIDATES = ["DejaVuSans-Bold.ttf", "arialbd.ttf", "LiberationSans-Bold.ttf"]


def _load_truetype(candidates, size):
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    # Last resort: Pillow's built-in font. load_default takes a size on Pillow >= 10.1.
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def load_fonts():
    """Return (label_font, wp_font) as sized TrueType fonts when available."""
    return (
        _load_truetype(_FONT_CANDIDATES, LABEL_FONT_SIZE),
        _load_truetype(_BOLD_FONT_CANDIDATES, WP_FONT_SIZE),
    )


def available_tasks():
    """Task folders that have screenshots, sorted by name."""
    if not os.path.isdir(BASE_FOLDER):
        return []
    return sorted(
        name for name in os.listdir(BASE_FOLDER)
        if os.path.isdir(os.path.join(BASE_FOLDER, name))
    )


def prompt_yes_no(question, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input(f"{question} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def is_gripper_frame(label):
    """True for the gripper open/close screenshots (e.g. label 'gripper_open_front')."""
    return label.startswith("gripper_open") or label.startswith("gripper_close")


def prompt_task_choice(tasks):
    """Show a numbered task menu where 0 = all. Returns a task name, or None for all."""
    print("\nAvailable tasks:")
    print("   0. ALL tasks")
    for index, name in enumerate(tasks, start=1):
        print(f"  {index:>2}. {name}")
    while True:
        choice = input(f"\nSelect a task to combine [0-{len(tasks)}] (0 = all, q to quit): ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            print("No task selected. Exiting.")
            sys.exit(0)
        if choice.isdigit():
            number = int(choice)
            if number == 0:
                return None
            if 1 <= number <= len(tasks):
                return tasks[number - 1]
        print(f"Please enter a number between 0 and {len(tasks)}.")


def combine_folder(folder_name, include_gripper=True):
    """Combine one task's per-waypoint camera screenshots into a single grid image.

    When include_gripper is False, the gripper open/close action rows are left out.
    """
    FOLDER = os.path.join(BASE_FOLDER, folder_name)
    if not os.path.isdir(FOLDER):
        print(f"No screenshot folder for: {folder_name}, skipping.")
        return

    groups = defaultdict(list)

    for file in os.listdir(FOLDER):
        if file.endswith(".png"):
            match = pattern.match(file)
            if match:
                base_name, wp, label = match.groups()
                if camera_of(label) not in CAMERA_ORDER:
                    continue
                if not include_gripper and is_gripper_frame(label):
                    continue
                full_path = os.path.join(FOLDER, file)
                groups[(base_name, wp)].append((label, full_path))

    if not groups:
        print(f"No matching images found in: {FOLDER}, skipping.")
        return

    row_images = []

    for (base_name, wp), items in groups.items():
        items.sort(key=lambda x: camera_rank(x[0]))
        images = [Image.open(path) for _, path in items]
        labels = [label for label, _ in items]

        max_h = max(img.height for img in images)
        resized = []
        for img in images:
            if img.height != max_h:
                ratio = max_h / img.height
                img = img.resize((int(img.width * ratio), max_h))
            resized.append(img)

        total_w = sum(img.width for img in resized)

        left_margin = 180
        header_h = 64
        row = Image.new("RGB", (total_w + left_margin, max_h + header_h), (255, 255, 255))
        draw = ImageDraw.Draw(row)

        font, big_font = load_fonts()

        draw.text((14, max_h // 2), wp, fill="black", font=big_font)

        x_offset = left_margin
        for img, label in zip(resized, labels):
            row.paste(img, (x_offset, header_h))
            draw.text((x_offset + 14, 14), label, fill="black", font=font)
            x_offset += img.width

        row_images.append((wp, row))

    # Sort key handles negatives and decimals (wp-1 < wp0 < wp0.5 < wp1 < wp1.5 ...)
    row_images.sort(key=lambda x: float(re.search(r"-?\d+(?:\.\d+)?", x[0]).group()))

    final_width = max(img.width for _, img in row_images)
    final_height = sum(img.height for _, img in row_images)
    final_img = Image.new("RGB", (final_width, final_height), (255, 255, 255))

    y_offset = 0
    for wp, img in row_images:
        final_img.paste(img, (0, y_offset))
        y_offset += img.height

    out_path = os.path.join(OUTPUT_FOLDER, f"{folder_name}_ALL_WAYPOINTS_COMBINED.png")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    final_img.save(out_path)
    print(f"Saved: {out_path}")


def main():
    tasks = available_tasks()
    if not tasks:
        print(f"No task folders found in {BASE_FOLDER}.")
        return

    if sys.stdin.isatty():
        include_gripper = prompt_yes_no("Include gripper open/close frames?", default=True)
        choice = prompt_task_choice(tasks)
        selected = tasks if choice is None else [choice]
    else:
        include_gripper = True
        selected = tasks

    for folder_name in selected:
        combine_folder(folder_name, include_gripper)

    print("\nAll done!")


if __name__ == "__main__":
    main()
