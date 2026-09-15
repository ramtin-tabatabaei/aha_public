
from aha_publish import paths
import json
import os

import numpy as np


WRIST_VALUES_DIR = os.path.join(
    str(paths.OUTPUT_DIR),
    'wrist_sensor_values',
)

WRIST_VISUALIZATION_MODES = [
    ("save", "Save PNG images only"),
    ("matplotlib", "Show Matplotlib windows and save PNG images"),
    ("off", "Do not save or show wrist images"),
    ("depth_segmentation", "Segment from wrist depth, show Matplotlib windows, and save PNG images"),
    ("rgb_segmentation", "Segment from wrist RGB, show Matplotlib windows, and save PNG images"),
]

DEPTH_SEGMENT_COUNT = 5
RGB_SEGMENT_COUNT = 7
SAM_CHECKPOINT_ENV = "AHA_SAM_CHECKPOINT"
SAM_MODEL_TYPE_ENV = "AHA_SAM_MODEL_TYPE"
_SAM_GENERATOR = None
_SAM_STATUS = None


def top_mask_values(mask, max_items=12):
    flat = np.asarray(mask).reshape(-1)
    values, counts = np.unique(flat, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [
        {"id": int(values[idx]), "pixels": int(counts[idx])}
        for idx in order[:max_items]
    ]


def center_patch(array, size=5):
    arr = np.asarray(array)
    if arr.ndim < 2:
        return []
    half = size // 2
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    patch = arr[
        max(0, cy - half):min(arr.shape[0], cy + half + 1),
        max(0, cx - half):min(arr.shape[1], cx + half + 1),
    ]
    return np.round(patch, 4).tolist()


def as_2d_image(array):
    arr = np.asarray(array)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def as_rgb_image(array):
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        if np.nanmax(arr) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def normalize_depth_for_display(depth):
    arr = as_2d_image(depth).astype(np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros(arr.shape, dtype=np.uint8)

    valid = arr[finite]
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)

    normalized = (arr - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~finite] = 0.0
    return (normalized * 255.0).astype(np.uint8)


def kmeans_labels(features, shape, num_segments, iterations=12, seed=0):
    features = np.asarray(features, dtype=np.float32)
    if features.size == 0:
        return np.zeros(shape, dtype=np.int32)

    num_segments = max(1, min(int(num_segments), len(features)))
    rng = np.random.default_rng(seed)
    initial_ids = rng.choice(len(features), size=num_segments, replace=False)
    centers = features[initial_ids].copy()

    labels = np.zeros(len(features), dtype=np.int32)
    for _ in range(iterations):
        distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1).astype(np.int32)

        for i in range(num_segments):
            members = features[labels == i]
            if len(members):
                centers[i] = np.mean(members, axis=0)
            else:
                centers[i] = features[rng.integers(0, len(features))]

    return labels.reshape(shape)


def segment_from_depth(depth, num_segments=DEPTH_SEGMENT_COUNT):
    arr = as_2d_image(depth).astype(np.float32)
    finite = np.isfinite(arr)
    labels = np.zeros(arr.shape, dtype=np.int32)
    if not np.any(finite):
        return labels

    valid = arr[finite]
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if hi <= lo:
        labels[finite] = 1
        return labels

    depth_feature = ((valid - lo) / (hi - lo)).reshape(-1, 1)
    clustered = kmeans_labels(
        depth_feature,
        (len(valid),),
        num_segments=num_segments,
        seed=11,
    )
    labels[finite] = clustered.astype(np.int32) + 1
    return labels


def segment_rgb_with_sam(rgb):
    global _SAM_GENERATOR, _SAM_STATUS

    checkpoint = os.environ.get(SAM_CHECKPOINT_ENV)
    if not checkpoint:
        return None, f"SAM skipped: set {SAM_CHECKPOINT_ENV} to a checkpoint path"

    try:
        if _SAM_GENERATOR is None:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

            model_type = os.environ.get(SAM_MODEL_TYPE_ENV, "vit_b")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            sam.to(device=device)
            _SAM_GENERATOR = SamAutomaticMaskGenerator(sam)
            _SAM_STATUS = f"SAM {model_type} on {device}"

        masks = _SAM_GENERATOR.generate(as_rgb_image(rgb))
        label_map = np.zeros(rgb.shape[:2], dtype=np.int32)
        ordered_masks = sorted(masks, key=lambda item: item.get("area", 0), reverse=True)
        for i, mask_info in enumerate(ordered_masks, start=1):
            label_map[np.asarray(mask_info["segmentation"], dtype=bool)] = i
        return label_map, _SAM_STATUS
    except Exception as e:
        return None, f"SAM unavailable: {e}"


def segment_rgb_with_kmeans(rgb, num_segments=RGB_SEGMENT_COUNT):
    image = as_rgb_image(rgb)
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    color = image.astype(np.float32) / 255.0
    xy = np.stack(
        [
            xx.astype(np.float32) / max(w - 1, 1),
            yy.astype(np.float32) / max(h - 1, 1),
        ],
        axis=-1,
    )
    features = np.concatenate([color, xy * 0.35], axis=-1).reshape(-1, 5)
    return kmeans_labels(features, (h, w), num_segments=num_segments, seed=23) + 1


def segment_from_rgb(rgb):
    image = as_rgb_image(rgb)
    sam_labels, sam_status = segment_rgb_with_sam(image)
    if sam_labels is not None:
        return sam_labels, sam_status
    return segment_rgb_with_kmeans(image), f"RGB k-means fallback ({sam_status})"


def colorize_mask(mask):
    arr = as_2d_image(mask)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)[..., 0]
    arr = arr.astype(np.int64)

    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    values = np.unique(arr)
    for value in values:
        if value < 0:
            color = np.array([0, 0, 0], dtype=np.uint8)
        else:
            # Deterministic pseudo-random color for each segmentation id.
            color = np.array([
                (value * 37 + 61) % 255,
                (value * 17 + 149) % 255,
                (value * 97 + 29) % 255,
            ], dtype=np.uint8)
        rgb[arr == value] = color
    return rgb


def save_image_with_matplotlib(path, image, cmap=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"    Could not save visualization with matplotlib: {e}")
        return False

    try:
        plt.imsave(path, image, cmap=cmap)
        return True
    except Exception as e:
        print(f"    Could not save visualization image '{path}': {e}")
        return False


def save_visualization_pngs(out_dir, label, depth=None, mask=None, rgb=None, mask_name="segmentation"):
    saved_paths = {}
    if depth is not None:
        depth_display = normalize_depth_for_display(depth)
        depth_png = os.path.join(out_dir, f'{label}_wrist_depth.png')
        if save_image_with_matplotlib(depth_png, depth_display, cmap='viridis'):
            saved_paths["depth_png"] = depth_png

    if rgb is not None:
        rgb_png = os.path.join(out_dir, f'{label}_wrist_rgb.png')
        if save_image_with_matplotlib(rgb_png, as_rgb_image(rgb)):
            saved_paths["rgb_png"] = rgb_png

    if mask is not None:
        mask_display = colorize_mask(mask)
        mask_png = os.path.join(out_dir, f'{label}_wrist_{mask_name}.png')
        if save_image_with_matplotlib(mask_png, mask_display):
            saved_paths[f"{mask_name}_png"] = mask_png

    return saved_paths


def show_with_matplotlib(label, depth=None, mask=None, rgb=None, mask_title="Segmentation"):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"    Matplotlib view unavailable: {e}")
        return False

    count = int(rgb is not None) + int(depth is not None) + int(mask is not None)
    if count == 0:
        return False

    plt.ion()
    fig = plt.figure("Wrist camera values")
    fig.clf()

    plot_index = 1
    if rgb is not None:
        ax = fig.add_subplot(1, count, plot_index)
        ax.imshow(as_rgb_image(rgb))
        ax.set_title(f"RGB: {label}")
        ax.axis('off')
        plot_index += 1

    if depth is not None:
        ax = fig.add_subplot(1, count, plot_index)
        ax.imshow(as_2d_image(depth), cmap='viridis')
        ax.set_title(f"Depth: {label}")
        ax.axis('off')
        plot_index += 1

    if mask is not None:
        ax = fig.add_subplot(1, count, plot_index)
        ax.imshow(colorize_mask(mask))
        ax.set_title(f"{mask_title}: {label}")
        ax.axis('off')

    fig.tight_layout()
    plt.pause(0.001)
    return True


def visualize_wrist_values(
    out_dir,
    label,
    visualization_mode,
    depth=None,
    mask=None,
    rgb=None,
    mask_name="segmentation",
    mask_title="Wrist segmentation",
):
    if visualization_mode == "off":
        return {}

    saved_paths = save_visualization_pngs(
        out_dir,
        label,
        depth=depth,
        mask=mask,
        rgb=rgb,
        mask_name=mask_name,
    )

    display_mode = visualization_mode
    if visualization_mode in ("depth_segmentation", "rgb_segmentation"):
        display_mode = "matplotlib"

    if display_mode == "matplotlib":
        show_with_matplotlib(label, depth=depth, mask=mask, rgb=rgb, mask_title=mask_title)

    return saved_paths


def close_wrist_visualization(visualization_mode):
    return


def report_wrist_values(scene, task_name, label, depth_units, visualization_mode="save"):
    obs = scene.get_observation()
    out_dir = os.path.join(WRIST_VALUES_DIR, task_name)
    os.makedirs(out_dir, exist_ok=True)

    summary = {"label": label, "depth_units": depth_units}
    print("\n  WRIST SENSOR VALUES")

    depth = None
    rgb = None
    simulator_mask = None
    visualization_mask = None
    visualization_mask_name = "segmentation"
    visualization_mask_title = "Wrist segmentation"

    if obs.wrist_depth is not None:
        depth = as_2d_image(obs.wrist_depth)
        finite_depth = depth[np.isfinite(depth)]
        depth_path = os.path.join(out_dir, f'{label}_wrist_depth.npy')
        np.save(depth_path, depth)
        if finite_depth.size:
            cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
            summary["depth"] = {
                "shape": list(depth.shape),
                "min": float(np.min(finite_depth)),
                "max": float(np.max(finite_depth)),
                "mean": float(np.mean(finite_depth)),
                "center": float(depth[cy, cx]),
                "center_5x5": center_patch(depth),
                "path": depth_path,
            }
            print(
                f"    Depth ({depth_units}): shape={depth.shape}, "
                f"min={summary['depth']['min']:.4f}, "
                f"max={summary['depth']['max']:.4f}, "
                f"mean={summary['depth']['mean']:.4f}, "
                f"center={summary['depth']['center']:.4f}"
            )
            print(f"    Depth center 5x5: {summary['depth']['center_5x5']}")
        else:
            summary["depth"] = {"shape": list(depth.shape), "path": depth_path}
            print(f"    Depth: shape={depth.shape}, no finite values")
    else:
        print("    Depth: not available")

    if getattr(obs, "wrist_rgb", None) is not None:
        rgb = as_rgb_image(obs.wrist_rgb)
        rgb_path = os.path.join(out_dir, f'{label}_wrist_rgb.npy')
        np.save(rgb_path, rgb)
        summary["rgb"] = {
            "shape": list(rgb.shape),
            "path": rgb_path,
        }
        print(f"    RGB: shape={rgb.shape}")
    else:
        print("    RGB: not available")

    if obs.wrist_mask is not None:
        simulator_mask = as_2d_image(obs.wrist_mask)
        visualization_mask = simulator_mask
        mask_path = os.path.join(out_dir, f'{label}_wrist_mask.npy')
        np.save(mask_path, simulator_mask)
        cy, cx = simulator_mask.shape[0] // 2, simulator_mask.shape[1] // 2
        top_values = top_mask_values(simulator_mask)
        summary["segmentation"] = {
            "source": "simulator",
            "shape": list(simulator_mask.shape),
            "center_id": int(simulator_mask[cy, cx]) if simulator_mask.ndim == 2 else np.asarray(simulator_mask[cy, cx]).tolist(),
            "unique_count": int(len(np.unique(simulator_mask.reshape(-1)))),
            "top_values": top_values,
            "center_5x5": center_patch(simulator_mask),
            "path": mask_path,
        }
        print(
            f"    Simulator segmentation: shape={simulator_mask.shape}, "
            f"unique_ids={summary['segmentation']['unique_count']}, "
            f"center_id={summary['segmentation']['center_id']}"
        )
        print(f"    Top simulator segmentation ids: {top_values}")
        print(f"    Simulator segmentation center 5x5: {summary['segmentation']['center_5x5']}")
    else:
        print("    Simulator segmentation: not available")

    if visualization_mode == "depth_segmentation":
        visualization_mask = None
        if depth is not None:
            depth_segments = segment_from_depth(depth)
            depth_segments_path = os.path.join(out_dir, f'{label}_wrist_depth_segments.npy')
            np.save(depth_segments_path, depth_segments)
            top_values = top_mask_values(depth_segments)
            cy, cx = depth_segments.shape[0] // 2, depth_segments.shape[1] // 2
            summary["depth_segmentation"] = {
                "source": "depth_kmeans",
                "shape": list(depth_segments.shape),
                "center_id": int(depth_segments[cy, cx]),
                "unique_count": int(len(np.unique(depth_segments.reshape(-1)))),
                "top_values": top_values,
                "center_5x5": center_patch(depth_segments),
                "path": depth_segments_path,
            }
            print(
                f"    Depth-derived segmentation: shape={depth_segments.shape}, "
                f"unique_ids={summary['depth_segmentation']['unique_count']}, "
                f"center_id={summary['depth_segmentation']['center_id']}"
            )
            print(f"    Top depth-derived ids: {top_values}")
            visualization_mask = depth_segments
            visualization_mask_name = "depth_segmentation"
            visualization_mask_title = "Wrist depth segmentation"
        else:
            print("    Depth-derived segmentation: depth not available")

    elif visualization_mode == "rgb_segmentation":
        visualization_mask = None
        if rgb is not None:
            rgb_segments, method = segment_from_rgb(rgb)
            rgb_segments_path = os.path.join(out_dir, f'{label}_wrist_rgb_segments.npy')
            np.save(rgb_segments_path, rgb_segments)
            top_values = top_mask_values(rgb_segments)
            cy, cx = rgb_segments.shape[0] // 2, rgb_segments.shape[1] // 2
            summary["rgb_segmentation"] = {
                "source": method,
                "shape": list(rgb_segments.shape),
                "center_id": int(rgb_segments[cy, cx]),
                "unique_count": int(len(np.unique(rgb_segments.reshape(-1)))),
                "top_values": top_values,
                "center_5x5": center_patch(rgb_segments),
                "path": rgb_segments_path,
            }
            print(
                f"    RGB-derived segmentation: method={method}, "
                f"shape={rgb_segments.shape}, "
                f"unique_ids={summary['rgb_segmentation']['unique_count']}, "
                f"center_id={summary['rgb_segmentation']['center_id']}"
            )
            print(f"    Top RGB-derived ids: {top_values}")
            visualization_mask = rgb_segments
            visualization_mask_name = "rgb_segmentation"
            visualization_mask_title = "Wrist RGB segmentation"
        else:
            print("    RGB-derived segmentation: RGB not available")

    visualization_paths = visualize_wrist_values(
        out_dir,
        label,
        visualization_mode,
        depth=depth,
        mask=visualization_mask,
        rgb=rgb if visualization_mode == "rgb_segmentation" else None,
        mask_name=visualization_mask_name,
        mask_title=visualization_mask_title,
    )
    if visualization_paths:
        summary["visualizations"] = visualization_paths
        for name, path in visualization_paths.items():
            print(f"    Saved {name}: {path}")

    summary_path = os.path.join(out_dir, f'{label}_wrist_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"    Saved wrist arrays/summary under: {out_dir}")
    return summary
