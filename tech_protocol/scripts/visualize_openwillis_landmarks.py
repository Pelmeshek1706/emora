#!/usr/bin/env python3
"""Generate numbered OpenWillis/MediaPipe face landmark region visualizations."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from matplotlib import patheffects
from matplotlib.lines import Line2D


NUM_LANDMARKS = 468
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "openwillis-face/src/openwillis/face/config/facial.json"
VIDEO_PATH = REPO_ROOT / "sample_data/expressive.mp4"
OUTPUT_DIR = REPO_ROOT / "tech_protocol/assets/openwillis_landmarks"

COLORS = {
    "overall": "#202124",
    "upper_face": "#2f66d0",
    "lower_face": "#009b8f",
    "lips": "#d93025",
    "eyebrows": "#f9ab00",
    "other": "#b8b8b8",
}


def openwillis_label(index: int) -> str:
    """Return the OpenWillis 1-based landmark label for a MediaPipe index."""
    return f"lmk{index + 1:03d}"


def load_regions() -> dict[str, list[int]]:
    with CONFIG_PATH.open() as f:
        config = json.load(f)

    lower_face = config["lower_face_landmarks"]
    lower_face_unique = sorted(set(lower_face))
    upper_face = [index for index in range(NUM_LANDMARKS) if index not in set(lower_face)]

    return {
        "overall": list(range(NUM_LANDMARKS)),
        "upper_face": upper_face,
        "lower_face": lower_face_unique,
        "lips": sorted(set(config["lips_landmarks"])),
        "eyebrows": sorted(set(config["eyebrows_landmarks"])),
    }


def extract_landmarks() -> tuple[int, np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO_PATH}")

    face_mesh_solution = mp.solutions.face_mesh
    with face_mesh_solution.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    ) as face_mesh:
        frame_idx = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)
            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark[:NUM_LANDMARKS]
                height, width = rgb.shape[:2]
                coords = np.array(
                    [[point.x * width, point.y * height, point.z] for point in landmarks],
                    dtype=float,
                )
                cap.release()
                return frame_idx, rgb, coords

            frame_idx += 1

    cap.release()
    raise RuntimeError(f"No face landmarks detected in {VIDEO_PATH}")


def face_axis_limits(coords: np.ndarray) -> tuple[float, float, float, float]:
    x_min, y_min = coords[:, :2].min(axis=0)
    x_max, y_max = coords[:, :2].max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    margin_x = width * 0.16
    margin_y = height * 0.16
    return x_min - margin_x, x_max + margin_x, y_min - margin_y, y_max + margin_y


def draw_labels(ax: plt.Axes, coords: np.ndarray, indices: Iterable[int], color: str, size: float) -> None:
    text_effect = [patheffects.withStroke(linewidth=1.5, foreground="white")]
    for index in indices:
        x, y, _ = coords[index]
        ax.text(
            x + 2.0,
            y - 1.5,
            f"{index + 1:03d}",
            color=color,
            fontsize=size,
            fontfamily="DejaVu Sans Mono",
            path_effects=text_effect,
        )


def save_plot(
    image: np.ndarray,
    coords: np.ndarray,
    selected_indices: list[int],
    title: str,
    filename_stem: str,
    selected_color: str,
    frame_idx: int,
    overlay_regions: dict[str, list[int]] | None = None,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x_min, x_max, y_min, y_max = face_axis_limits(coords)
    fig, ax = plt.subplots(figsize=(10.5, 12), dpi=220)
    ax.imshow(image)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)
    ax.set_axis_off()

    if overlay_regions:
        lower = set(overlay_regions["lower_face"])
        lips = set(overlay_regions["lips"])
        eyebrows = set(overlay_regions["eyebrows"])

        # Display categories are exclusive so nested lips/eyebrows remain visible.
        categories = {
            "upper_face": [],
            "lower_face": [],
            "lips": [],
            "eyebrows": [],
        }
        for index in range(NUM_LANDMARKS):
            if index in lips:
                categories["lips"].append(index)
            elif index in eyebrows:
                categories["eyebrows"].append(index)
            elif index in lower:
                categories["lower_face"].append(index)
            else:
                categories["upper_face"].append(index)

        for name, indices in categories.items():
            xy = coords[indices, :2]
            ax.scatter(
                xy[:, 0],
                xy[:, 1],
                s=18,
                c=COLORS[name],
                edgecolors="white",
                linewidths=0.4,
                alpha=0.96,
                label=name.replace("_", " "),
            )
            draw_labels(ax, coords, indices, COLORS[name], size=3.3)
    else:
        xy_all = coords[:, :2]
        ax.scatter(
            xy_all[:, 0],
            xy_all[:, 1],
            s=9,
            c=COLORS["other"],
            edgecolors="none",
            alpha=0.38,
        )

        xy_selected = coords[selected_indices, :2]
        ax.scatter(
            xy_selected[:, 0],
            xy_selected[:, 1],
            s=24,
            c=selected_color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.98,
        )
        draw_labels(
            ax,
            coords,
            selected_indices,
            selected_color,
            size=3.2 if len(selected_indices) > 120 else 5.2,
        )

    legend_handles = []
    if overlay_regions:
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["upper_face"], label="upper face", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["lower_face"], label="lower face", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["lips"], label="lips", markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["eyebrows"], label="eyebrows", markersize=7),
        ]
    elif filename_stem != "overall_all_landmarks_numbered":
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=selected_color, label=title, markersize=7),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["other"], label="other landmarks", markersize=7),
        ]

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8, framealpha=0.86)

    ax.set_title(
        f"{title}\nOpenWillis labels are 1-based; MediaPipe index = label - 1; source frame {frame_idx}",
        fontsize=12,
        pad=12,
    )
    fig.tight_layout(pad=0.5)
    fig.savefig(OUTPUT_DIR / f"{filename_stem}.svg")
    fig.savefig(OUTPUT_DIR / f"{filename_stem}.png")
    plt.close(fig)


def write_region_membership(regions: dict[str, list[int]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lower = set(regions["lower_face"])
    upper = set(regions["upper_face"])
    lips = set(regions["lips"])
    eyebrows = set(regions["eyebrows"])

    with (OUTPUT_DIR / "openwillis_landmark_region_membership.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mediapipe_index",
                "openwillis_landmark",
                "upper_face",
                "lower_face",
                "lips",
                "eyebrows",
                "display_region",
            ],
        )
        writer.writeheader()
        for index in range(NUM_LANDMARKS):
            if index in lips:
                display_region = "lips"
            elif index in eyebrows:
                display_region = "eyebrows"
            elif index in lower:
                display_region = "lower_face"
            else:
                display_region = "upper_face"

            writer.writerow(
                {
                    "mediapipe_index": index,
                    "openwillis_landmark": openwillis_label(index),
                    "upper_face": index in upper,
                    "lower_face": index in lower,
                    "lips": index in lips,
                    "eyebrows": index in eyebrows,
                    "display_region": display_region,
                }
            )


def main() -> None:
    frame_idx, image, coords = extract_landmarks()
    regions = load_regions()
    write_region_membership(regions)

    save_plot(
        image,
        coords,
        regions["overall"],
        "Overall: all 468 landmarks",
        "overall_all_landmarks_numbered",
        COLORS["overall"],
        frame_idx,
    )
    save_plot(
        image,
        coords,
        regions["overall"],
        "OpenWillis region overlay",
        "openwillis_region_overlay_numbered",
        COLORS["overall"],
        frame_idx,
        overlay_regions=regions,
    )

    for name in ["upper_face", "lower_face", "lips", "eyebrows"]:
        save_plot(
            image,
            coords,
            regions[name],
            f"{name.replace('_', ' ').title()} landmarks",
            f"{name}_landmarks_numbered",
            COLORS[name],
            frame_idx,
        )

    print(f"Wrote landmark visualizations to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
