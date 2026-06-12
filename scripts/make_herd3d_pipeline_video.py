"""Generate a single-row dialogue-style Herd3D pipeline video."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "resources" / "herd3d_pipeline.mp4"
FPS = 10
W, H = 1400, 360


def robot_icon(ax, x, y, r=0.018):
    c = Circle((x, y), r, transform=ax.transAxes, facecolor="#f7fafc", edgecolor="#2d3748", linewidth=1.5, zorder=4)
    ax.add_patch(c)
    ax.add_patch(Circle((x - 0.006, y + 0.004), 0.003, transform=ax.transAxes, facecolor="#2d3748", zorder=5))
    ax.add_patch(Circle((x + 0.006, y + 0.004), 0.003, transform=ax.transAxes, facecolor="#2d3748", zorder=5))
    ax.plot([x - 0.007, x + 0.007], [y - 0.005, y - 0.005], color="#2d3748", linewidth=1.5,
            transform=ax.transAxes, zorder=5)


def speech_bubble(ax, x, y, w, h, text, fontsize=8.2, active=False):
    edge = "#3182ce" if active else "#cbd5e0"
    face = "#ebf8ff" if active else "#ffffff"
    lw = 2.2 if active else 1.4
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=lw, zorder=3,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color="#1a202c", transform=ax.transAxes, zorder=4, linespacing=1.35)


def panel(ax, x, y, w, h, title, draw_fn=None, active=False):
    edge = "#38b2ac" if active else "#a0aec0"
    lw = 2.4 if active else 1.5
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
        transform=ax.transAxes, facecolor="#ffffff", edgecolor=edge, linewidth=lw, zorder=2,
    ))
    ax.text(x + w / 2, y + h + 0.02, title, ha="center", va="bottom", fontsize=9,
            fontweight="bold", color="#2d3748", transform=ax.transAxes, zorder=4)
    if draw_fn:
        draw_fn(ax, x, y, w, h)


def draw_rgb(ax, x, y, w, h):
    for i, cx in enumerate(np.linspace(x + 0.015, x + w - 0.04, 3)):
        cy = y + 0.025 + 0.01 * (i % 2)
        ax.add_patch(mpatches.Ellipse((cx + 0.018, cy + 0.04), 0.022, 0.032,
                                      transform=ax.transAxes, facecolor="#ed8936", edgecolor="#c05621", zorder=3))
        ax.add_patch(mpatches.Rectangle((cx + 0.005, cy + 0.015), 0.026, 0.022, transform=ax.transAxes,
                                        facecolor="#faf089", edgecolor="#b7791f", zorder=3))
    ax.add_patch(mpatches.Rectangle((x + 0.01, y + 0.01), w - 0.02, 0.012, transform=ax.transAxes,
                                    facecolor="#68d391", edgecolor="none", zorder=3))


def draw_output(ax, x, y, w, h):
    draw_rgb(ax, x, y, w, h)
    ax.text(x + w / 2, y + 0.08, "+ labels", ha="center", fontsize=7, color="#4a5568",
            transform=ax.transAxes, zorder=4)


def draw_controlnet(ax, x, y, w, h):
    ax.text(x + w / 2, y + h * 0.55, "Qwen-Image", ha="center", fontsize=8.5, fontweight="bold",
            color="#553c9a", transform=ax.transAxes, zorder=3)
    ax.text(x + w / 2, y + h * 0.35, "ControlNet", ha="center", fontsize=8.5, fontweight="bold",
            color="#553c9a", transform=ax.transAxes, zorder=3)
    ax.text(x + w / 2, y + h * 0.15, "Union", ha="center", fontsize=8, color="#6b46c1",
            transform=ax.transAxes, zorder=3)


def arrow(ax, x1, y1, x2, y2, alpha=1.0, lw=2.5):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color="#718096", alpha=alpha, transform=ax.transAxes, zorder=2,
    )
    ax.add_patch(arr)


def draw_frame(active_idx, pulse):
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fafbfc")

    ax.text(0.5, 0.92, "Herd3D Data Generation Pipeline", ha="center", fontsize=16,
            fontweight="bold", color="#1a202c")

    y_panel = 0.18
    ph, pw = 0.42, 0.11

    # 1 RGB
    panel(ax, 0.03, y_panel, pw, ph, "RGB render", draw_rgb, active=active_idx == 0)

    # 2 QwenVL stage 1
    robot_icon(ax, 0.17, 0.72)
    speech_bubble(
        ax, 0.155, 0.58, 0.17, 0.11,
        "Qwen3-VL · Stage 1\nRead per-animal facing\nDog₁ ←  Dog₂ →  Dog₃ ←",
        active=active_idx == 1,
    )

    # 3 QwenVL stage 2 (composed prompt)
    robot_icon(ax, 0.36, 0.72)
    speech_bubble(
        ax, 0.345, 0.58, 0.19, 0.11,
        "Qwen3-VL · Stage 2\nComposed prompt:\n\"3 dogs on grass, left-facing…\"",
        active=active_idx == 2,
    )

    # arrows RGB -> bubbles -> controlnet -> output
    a_alpha = 0.45 + 0.55 * pulse
    arrow(ax, 0.14, 0.39, 0.155, 0.58, alpha=a_alpha if active_idx >= 1 else 0.35)
    arrow(ax, 0.325, 0.625, 0.345, 0.625, alpha=a_alpha if active_idx >= 2 else 0.35)
    arrow(ax, 0.535, 0.625, 0.56, 0.39, alpha=a_alpha if active_idx >= 3 else 0.35, lw=3.0)

    # 4 ControlNet
    panel(ax, 0.56, y_panel, 0.12, ph, "ControlNet", draw_controlnet, active=active_idx == 3)

    arrow(ax, 0.68, 0.39, 0.71, 0.39, alpha=a_alpha if active_idx >= 4 else 0.35, lw=3.0)

    # 5 Output
    panel(ax, 0.71, y_panel, pw, ph, "Synthetic image", draw_output, active=active_idx == 4)

    ax.text(0.5, 0.06, "RGB  →  orientation reading  →  prompt synthesis  →  ControlNet  →  labeled image",
            ha="center", fontsize=10, color="#718096", style="italic")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    img = buf[:, :, :3].copy()
    plt.close(fig)
    return img


def main():
    frames = []
    order = [0, 1, 2, 3, 4]
    for idx in order:
        for f in range(14):
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * f / 14)
            frames.append(draw_frame(idx, pulse))
    for idx in reversed(order[:-1]):
        for _ in range(4):
            frames.append(draw_frame(idx, 0))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        imageio.mimsave(OUT, frames, fps=FPS, codec="libx264", quality=8,
                        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    except Exception:
        import cv2
        writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for fr in frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()
    print(f"Wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
