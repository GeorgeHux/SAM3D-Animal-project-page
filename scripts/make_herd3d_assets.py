"""Generate Herd3D pipeline assets: static images + continuous Qwen chat video."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resources"
INPUT_PATH = RES / "herd3d_qwen_input.png"
SYNTH_SRC = Path(r"C:\Users\XUYI HU\Desktop\train_flat\images\train")
SYNTH_OUT = RES / "herd3d_synthetic.mp4"

VID_W, VID_H = 460, 540
ICON_SIZE = 0.040
ICON_GAP = 0.016
BUBBLE_GAP = 0.038
MARGIN_TOP = 0.012
CHAT_FONT_SIZE = 9.0
CHAT_LINE_SPACING = 1.10
BUBBLE_TEXT_PAD = 0.012
BUBBLE_BW = 0.70
_BLOCK_W = 2 * ICON_SIZE + 2 * ICON_GAP + BUBBLE_BW
_BLOCK_X0 = (1.0 - _BLOCK_W) / 2
ASST_ICON_X = _BLOCK_X0 + ICON_SIZE / 2
ASST_ICON_DX = -0.008
ASST_ICON_DY = -0.02
BUBBLE_BX = _BLOCK_X0 + ICON_SIZE + ICON_GAP
USER_ICON_X = BUBBLE_BX + BUBBLE_BW + ICON_GAP + ICON_SIZE / 2

_FIXED_CROP_BOX = None

QWEN_LOGO = RES / "qwen_logo.png"
USER_LOGO = RES / "user_avatar.png"

MSG = {
    "u1": (
        "Group of animals — for each visible\n"
        "animal, determine facing direction\n"
        "(head/nose). Number left-to-right:\n"
        "1), 2), 3) ...\n"
        "facing left | right | toward camera\n"
        "| away | slightly left | slightly right"
    ),
    "a1": (
        "1) facing away    2) facing right\n"
        "3) facing left    4) facing left\n"
        "5) facing left    6) toward camera"
    ),
    "u2": (
        "Compose one concise photorealistic prompt\n"
        "for this group (species, per-animal directions\n"
        "left-to-right, camera & scenery).\n"
        "Return only the final prompt."
    ),
    "a2": (
        "Photorealistic portrait shot with OPPO Reno10 Pro: ten sheep, all facing left, "
        "arranged left-to-right in a line along a rural road at night, long exposure light trails glowing softly in the background, "
        "shallow depth of field, cinematic ambient glow, realistic fur textures, calm rural atmosphere."
    ),
}

CHAT_ITEMS = (
    {"role": "user", "key": "u1", "h": 0.198, "with_image": True},
    {"role": "assistant", "key": "a1", "h": 0.088, "family": "monospace"},
    {"role": "user", "key": "u2", "h": 0.125, "with_image": False},
    {"role": "assistant", "key": "a2", "h": 0.224, "text_fn": "prompt_lines", "text_va": "top"},
)


def chat_centers():
    centers = []
    y = 1.0 - MARGIN_TOP
    for item in CHAT_ITEMS:
        h = item["h"]
        y -= h / 2
        centers.append((y, item))
        y -= h / 2 + BUBBLE_GAP
    return centers


def save_fig(fig, path, dpi=120):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.08, facecolor="#fafbfc")
    plt.close(fig)


def draw_animals(ax, x0, y0, w, h, n=3):
    for i, t in enumerate(np.linspace(x0 + 0.08, x0 + w - 0.12, n)):
        cy = y0 + 0.12 + 0.04 * (i % 2)
        ax.add_patch(mpatches.Ellipse((t + 0.04, cy + 0.12), 0.06, 0.09, facecolor="#ed8936", edgecolor="#c05621", lw=1.2))
        ax.add_patch(mpatches.Rectangle((t, cy + 0.04), 0.08, 0.07, facecolor="#ecc94b", edgecolor="#b7791f", lw=1.2))
    ax.add_patch(mpatches.Rectangle((x0 + 0.05, y0 + 0.04), w - 0.1, 0.05, facecolor="#68d391", edgecolor="none"))


def ensure_logos():
    from PIL import Image, ImageDraw, ImageFont

    RES.mkdir(parents=True, exist_ok=True)
    if not QWEN_LOGO.exists():
        img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((4, 4, 124, 124), fill="#6B46C1")
        d.ellipse((10, 10, 118, 118), fill="#805AD5")
        d.ellipse((36, 30, 92, 86), fill="#FAF5FF")
        d.rectangle((62, 52, 92, 92), fill="#6B46C1")
        try:
            font = ImageFont.truetype("arialbd.ttf", 52)
        except OSError:
            font = ImageFont.load_default()
        d.text((46, 24), "Q", fill="#553C9A", font=font)
        img.save(QWEN_LOGO)

    if not USER_LOGO.exists():
        img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse((4, 4, 124, 124), fill="#4299E1")
        d.ellipse((44, 24, 84, 72), fill="white")
        d.pieslice((24, 68, 104, 140), 200, 340, fill="white")
        img.save(USER_LOGO)


def ensure_input_image():
    if INPUT_PATH.exists():
        return
    try:
        from PIL import ImageGrab
        img = ImageGrab.grabclipboard()
        if img is not None:
            img.save(INPUT_PATH)
            print(f"Saved clipboard image to {INPUT_PATH}")
            return
    except Exception:
        pass
    print(f"Warning: place input image at {INPUT_PATH}")


def thumb_inset(ax, x, y, w, h):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004", facecolor="#111", edgecolor="#cbd5e0", lw=1.0,
        transform=ax.transAxes, zorder=3,
    ))
    if INPUT_PATH.exists():
        img = plt.imread(str(INPUT_PATH))
        ax.imshow(img, extent=(x, x + w, y, y + h), transform=ax.transAxes, aspect="auto", zorder=4)
    else:
        draw_animals(ax, x + 0.01, y + 0.01, w - 0.02, h - 0.02, n=3)


def make_controlnet_png():
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fafbfc")
    ax.add_patch(FancyBboxPatch((0.08, 0.2), 0.84, 0.55, boxstyle="round,pad=0.02", facecolor="#faf5ff", edgecolor="#6b46c1", lw=2.2))
    ax.text(0.5, 0.62, "Qwen-Image", ha="center", fontsize=12, fontweight="bold", color="#553c9a")
    ax.text(0.5, 0.48, "ControlNet-Union", ha="center", fontsize=11, fontweight="bold", color="#553c9a")
    ax.text(0.5, 0.34, "geometry +\nocclusion", ha="center", fontsize=9, color="#6b46c1")
    ax.text(0.5, 0.96, "Qwen-ControlNet Synthesis", ha="center", fontsize=11, fontweight="bold", color="#2d3748")
    save_fig(fig, RES / "herd3d_controlnet.png")


def make_output_png():
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fafbfc")
    ax.add_patch(FancyBboxPatch((0.05, 0.08), 0.9, 0.82, boxstyle="round,pad=0.02", facecolor="#fff", edgecolor="#38b2ac", lw=2.2))
    draw_animals(ax, 0.08, 0.12, 0.84, 0.7, n=4)
    ax.text(0.5, 0.22, "SMAL+ · 2D/3D KP · BBox", ha="center", fontsize=8, color="#2c7a7b")
    ax.text(0.5, 0.96, "Synthetic Image + Labels", ha="center", fontsize=11, fontweight="bold", color="#2d3748")
    save_fig(fig, RES / "herd3d_output.png")


def prompt_lines(text):
    return (
        "Photorealistic portrait shot with OPPO Reno10 Pro:\n"
        "ten sheep, all facing left, arranged left-to-right\n"
        "in a line along a rural road at night,\n"
        "long exposure light trails glowing softly\n"
        "in the background, shallow depth of field,\n"
        "cinematic ambient glow, realistic fur textures,\n"
        "calm rural atmosphere."
    )


def crop_box(img, pad=6):
    mask = np.mean(img, axis=2) < 252
    ys, xs = np.where(mask)
    if len(xs) == 0:
        h, w = img.shape[:2]
        return 0, h, 0, w
    y0 = max(0, ys.min() - pad)
    y1 = min(img.shape[0], ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(img.shape[1], xs.max() + pad + 1)
    return y0, y1, x0, x1


def apply_crop(img, box):
    y0, y1, x0, x1 = box
    return img[y0:y1, x0:x1]


def fixed_crop_box():
    global _FIXED_CROP_BOX
    if _FIXED_CROP_BOX is None:
        _FIXED_CROP_BOX = crop_box(_render_chat_image(3, 1.0), pad=6)
    return _FIXED_CROP_BOX


def logo_icon(ax, path, x, y, size=ICON_SIZE, alpha=1.0):
    half = size / 2
    ax.add_patch(Circle(
        (x, y), half * 1.02, facecolor=(1, 1, 1, alpha), edgecolor=(226 / 255, 232 / 255, 240 / 255, alpha),
        linewidth=1.0, transform=ax.transAxes, zorder=8,
    ))
    img = plt.imread(str(path))
    ax.imshow(
        img, extent=(x - half, x + half, y - half, y + half),
        transform=ax.transAxes, zorder=9, alpha=alpha, aspect="equal",
    )


def bubble_text(ax, x, text, alpha=1.0, cy=None, by=None, bh=None, va="center", family="sans-serif"):
    y = (by + bh - BUBBLE_TEXT_PAD) if va == "top" and by is not None and bh is not None else cy
    ax.text(
        x, y, text, ha="left", va=va, fontsize=CHAT_FONT_SIZE, style="italic",
        family=family, color=(45 / 255, 55 / 255, 72 / 255, alpha),
        transform=ax.transAxes, zorder=5, linespacing=CHAT_LINE_SPACING,
    )


def user_bubble(ax, text, cy, bh, alpha=1.0, with_image=False):
    by = cy - bh / 2
    ax.add_patch(FancyBboxPatch(
        (BUBBLE_BX, by), BUBBLE_BW, bh, boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=(1, 1, 1, alpha), edgecolor=(203 / 255, 213 / 255, 224 / 255, alpha),
        linewidth=1.2, transform=ax.transAxes, zorder=2,
    ))
    text_x = BUBBLE_BX + 0.02
    if with_image:
        thumb_w = 0.19
        thumb_inset(ax, BUBBLE_BX + 0.010, by + 0.012, thumb_w, bh - 0.024)
        text_x = BUBBLE_BX + thumb_w + 0.028
    bubble_text(ax, text_x, text, alpha=alpha, cy=cy)
    logo_icon(ax, USER_LOGO, USER_ICON_X, cy, alpha=alpha)


def assistant_bubble(ax, text, cy, bh, alpha=1.0, text_va="center", family="sans-serif"):
    by = cy - bh / 2
    ax.add_patch(FancyBboxPatch(
        (BUBBLE_BX, by), BUBBLE_BW, bh, boxstyle="round,pad=0.01,rounding_size=0.015",
        facecolor=(1, 1, 1, alpha), edgecolor=(203 / 255, 213 / 255, 224 / 255, alpha),
        linewidth=1.2, transform=ax.transAxes, zorder=2,
    ))
    bubble_text(
        ax, BUBBLE_BX + 0.015, text, alpha=alpha,
        cy=cy, by=by, bh=bh, va=text_va, family=family,
    )
    logo_icon(ax, QWEN_LOGO, ASST_ICON_X + ASST_ICON_DX, cy + ASST_ICON_DY, alpha=alpha)


def _draw_chat(ax, stage, progress=1.0):
    def fade_alpha(msg_stage):
        if stage > msg_stage:
            return 1.0
        if stage == msg_stage:
            return min(1.0, max(0.0, progress))
        return 0.0

    for idx, (cy, item) in enumerate(chat_centers()):
        if stage < idx:
            continue
        alpha = fade_alpha(idx)
        text = MSG[item["key"]]
        if item.get("text_fn") == "prompt_lines":
            text = prompt_lines(text)
        if item["role"] == "user":
            user_bubble(ax, text, cy, item["h"], alpha=alpha, with_image=item.get("with_image", False))
        else:
            assistant_bubble(
                ax, text, cy, item["h"], alpha=alpha,
                text_va=item.get("text_va", "center"),
                family=item.get("family", "sans-serif"),
            )


def _render_chat_image(stage, progress=1.0):
    fig, ax = plt.subplots(figsize=(VID_W / 100, VID_H / 100), dpi=120)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_position([0, 0, 1, 1])
    fig.patch.set_facecolor("#ffffff")
    _draw_chat(ax, stage, progress)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return img


def chat_frame(stage, progress=1.0):
    return apply_crop(_render_chat_image(stage, progress), fixed_crop_box())


def pad_frames(frames):
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded = []
    for fr in frames:
        canvas = np.full((max_h, max_w, 3), 255, dtype=fr.dtype)
        canvas[0:fr.shape[0], 0:fr.shape[1]] = fr
        padded.append(canvas)
    return padded


def pick_evenly(files, n=10):
    files = sorted(files)
    if len(files) <= n:
        return files
    return [files[int(i * (len(files) - 1) / (n - 1))] for i in range(n)]


def save_video(frames, out, fps=8):
    try:
        import imageio.v2 as imageio
        imageio.mimsave(out, frames, fps=fps, codec="libx264", quality=8,
                        macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p"])
    except Exception:
        import cv2
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for fr in frames:
            writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        writer.release()


def make_synthetic_video(src_dir=SYNTH_SRC, out=SYNTH_OUT, n=10, size=512, hold_s=1.0, fps=8):
    from PIL import Image

    src = Path(src_dir)
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = [f for f in src.iterdir() if f.suffix.lower() in exts]
    if not files:
        raise FileNotFoundError(f"No images found in {src}")

    picks = pick_evenly(files, n)
    hold_frames = max(1, int(round(hold_s * fps)))
    frames = []
    for path in picks:
        img = Image.open(path).convert("RGB")
        if img.size != (size, size):
            img = img.resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(img)
        frames.extend([arr] * hold_frames)

    save_video(frames, out, fps=fps)
    print(f"Wrote {out} ({len(frames)} frames, {size}x{size}, {n} images)")
    print("Selected:", ", ".join(p.name for p in picks))


def make_qwen_video():
    fade_frames = 5
    hold_frames = 14
    frames = []
    for stage in range(4):
        for f in range(fade_frames):
            frames.append(chat_frame(stage, (f + 1) / fade_frames))
        for _ in range(hold_frames):
            frames.append(chat_frame(stage, 1.0))
    frames = pad_frames(frames)

    out = RES / "herd3d_qwen.mp4"
    save_video(frames, out, fps=5)
    print(f"Wrote {out} ({len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]})")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to RGB/normal-map input image")
    args = parser.parse_args()

    RES.mkdir(parents=True, exist_ok=True)
    if args.input:
        src = Path(args.input)
        if src.exists():
            import shutil
            shutil.copy2(src, INPUT_PATH)
            print(f"Copied {src} -> {INPUT_PATH}")
    ensure_logos()
    ensure_input_image()
    make_controlnet_png()
    make_output_png()
    make_qwen_video()
    make_synthetic_video()


if __name__ == "__main__":
    main()
