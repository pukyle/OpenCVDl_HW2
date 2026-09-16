"""Regenerate the figures in docs/figures/.

    python3 scripts/make_figures.py          # run from the repository root

Every figure that shows data is built from assets already in this repository
(HW1/Q1_TestData, HW2/Q2_inference_img) by replaying the exact preprocessing
steps in HW1/main.py and HW2/main.py -- nothing is mocked up.
Each figure is rendered twice, light and dark, so the README can serve the
matching asset with <picture media="(prefers-color-scheme: ...)">.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from theme import MODES, apply, save

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)


# --------------------------------------------------------------- figure 1
def fig_q1_preprocessing(c, mode):
    """The Q1 inference preprocessing chain, replayed on a real test image."""
    path = os.path.join(ROOT, "HW1", "Q1_TestData", "test_2.png")
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)          # as main.py loads it
    inverted = cv2.bitwise_not(gray)                        # the critical step
    resized = cv2.resize(inverted, (32, 32), interpolation=cv2.INTER_LINEAR)
    normed = resized.astype(np.float32) / 255.0
    normed = (normed - 0.5) / 0.5                           # Normalize((0.5,), (0.5,))

    panels = [
        (gray,     f"loaded  {gray.shape[1]}×{gray.shape[0]}",
         "cv2.imread(..., IMREAD_GRAYSCALE)", "black digit on white"),
        (inverted, f"inverted  {inverted.shape[1]}×{inverted.shape[0]}",
         "cv2.bitwise_not(img)", "now matches MNIST polarity"),
        (resized,  "resized  32×32",
         "transforms.Resize((32, 32))", "LeNet-5 input size"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.3))
    for ax, (img, title, code, note) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(c["grid"]); s.set_linewidth(1.2)
        ax.set_title(title, color=c["text"], fontsize=11, pad=8)
        ax.text(0.5, -0.09, code, transform=ax.transAxes, ha="center", va="top",
                color=c["muted"], fontsize=8.6, family="monospace")
        ax.text(0.5, -0.20, note, transform=ax.transAxes, ha="center", va="top",
                color=c["faint"], fontsize=9)

    for i in (0, 1):
        x = 0.365 + i * 0.278
        fig.patches.append(FancyArrowPatch(
            (x, 0.60), (x + 0.045, 0.60), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=15, color=c["faint"], lw=1.4))

    fig.suptitle("Q1 inference preprocessing, replayed on HW1/Q1_TestData/test_2.png",
                 x=0.012, ha="left", color=c["text"], fontsize=12.5, weight="600")
    fig.text(0.012, 0.015,
             "A final Normalize((0.5,), (0.5,)) maps [0, 255] to [−1, 1]. "
             "The inversion is the step that matters: the test digits are\n"
             "black-on-white, MNIST is white-on-black, and without it every "
             "prediction is made on a photographic negative of the training data.",
             color=c["muted"], fontsize=9.3, va="bottom")
    fig.tight_layout(rect=[0, 0.14, 1, 0.91])
    save(fig, os.path.join(OUT, "fig1-q1-preprocessing"), mode)


# --------------------------------------------------------------- figure 2
def fig_q1_testdata(c, mode):
    d = os.path.join(ROOT, "HW1", "Q1_TestData")
    fig, axes = plt.subplots(1, 10, figsize=(9.2, 1.75))
    for i, ax in enumerate(axes):
        img = cv2.imread(os.path.join(d, f"test_{i}.png"), cv2.IMREAD_GRAYSCALE)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(c["grid"]); s.set_linewidth(1.0)
        ax.set_xlabel(f"test_{i}", color=c["faint"], fontsize=9, labelpad=4)
    fig.suptitle("Q1 evaluation set — ten 28×28 digits, black on white",
                 x=0.012, ha="left", color=c["text"], fontsize=12, weight="600")
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, os.path.join(OUT, "fig2-q1-testdata"), mode)


# --------------------------------------------------------------- figure 3
def fig_resnet_mod(c, mode):
    """Original vs. CIFAR-adapted ResNet-18 stem, with feature-map sizes."""
    W = [1.55, 1.95, 1.75, 1.85, 1.65, 1.45, 1.65, 1.35]
    fig, ax = plt.subplots(figsize=(9.4, 4.0))
    ax.set_xlim(-0.1, sum(W) + 0.3); ax.set_ylim(-2.7, 6.0); ax.axis("off")

    rows = [
        ("original stem, sized for 224×224 ImageNet input", 3.45, c["faint"],
         ["input\n3×32×32", "conv 7×7\nstride 2", "64×16×16",
          "maxpool\nstride 2", "64×8×8", "layers\n1–4",
          "512×1×1", "fc\n→1000"]),
        ("adapted stem, sized for 32×32 CIFAR-10 input", 0.55, c["s3"],
         ["input\n3×32×32", "conv 3×3\nstride 1", "64×32×32",
          "Identity\n(removed)", "64×32×32", "layers\n1–4",
          "512×4×4", "fc\n→10"]),
    ]
    changed = {1, 3, 7}
    centres = {}
    for label, y, col, blocks in rows:
        ax.text(0.0, y + 1.45, label, color=c["text"], fontsize=10.8, weight="600")
        x = 0.0
        for i, txt in enumerate(blocks):
            w = W[i]
            face = col if i in changed else c["panel"]
            fg = "#ffffff" if i in changed else c["muted"]
            ax.add_patch(FancyBboxPatch((x, y), w - 0.18, 1.2,
                                        boxstyle="round,pad=0,rounding_size=0.09",
                                        facecolor=face, edgecolor="none"))
            ax.text(x + (w - 0.18) / 2, y + 0.6, txt, ha="center", va="center",
                    color=fg, fontsize=8.9, weight="600" if i in changed else "400",
                    linespacing=1.4)
            centres.setdefault(i, []).append((x + (w - 0.18) / 2, y))
            if i < len(blocks) - 1:
                ax.add_patch(FancyArrowPatch(
                    (x + w - 0.17, y + 0.6), (x + w - 0.01, y + 0.6),
                    arrowstyle="-|>", mutation_scale=9, color=c["faint"], lw=1.1))
            x += w
    for i in changed:
        cx = centres[i][0][0]
        ax.plot([cx, cx], [2.32, 3.4], color=c["s3"], lw=1.3, ls=":", zorder=0)

    ax.text(0.0, -0.55,
            "Three changes, one purpose: a 32×32 image cannot afford the stem's "
            "4× downsample. The 7×7/stride-2 convolution becomes 3×3/stride-1\n"
            "and the max-pool becomes an identity, so the residual stages start at "
            "full resolution and finish at 4×4 rather than 1×1. The classifier\n"
            "head is retargeted from 1000 ImageNet classes to 10.",
            color=c["muted"], fontsize=9.4, va="top", linespacing=1.6)
    ax.text(0.0, 5.75, "ResNet-18 adapted for CIFAR-10",
            color=c["text"], fontsize=13, weight="600")
    save(fig, os.path.join(OUT, "fig3-resnet-mod"), mode)


# --------------------------------------------------------------- figure 4
def fig_q2_images(c, mode):
    d = os.path.join(ROOT, "HW2", "Q2_inference_img")
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    names = classes + ["chair"]
    fig, axes = plt.subplots(2, 6, figsize=(9.2, 3.6))
    flat = axes.ravel()
    for ax, n in zip(flat, names):
        img = cv2.cvtColor(cv2.imread(os.path.join(d, f"{n}.jpg")), cv2.COLOR_BGR2RGB)
        ax.imshow(img, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ood = (n == "chair")
        for s in ax.spines.values():
            s.set_edgecolor(c["s8"] if ood else c["grid"])
            s.set_linewidth(2.2 if ood else 1.0)
        ax.set_xlabel(n, color=c["s8"] if ood else c["faint"],
                      fontsize=9, weight="600" if ood else "400", labelpad=4)
    flat[-1].axis("off")
    flat[-1].text(0.02, 0.55,
                  "chair.jpg is the\nout-of-distribution\ncase: 640×640, and\n"
                  "none of the ten\nclasses. It is what\nthe threshold rule\nhas to catch.",
                  transform=flat[-1].transAxes, va="center", ha="left",
                  color=c["s8"], fontsize=9, linespacing=1.55)
    fig.suptitle("Q2 inference set — the ten CIFAR-10 classes plus one "
                 "out-of-distribution image",
                 x=0.012, ha="left", color=c["text"], fontsize=12, weight="600")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    save(fig, os.path.join(OUT, "fig4-q2-images"), mode)


for mode, c in MODES:
    apply(c)
    fig_q1_preprocessing(c, mode)
    fig_q1_testdata(c, mode)
    fig_resnet_mod(c, mode)
    fig_q2_images(c, mode)
print("figures written to", OUT)
