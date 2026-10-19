"""Figures for the gap-filling sections (bias-variance, batch norm, CNN architectures, detection, PEFT,
VLM interpretation, generative extras ...). Run after make_figures.py:
python Lernen/_figures/make_figures_gaps.py
Same colour code: blue = input/data, orange = learned/trained, green = output, red = true label, purple = setting you choose.
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

from make_figures import (BLUE, ORANGE, GREEN, RED, PURPLE, GREY, INK, SOFT, save, box, arrow, canvas)

LBLUE, LORANGE, LGREEN, LRED, LPURPLE, LGREY = "#eff6ff", "#fff7ed", "#f0fdf4", "#fef2f2", "#f3e8ff", "#e2e8f0"


def thumb(ax, img, x, y, w, h, cmap="gray", vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, extent=(x, x + w, y, y + h), vmin=vmin, vmax=vmax, zorder=3)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=SOFT, lw=1, zorder=4))


# ---------------------------------------------------------------- 01 basics
def fig_bias_variance():
    rng = np.random.RandomState(3)
    fig = plt.figure(figsize=(12.5, 4.3))
    gs = fig.add_gridspec(2, 3, width_ratios=[2.3, 1, 1], wspace=0.15, hspace=0.35)
    ax = fig.add_subplot(gs[:, 0])
    c = np.linspace(0, 1, 300)
    bias2 = 0.85 * np.exp(-4.5 * c) + 0.02
    var = 0.03 + 0.85 * c ** 3
    noise = np.full_like(c, 0.12)
    total = bias2 + var + noise
    ax.plot(c, bias2, color=BLUE, lw=2.4, label="bias²")
    ax.plot(c, var, color=ORANGE, lw=2.4, label="variance")
    ax.plot(c, noise, color=GREY, lw=1.8, ls=":", label="noise (cannot be removed)")
    ax.plot(c, total, color=INK, lw=2.8, label="expected test error = sum")
    best = c[np.argmin(total)]
    ax.axvline(best, color=PURPLE, ls="--", lw=1.4)
    ax.text(best + 0.015, 1.02, "best\ncapacity", color=PURPLE, fontsize=9.5)
    ax.text(0.03, 0.2, "underfitting\n(high bias)", color=BLUE, fontsize=10)
    ax.text(0.76, 0.2, "overfitting\n(high variance)", color=ORANGE, fontsize=10)
    ax.set_xlabel("model capacity →  (more layers, more neurons, less regularisation)")
    ax.set_ylabel("error"); ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(0, 1.5)
    ax.legend(fontsize=9, loc="upper center", ncol=2)
    ax.set_title("The bias–variance trade-off")
    specs = [(0, 1, (0, 0), 0.08, "low bias, low variance"), (0, 2, (0, 0), 0.32, "low bias, high variance"),
             (1, 1, (0.5, 0.35), 0.08, "high bias, low variance"), (1, 2, (0.5, 0.35), 0.32, "high bias, high variance")]
    for r, col, off, spread, title in specs:
        a = fig.add_subplot(gs[r, col])
        for rad, fc in [(1.0, "#e2e8f0"), (0.66, "#cbd5e1"), (0.33, "#94a3b8")]:
            a.add_patch(Circle((0, 0), rad, color=fc))
        a.scatter([0], [0], marker="+", s=80, color=INK, zorder=4)
        pts = rng.randn(10, 2) * spread + np.array(off)
        a.scatter(*pts.T, s=20, color=RED, zorder=5)
        a.set_xlim(-1.1, 1.1); a.set_ylim(-1.1, 1.1); a.set_aspect("equal"); a.axis("off")
        a.set_title(title, fontsize=9.5)
    fig.text(0.79, 0.005, "centre = the true answer · each dot = a model trained on a different training set",
             ha="center", fontsize=8.5, color=SOFT)
    save(fig, "01_bias_variance")


def fig_initialisation():
    torch.manual_seed(0)
    n = 256
    x = torch.randn(2000, n)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for std, name, col in [(0.01, "too small (std 0.01): signal dies → vanishing gradients", BLUE),
                           (0.2, "too large (std 0.2): signal explodes → exploding gradients", RED),
                           (np.sqrt(2 / n), "He init (std = √(2/n) ≈ 0.09): signal stays stable", GREEN)]:
        h = x; stds = [h.std().item()]
        for _ in range(10):
            h = torch.relu(h @ (torch.randn(n, n) * std)); stds.append(h.std().item())
        ax.semilogy(range(11), stds, "-o", color=col, lw=2, ms=4, label=name)
    ax.set_xlabel("layer"); ax.set_ylabel("spread (std) of the activations\n(log scale)")
    ax.set_title("10 ReLU layers, 256 neurons each: only the right scale keeps the signal alive")
    ax.legend(fontsize=8.8, loc="lower left")
    save(fig, "01_initialisation")


def fig_batchnorm():
    rng = np.random.RandomState(1)
    z = rng.normal(6, 3, 64)
    zhat = (z - z.mean()) / np.sqrt(z.var() + 1e-5)
    zt = 1.5 * zhat + 0.5
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.0), sharey=True)
    bins = np.linspace(-6, 16, 34)
    for ax, v, col, title in [(axes[0], z, GREY, f"1. raw z of one neuron, batch of 64\nmean {z.mean():.1f}, std {z.std():.1f}"),
                              (axes[1], zhat, BLUE, "2. normalise: subtract batch mean,\ndivide by batch std → mean 0, std 1"),
                              (axes[2], zt, GREEN, "3. scale and shift: γ = 1.5, β = 0.5\n(γ and β are learned)")]:
        ax.hist(v, bins=bins, color=col, alpha=0.85)
        ax.axvline(v.mean(), color=INK, lw=1.2, ls="--")
        ax.set_title(title, fontsize=10.5); ax.set_xlabel("value")
    axes[0].set_ylabel("count")
    save(fig, "01_batchnorm")


# ---------------------------------------------------------------- 07 evaluation
def fig_time_series_cv():
    fig, ax = canvas(11, 5.0, (0, 11), (-0.3, 5.1))
    ax.text(0.1, 4.75, "Shuffled k-fold: wrong for time series", weight="bold", fontsize=10.5, color=RED)
    order = ["train"] * 10; order[3] = "val"; order[4] = "val"
    for t in range(10):
        val = order[t] == "val"
        ax.add_patch(Rectangle((1.6 + t * 0.9, 4.0), 0.82, 0.5, color=ORANGE if val else "#dbeafe"))
        ax.text(1.6 + t * 0.9 + 0.41, 4.25, f"{2015 + t}", ha="center", va="center", fontsize=8)
    ax.text(1.4, 4.25, "one fold", ha="right", va="center", fontsize=9)
    ax.annotate("", xy=(5.3, 3.95), xytext=(8.0, 3.95), arrowprops=dict(arrowstyle="-|>", color=RED, connectionstyle="arc3,rad=-0.3"))
    ax.text(6.6, 3.2, "the model trains on 2020–2024 and is tested on 2018–2019: it has seen the future", ha="center", fontsize=8.8, color=RED)
    ax.text(0.1, 2.75, "Forward chaining (expanding window): always train on the past, test on the next period", weight="bold", fontsize=10.5, color=GREEN)
    for r in range(4):
        for t in range(10):
            k = r + 5
            col = "#dbeafe" if t < k else (ORANGE if t == k else "#f1f5f9")
            ax.add_patch(Rectangle((1.6 + t * 0.9, 2.1 - r * 0.55), 0.82, 0.45, color=col))
        ax.text(1.4, 2.32 - r * 0.55, f"fold {r + 1}", ha="right", va="center", fontsize=9)
    arrow(ax, 1.6, -0.15, 10.5, -0.15, color=SOFT); ax.text(10.55, -0.2, "time", fontsize=9, color=SOFT)
    ax.text(5.8, 0.05, "blue = train  ·  orange = test  ·  grey = future, not used yet", ha="center", fontsize=9, color=SOFT)
    save(fig, "07_time_series_cv")


def fig_group_split():
    rng = np.random.RandomState(4)
    fig, ax = canvas(12, 3.9, (0, 12), (-0.6, 3.9))
    cols = {"train": "#dbeafe", "val": "#fde68a", "test": ORANGE}
    for panel, (x0, title, tcol) in enumerate([(0.9, "Random split by clip:\nthe same person is in train AND test", RED),
                                               (7.0, "Split by person:\ntest people are never seen in training", GREEN)]):
        ax.text(x0 - 0.8, 3.4, title, fontsize=10, weight="bold", color=tcol)
        for p, name in enumerate("ABCDE"):
            ax.text(x0 - 0.15, 2.95 - p * 0.62, f"person {name}", ha="right", va="center", fontsize=9)
            for c in range(6):
                if panel == 0:
                    split = rng.choice(["train", "train", "train", "val", "test"])
                else:
                    split = "train" if p < 3 else ("val" if p == 3 else "test")
                ax.add_patch(Rectangle((x0 + c * 0.55, 2.75 - p * 0.62), 0.48, 0.42, color=cols[split]))
    for i, (k, v) in enumerate(cols.items()):
        ax.add_patch(Rectangle((3.6 + i * 1.6, -0.5), 0.35, 0.3, color=v)); ax.text(4.05 + i * 1.6, -0.35, k, va="center", fontsize=9)
    save(fig, "07_group_split")


def fig_roc():
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))

    def roc(sep):
        neg = rng.normal(0, 1, 3000); pos = rng.normal(sep, 1, 3000)
        th = np.linspace(-5, 8, 400)
        tpr = np.array([(pos >= t).mean() for t in th]); fpr = np.array([(neg >= t).mean() for t in th])
        return fpr, tpr, np.trapezoid(tpr[::-1], fpr[::-1]), th, pos, neg

    for sep, col, name in [(2.2, GREEN, "good model"), (0.8, ORANGE, "weak model")]:
        fpr, tpr, auc, th, pos, neg = roc(sep)
        ax.plot(fpr, tpr, color=col, lw=2.4, label=f"{name}: AUC = {auc:.2f}")
        if sep > 2:
            ax.fill_between(fpr, tpr, color=col, alpha=0.08)
            t = 1.1; ax.scatter([(neg >= t).mean()], [(pos >= t).mean()], color=PURPLE, zorder=5, s=40)
            ax.text((neg >= t).mean() + 0.03, (pos >= t).mean() - 0.07, "one threshold\n= one point", color=PURPLE, fontsize=9)
    ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.6, label="random guessing: AUC = 0.50")
    ax.set_xlabel("false positive rate = FP / (FP + TN)"); ax.set_ylabel("true positive rate (recall) = TP / (TP + FN)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="lower right"); ax.set_title("ROC curve: sweep the threshold from high to low")
    save(fig, "07_roc")


def fig_smote():
    rng = np.random.RandomState(2)
    maj = rng.randn(150, 2) * 0.9
    mino = rng.randn(8, 2) * 0.6 + np.array([3.0, 2.4])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.scatter(*maj.T, s=12, color=BLUE, alpha=0.6, label="majority class (many)")
    ax.scatter(*mino.T, s=40, color=RED, label="minority class (few)", zorder=4)
    for i in [0, 3, 5]:
        d = np.linalg.norm(mino - mino[i], axis=1); nn = np.argsort(d)[1:3]
        for j in nn:
            ax.plot(*np.c_[mino[i], mino[j]], color=PURPLE, lw=1, ls="--")
            lam = rng.uniform(0.2, 0.8); new = mino[i] + lam * (mino[j] - mino[i])
            ax.scatter(*new, marker="*", s=130, color=PURPLE, zorder=5)
    ax.scatter([], [], marker="*", s=130, color=PURPLE, label="new synthetic minority points")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("SMOTE: new points on the lines between\na minority example and its minority neighbours", fontsize=11)
    save(fig, "07_smote")


# ---------------------------------------------------------------- 10 CNNs
def fig_architectures():
    names = ["AlexNet\n2012", "ZFNet\n2013", "VGG\n2014", "GoogLeNet\n2014", "ResNet\n2015"]
    err = [16.4, 11.7, 7.3, 6.7, 3.6]; layers = [8, 8, 19, 22, 152]
    params = ["61 M", "", "138 M", "5 M", "60 M"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    axes[0].bar(names, err, color=[GREY, GREY, BLUE, ORANGE, GREEN])
    for i, v in enumerate(err): axes[0].text(i, v + 0.3, f"{v}%", ha="center", fontsize=9.5)
    axes[0].set_title("ImageNet top-5 error (lower is better)"); axes[0].set_ylabel("error in %")
    names2 = [n + (f"\n{p}" if p else "\n") for n, p in zip(names, params)]
    axes[1].bar(names2, layers, color=[GREY, GREY, BLUE, ORANGE, GREEN])
    for i, v in enumerate(layers):
        axes[1].text(i, v + 3, f"{v} layers", ha="center", fontsize=9)
    axes[1].set_title("Depth (bars) and number of weights (under the name)"); axes[1].set_ylim(0, 190)
    for a in axes: a.tick_params(axis="x", labelsize=9)
    save(fig, "10_architectures")


def fig_inception():
    fig, ax = canvas(12, 5.6, (0, 12), (0, 5.8))
    box(ax, 0.3, 0.2, 8.6, 0.6, "previous layer: 28 × 28 × 256", BLUE, fill=LBLUE, fs=10)
    xs = [0.4, 2.6, 4.8, 7.0]
    w = 1.9
    box(ax, xs[0], 2.7, w, 0.7, "1×1 conv, 128", ORANGE, fill=LORANGE, fs=9.5)
    box(ax, xs[1], 1.4, w, 0.7, "1×1 conv, 64", PURPLE, fill=LPURPLE, fs=9.5)
    box(ax, xs[1], 2.7, w, 0.7, "3×3 conv, 192", ORANGE, fill=LORANGE, fs=9.5)
    box(ax, xs[2], 1.4, w, 0.7, "1×1 conv, 64", PURPLE, fill=LPURPLE, fs=9.5)
    box(ax, xs[2], 2.7, w, 0.7, "5×5 conv, 96", ORANGE, fill=LORANGE, fs=9.5)
    box(ax, xs[3], 1.4, w, 0.7, "3×3 max pool", GREY, fill=LGREY, fs=9.5)
    box(ax, xs[3], 2.7, w, 0.7, "1×1 conv, 64", PURPLE, fill=LPURPLE, fs=9.5)
    box(ax, 0.3, 4.6, 8.6, 0.8, "concatenate along channels: 28 × 28 × (128 + 192 + 96 + 64) = 28 × 28 × 480", GREEN, fill=LGREEN, fs=9.5)
    for x in xs:
        arrow(ax, x + w / 2, 0.8, x + w / 2, 1.4 if x != xs[0] else 2.7)
        if x != xs[0]: arrow(ax, x + w / 2, 2.1, x + w / 2, 2.7)
        arrow(ax, x + w / 2, 3.4, x + w / 2, 4.6)
    ax.text(9.2, 3.9, "cost of the 3×3 path:", fontsize=9.5, weight="bold")
    ax.text(9.2, 3.25, "without the 1×1:\n3·3·256·192\n= 442,368 weights", fontsize=9, color=RED)
    ax.text(9.2, 2.05, "with a 1×1 to 64 first:\n1·1·256·64 + 3·3·64·192\n= 126,976 weights", fontsize=9, color=GREEN)
    ax.text(9.2, 1.2, "purple = 1×1 'bottleneck'\nconvs that cut channels", fontsize=9, color=PURPLE)
    save(fig, "10_inception")


def fig_resnet():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    it = np.linspace(0, 6, 200)
    axes[0].plot(it, 6 + 14 * np.exp(-it / 1.3), color=BLUE, lw=2.3, label="20-layer plain net")
    axes[0].plot(it, 9 + 13 * np.exp(-it / 1.6), color=RED, lw=2.3, label="56-layer plain net")
    axes[0].set_xlabel("training iterations"); axes[0].set_ylabel("TRAINING error (%)")
    axes[0].set_title("Plain nets: the deeper one is worse even on training data")
    axes[0].legend(fontsize=9); axes[0].set_xticks([])
    axes[0].text(3.2, 15, "not overfitting:\nit is an optimisation problem", fontsize=9, color=SOFT)
    a = axes[1]; a.set_xlim(0, 10); a.set_ylim(0, 4.2); a.axis("off")
    box(a, 0.2, 1.6, 1.2, 0.8, "x", BLUE, fill=LBLUE)
    box(a, 2.2, 1.6, 1.8, 0.8, "3×3 conv\n+ ReLU", ORANGE, fill=LORANGE, fs=9)
    box(a, 4.6, 1.6, 1.6, 0.8, "3×3 conv", ORANGE, fill=LORANGE, fs=9)
    box(a, 6.7, 1.7, 0.6, 0.6, "+", INK, fs=14)
    box(a, 7.7, 1.6, 2.2, 0.8, "ReLU → H(x)", GREEN, fill=LGREEN, fs=9)
    arrow(a, 1.4, 2.0, 2.2, 2.0); arrow(a, 4.0, 2.0, 4.6, 2.0); arrow(a, 6.2, 2.0, 6.7, 2.0); arrow(a, 7.3, 2.0, 7.7, 2.0)
    a.annotate("", xy=(7.0, 2.3), xytext=(0.8, 2.4), arrowprops=dict(arrowstyle="-|>", color=PURPLE, lw=1.8, connectionstyle="arc3,rad=-0.35"))
    a.text(3.9, 3.85, "skip connection: identity x", color=PURPLE, fontsize=9.5, ha="center")
    a.text(3.4, 1.15, "F(x): the layers learn only the change", color=ORANGE, fontsize=9.5, ha="center")
    a.text(5.0, 0.35, "H(x) = F(x) + x.  If nothing should change, F(x) = 0 is easy to learn.", ha="center", fontsize=9.5)
    a.set_title("Residual block")
    save(fig, "10_resnet")


def fig_detection():
    fig, ax = canvas(12.5, 6.4, (0, 12.5), (0, 6.6))
    rows = [(5.0, "One object:\nclassification\n+ localisation"), (2.9, "Two-stage\n(R-CNN family)"), (0.8, "One-stage\n(YOLO, SSD)")]
    for y, label in rows:
        ax.text(0.05, y + 0.45, label, fontsize=9.5, weight="bold", va="center")
        box(ax, 1.6, y, 1.2, 0.9, "image", BLUE, fill=LBLUE, fs=9)
    y = 5.0
    box(ax, 3.4, y, 1.5, 0.9, "CNN\nbackbone", ORANGE, fill=LORANGE, fs=9)
    box(ax, 5.6, y + 0.55, 3.6, 0.6, "class head: 'dog' (softmax + CE)", GREEN, fill=LGREEN, fs=9)
    box(ax, 5.6, y - 0.25, 3.6, 0.6, "box head: x, y, w, h (regression)", GREEN, fill=LGREEN, fs=9)
    arrow(ax, 2.8, y + 0.45, 3.4, y + 0.45); arrow(ax, 4.9, y + 0.6, 5.6, y + 0.85); arrow(ax, 4.9, y + 0.3, 5.6, y + 0.05)
    ax.text(9.5, y + 0.3, "loss = CE + λ · box loss", fontsize=9.5, color=PURPLE)
    y = 2.9
    box(ax, 3.4, y, 2.0, 0.9, "region proposals\n(~2,000 boxes)", PURPLE, fill=LPURPLE, fs=9)
    box(ax, 6.0, y, 2.0, 0.9, "CNN features\nper proposal", ORANGE, fill=LORANGE, fs=9)
    box(ax, 8.6, y, 2.0, 0.9, "class + refined\nbox per proposal", GREEN, fill=LGREEN, fs=9)
    box(ax, 11.1, y, 1.2, 0.9, "NMS", GREY, fill=LGREY, fs=9)
    for x1, x2 in [(2.8, 3.4), (5.4, 6.0), (8.0, 8.6), (10.6, 11.1)]: arrow(ax, x1, y + 0.45, x2, y + 0.45)
    ax.text(3.4, y - 0.35, "R-CNN: classic algorithm proposes boxes  ·  Faster R-CNN: a learned Region Proposal Network  →  accurate, slower",
            fontsize=8.6, color=SOFT)
    y = 0.8
    box(ax, 3.4, y, 2.0, 0.9, "CNN\n(one pass)", ORANGE, fill=LORANGE, fs=9)
    box(ax, 6.0, y, 4.6, 0.9, "grid: every cell predicts boxes\n+ class scores at once", GREEN, fill=LGREEN, fs=9)
    box(ax, 11.1, y, 1.2, 0.9, "NMS", GREY, fill=LGREY, fs=9)
    for x1, x2 in [(2.8, 3.4), (5.4, 6.0), (10.6, 11.1)]: arrow(ax, x1, y + 0.45, x2, y + 0.45)
    ax.text(3.4, y - 0.35, "no separate proposal step  →  fast enough for real time", fontsize=8.6, color=SOFT)
    save(fig, "10_detection")


def fig_nms():
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    a = axes[0]
    a.add_patch(Rectangle((1, 1), 4, 3, fill=True, color=BLUE, alpha=0.2)); a.add_patch(Rectangle((1, 1), 4, 3, fill=False, ec=BLUE, lw=2))
    a.add_patch(Rectangle((3, 2), 4, 3, fill=True, color=ORANGE, alpha=0.2)); a.add_patch(Rectangle((3, 2), 4, 3, fill=False, ec=ORANGE, lw=2))
    a.add_patch(Rectangle((3, 2), 2, 2, fill=False, hatch="///", ec=INK, lw=0))
    a.text(1.2, 3.6, "box A", color=BLUE, fontsize=9); a.text(5.6, 4.6, "box B", color=ORANGE, fontsize=9)
    a.text(4.0, 0.2, "IoU = overlap / union\n= 4 / (12 + 12 − 4) = 0.2", ha="center", fontsize=9.5)
    a.set_xlim(0, 8); a.set_ylim(-0.6, 5.6); a.set_aspect("equal"); a.axis("off")
    a.set_title("Intersection over Union (IoU)")
    # (box, score, kept?, label position)
    # IoU with the kept box of the same object: 0.85 → 0.64, 0.71 → 0.65, 0.62 → 0.61 (all above 0.5, so removed)
    boxes = [((1.0, 1.0, 3.2, 2.6), 0.92, True, (0.3, 3.65)), ((1.4, 1.4, 3.0, 2.5), 0.85, False, (3.85, 3.98)),
             ((0.7, 0.8, 3.0, 2.5), 0.71, False, (0.2, 0.45)),
             ((5.0, 1.6, 2.4, 2.0), 0.80, True, (5.05, 3.68)), ((5.3, 1.3, 2.3, 2.0), 0.62, False, (7.65, 1.3))]
    for k, a in enumerate(axes[1:]):
        a.add_patch(plt.matplotlib.patches.Ellipse((2.6, 2.3), 2.6, 2.0, color="#cbd5e1"))
        a.add_patch(plt.matplotlib.patches.Ellipse((6.2, 2.6), 1.8, 1.6, color="#cbd5e1"))
        a.text(2.6, 2.3, "dog", ha="center", va="center", fontsize=10, color=SOFT); a.text(6.2, 2.6, "cat", ha="center", va="center", fontsize=10, color=SOFT)
        for (x, y, w, h), s, kept, (lx, ly) in boxes:
            if k == 1 and not kept: continue
            c = (GREEN if kept else RED) if k == 0 else GREEN
            a.add_patch(Rectangle((x, y), w, h, fill=False, ec=c, lw=1.6))
            a.text(lx, ly, f"{s:.2f}", fontsize=9, color=c, weight="bold")
        a.set_xlim(-0.2, 8.2); a.set_ylim(0, 5); a.set_aspect("equal"); a.axis("off")
    axes[1].set_title("Before NMS: several boxes per object\n(red = overlaps a better box, IoU > 0.5)", fontsize=11)
    axes[2].set_title("After NMS: keep the best box\nper object, drop the overlaps", fontsize=11)
    save(fig, "10_nms")


def fig_frame_fusion():
    fig, ax = canvas(12.5, 4.4, (0, 12.5), (0, 4.6))
    ax.text(0.1, 4.25, "Early fusion: stack the frames as channels", weight="bold", fontsize=10.5)
    for i in range(5):
        box(ax, 0.3 + i * 0.12, 1.6 + i * 0.12, 1.0, 1.0, "" if i < 4 else f"5 frames", BLUE, fill=LBLUE, fs=8)
    arrow(ax, 1.9, 2.3, 2.5, 2.3)
    box(ax, 2.5, 1.7, 1.6, 1.2, "one 2-D CNN\n15 input channels\n(5 × RGB)", ORANGE, fill=LORANGE, fs=8.5)
    arrow(ax, 4.1, 2.3, 4.6, 2.3)
    box(ax, 4.6, 1.95, 1.2, 0.7, "13\nclasses", GREEN, fill=LGREEN, fs=8.5)
    ax.text(0.2, 0.7, "+ cheapest (one CNN pass), motion visible in layer 1\n− cannot reuse RGB ImageNet weights directly", fontsize=8.6, color=SOFT)
    ax.text(6.6, 4.25, "Late fusion: same CNN on each frame, then combine", weight="bold", fontsize=10.5)
    for i in range(5):
        y = 3.3 - i * 0.62
        box(ax, 6.7, y, 0.7, 0.5, f"f{i + 1}", BLUE, fill=LBLUE, fs=8)
        arrow(ax, 7.4, y + 0.25, 7.9, y + 0.25)
        box(ax, 7.9, y, 1.3, 0.5, "CNN (shared)", ORANGE, fill=LORANGE, fs=7.5)
        arrow(ax, 9.2, y + 0.25, 9.7, 2.05)
    box(ax, 9.7, 1.7, 1.3, 0.7, "average or\nconcatenate", PURPLE, fill=LPURPLE, fs=8.5)
    arrow(ax, 11.0, 2.05, 11.4, 2.05)
    box(ax, 11.4, 1.7, 1.0, 0.7, "13\nclasses", GREEN, fill=LGREEN, fs=8.5)
    ax.text(6.6, 0.15, "+ reuses ImageNet weights as they are  − 5 CNN passes, order of frames is lost if you average", fontsize=8.6, color=SOFT)
    save(fig, "10_frame_fusion")


def fig_depthwise():
    fig, ax = canvas(12, 3.4, (0, 12), (0, 3.6))
    ax.text(0.1, 3.3, "Standard 3×3 conv, 64 → 128 channels", weight="bold", fontsize=10)
    box(ax, 0.2, 1.3, 1.4, 1.2, "input\n64 channels", BLUE, fill=LBLUE, fs=9)
    arrow(ax, 1.6, 1.9, 2.2, 1.9)
    box(ax, 2.2, 1.3, 2.2, 1.2, "128 filters, each\n3 × 3 × 64", ORANGE, fill=LORANGE, fs=9)
    arrow(ax, 4.4, 1.9, 4.9, 1.9)
    box(ax, 4.9, 1.3, 1.1, 1.2, "128\nch.", GREEN, fill=LGREEN, fs=9)
    ax.text(0.2, 0.6, "3·3·64·128 = 73,728 weights", fontsize=9.5, color=RED)
    ax.text(6.5, 3.3, "Depthwise separable (MobileNet)", weight="bold", fontsize=10)
    box(ax, 6.5, 1.3, 1.2, 1.2, "input\n64 ch.", BLUE, fill=LBLUE, fs=9)
    arrow(ax, 7.7, 1.9, 8.0, 1.9)
    box(ax, 8.0, 1.3, 1.6, 1.2, "depthwise:\none 3 × 3\nper channel", ORANGE, fill=LORANGE, fs=8.5)
    arrow(ax, 9.6, 1.9, 9.9, 1.9)
    box(ax, 9.9, 1.3, 1.3, 1.2, "pointwise:\n1 × 1 conv\n64 → 128", PURPLE, fill=LPURPLE, fs=8.5)
    arrow(ax, 11.2, 1.9, 11.4, 1.9)
    box(ax, 11.4, 1.3, 0.55, 1.2, "128", GREEN, fill=LGREEN, fs=8)
    ax.text(6.5, 0.6, "3·3·64 + 64·128 = 576 + 8,192 = 8,768 weights (≈ 8× fewer)", fontsize=9.5, color=GREEN)
    save(fig, "10_depthwise")


# ---------------------------------------------------------------- 05 pretraining
def toy_scene(n=64):
    img = np.zeros((n, n, 3)); img[:] = [0.82, 0.9, 1.0]
    img[int(n * 0.72):, :] = [0.55, 0.8, 0.5]
    img[int(n * 0.45):int(n * 0.72), int(n * 0.2):int(n * 0.55)] = [0.85, 0.45, 0.35]
    for r in range(int(n * 0.18)):
        y = int(n * 0.45) - r; x0 = int(n * 0.16) + r; x1 = int(n * 0.59) - r
        if x1 > x0: img[y, x0:x1] = [0.5, 0.3, 0.25]
    yy, xx = np.mgrid[:n, :n]
    img[(yy - n * 0.2) ** 2 + (xx - n * 0.78) ** 2 < (n * 0.1) ** 2] = [1.0, 0.8, 0.2]
    return img


def fig_ssl_pretext():
    img = toy_scene()
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    axes[0].imshow(np.rot90(img))
    hole = img.copy(); hole[26:46, 14:38] = 0
    axes[1].imshow(hole)
    frames = []
    for pos in [34, 8, 21]:
        f = np.ones((64, 42, 3)) * 0.95; yy, xx = np.mgrid[:64, :42]
        f[(yy - 32) ** 2 + (xx - pos) ** 2 < 30] = [0.23, 0.51, 0.96]; frames.append(f)
        frames.append(np.ones((64, 3, 3)))
    axes[2].imshow(np.concatenate(frames[:-1], axis=1))
    axes[3].set_xlim(0, 1); axes[3].set_ylim(0, 1)
    axes[3].text(0.5, 0.62, "The [MASK] sat\non the mat.", ha="center", va="center", fontsize=13,
                 bbox=dict(boxstyle="round", fc=LBLUE, ec=BLUE))
    axes[3].text(0.5, 0.22, "→  cat", ha="center", fontsize=13, color=RED, weight="bold")
    for a in axes: a.set_xticks([]); a.set_yticks([])
    axes[3].axis("off")
    titles = ["Rotation: by how much was\nit turned?  →  90°", "Inpainting: fill the\nblacked-out region",
              "Temporal order: are these\nframes in the right order?  →  no", "Masked tokens (BERT):\npredict the hidden word"]
    for a, t in zip(axes, titles):
        p = a.get_position(); fig.text((p.x0 + p.x1) / 2, 0.93, t, ha="center", va="bottom", fontsize=10, weight="bold")
    fig.suptitle("Proxy tasks: the label comes free from the data, so millions of unlabelled examples can be used", fontsize=11, y=1.12)
    save(fig, "05_ssl_pretext")


def fig_scaling_law():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6))
    c = np.logspace(0, 6, 50)
    axes[0].loglog(c, 5 * c ** -0.08, color=GREEN, lw=2.4)
    pts = np.logspace(0.5, 5.5, 6); axes[0].scatter(pts, 5 * pts ** -0.08 * (1 + 0.01 * np.random.RandomState(0).randn(6)), color=INK, zorder=4, s=25)
    axes[0].set_xlabel("compute (or parameters, or data), log scale"); axes[0].set_ylabel("test loss, log scale")
    axes[0].set_title("Scaling law: a straight line on a log–log plot\n= loss falls as a power of scale")
    s = np.logspace(0, 4, 200)
    acc = 25 + 60 / (1 + np.exp(-(np.log10(s) - 2.9) * 6))
    axes[1].semilogx(s, acc, color=ORANGE, lw=2.4); axes[1].axhline(25, color=GREY, ls="--", lw=1.2)
    axes[1].text(1.3, 27.5, "random guessing", color=SOFT, fontsize=9)
    axes[1].set_xlabel("model size, log scale"); axes[1].set_ylabel("accuracy on a hard task (%)")
    axes[1].set_title("Emergent ability: near zero, then a jump\nbeyond some scale")
    fig.text(0.5, -0.1, "Schematic illustration of the shapes described in the Week 9 slides, not real measurements.", ha="center", fontsize=8.5, color=SOFT)
    save(fig, "05_scaling_law")


def fig_peft_methods():
    fig, ax = canvas(13, 4.6, (0, 13), (-0.5, 4.5))
    titles = ["Prompt tuning", "Prefix tuning", "Adapters", "LoRA"]
    for k, t in enumerate(titles):
        x = 0.3 + k * 3.2
        ax.text(x + 1.3, 4.2, t, ha="center", weight="bold", fontsize=10.5)
        for i in range(5):
            ax.add_patch(Rectangle((x + 0.45 + i * 0.4, 0.2), 0.32, 0.32, color=LGREY, ec=GREY))
        if k == 0:
            for i in range(2): ax.add_patch(Rectangle((x - 0.35 + i * 0.4, 0.2), 0.32, 0.32, color=ORANGE))
        box(ax, x + 0.3, 1.0, 2.2, 0.8, "self-attention\nW_Q  W_K  W_V", GREY, fill=LGREY, fs=8.5)
        box(ax, x + 0.3, 2.5, 2.2, 0.7, "feed-forward", GREY, fill=LGREY, fs=8.5)
        arrow(ax, x + 1.4, 0.55, x + 1.4, 1.0); arrow(ax, x + 1.4, 1.8, x + 1.4, 2.5); arrow(ax, x + 1.4, 3.2, x + 1.4, 3.85)
        if k == 1:
            for i in range(2): ax.add_patch(Rectangle((x - 0.3, 1.05 + i * 0.4), 0.45, 0.3, color=ORANGE))
            ax.text(x - 0.08, 1.95, "K, V\nprefix", ha="center", fontsize=7.5, color=ORANGE)
        if k == 2:
            box(ax, x + 0.95, 3.32, 0.9, 0.4, "adapter", ORANGE, fill=LORANGE, fs=7.5)
            box(ax, x + 0.95, 2.02, 0.9, 0.4, "adapter", ORANGE, fill=LORANGE, fs=7.5)
        if k == 3:
            for dx in [0.3, 1.55]:
                box(ax, x + 0.2 + dx, 0.62, 0.45, 0.32, "BA", ORANGE, fill=LORANGE, fs=7)
    notes = ["learned 'virtual tokens'\nadded to the input", "learned key/value vectors\nin every layer",
             "small bottleneck layers\ninserted in each block", "low-rank update on W_Q, W_V;\nmerged after training"]
    for k, n in enumerate(notes):
        ax.text(0.3 + k * 3.2 + 1.3, -0.4, n, ha="center", fontsize=8.5, color=SOFT)
    save(fig, "05_peft_methods")


# ---------------------------------------------------------------- 08 interpretability
def fig_tsne():
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    X, y = load_digits(return_X_y=True)
    p = PCA(2).fit_transform(X)
    t = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for a, z, title in [(axes[0], p, "PCA: a linear projection that keeps the largest spread"),
                        (axes[1], t, "t-SNE: a non-linear map that keeps neighbours together")]:
        sc = a.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=6)
        a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=10.5)
    for d in range(10):
        m = t[y == d].mean(0); axes[1].text(m[0], m[1], str(d), fontsize=13, weight="bold", ha="center", va="center",
                                            bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec="none", alpha=0.8))
    fig.suptitle("1,797 handwritten digits (64 pixel values each) squeezed into 2-D. Colour = true digit (never used by either method).", fontsize=10.5, y=1.0)
    save(fig, "08_tsne")


def fig_rollout():
    toks = ["[CLS]", "not", "bad", "film"]
    A = [np.array([[.40, .20, .20, .20], [.10, .60, .20, .10], [.05, .45, .45, .05], [.05, .25, .40, .30]]),
         np.array([[.30, .10, .20, .40], [.10, .50, .30, .10], [.10, .30, .50, .10], [.10, .20, .30, .40]]),
         np.array([[.20, .05, .15, .60], [.25, .25, .25, .25], [.25, .25, .25, .25], [.10, .20, .30, .40]])]
    Ah = [0.5 * (a + np.eye(4)) for a in A]; Ah = [a / a.sum(1, keepdims=True) for a in Ah]
    R = Ah[2] @ Ah[1] @ Ah[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    a = axes[0]
    for l in range(4):
        for i, t in enumerate(toks):
            a.add_patch(Circle((i * 1.6, l * 1.2), 0.22, color=GREEN if (l == 3 and i == 0) else (BLUE if l == 0 else "#cbd5e1"), zorder=3))
            if l == 0: a.text(i * 1.6, -0.55, t, ha="center", fontsize=10)
    for l in range(3):
        for i in range(4):
            for j in range(4):
                a.plot([i * 1.6, j * 1.6], [(l + 1) * 1.2, l * 1.2], color=ORANGE, lw=6 * Ah[l][i, j], alpha=0.75, zorder=1)
        a.text(5.4, l * 1.2 + 0.6, f"layer {l + 1}", fontsize=9, color=SOFT)
    a.set_xlim(-0.6, 6.3); a.set_ylim(-0.9, 4.0); a.axis("off")
    a.set_title("Attention flows through every layer (line width = weight, incl. residual)", fontsize=10)
    b = axes[1]; x = np.arange(4)
    b.bar(x - 0.2, A[2][0], width=0.4, color=GREY, label="raw attention, last layer only")
    b.bar(x + 0.2, R[0], width=0.4, color=ORANGE, label="attention rollout (all layers)")
    b.set_xticks(x); b.set_xticklabels(toks); b.set_ylim(0, 0.7); b.legend(fontsize=9)
    b.set_title("How much each input token flows into [CLS]", fontsize=10)
    save(fig, "08_rollout")
    print("rollout row CLS:", np.round(R[0], 3), "raw:", A[2][0])


def fig_activation_patching():
    fig, ax = canvas(12.5, 4.6, (0, 12.5), (0, 4.7))
    rows = [(3.6, "clean run", "The Eiffel Tower is in", "Paris", GREEN),
            (2.1, "corrupted run", "The Colosseum is in", "Rome", RED),
            (0.6, "patched run", "The Colosseum is in", "Paris?", PURPLE)]
    for y, name, text, out, col in rows:
        ax.text(0.05, y + 0.3, name, fontsize=9.5, weight="bold", va="center")
        box(ax, 1.5, y, 2.6, 0.6, text, BLUE, fill=LBLUE, fs=9)
        for k in range(5):
            patched = (name == "patched run" and k == 2)
            src = (name == "clean run" and k == 2)
            box(ax, 4.6 + k * 1.15, y, 0.95, 0.6, f"L{k + 1}", ORANGE if (patched or src) else GREY,
                fill=LORANGE if (patched or src) else LGREY, fs=9)
            if k: arrow(ax, 4.6 + k * 1.15 - 0.2, y + 0.3, 4.6 + k * 1.15, y + 0.3)
        arrow(ax, 4.1, y + 0.3, 4.6, y + 0.3)
        arrow(ax, 10.1, y + 0.3, 10.5, y + 0.3)
        box(ax, 10.5, y, 1.2, 0.6, out, col, fill="white", fs=10, weight="bold")
    ax.annotate("", xy=(7.37, 1.2), xytext=(7.37, 3.6), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=2, ls="--",
                                                                        connectionstyle="arc3,rad=0.35"))
    ax.text(8.0, 2.85, "copy layer 3's activation\nfrom the clean run", fontsize=8.8, color=ORANGE)
    ax.text(6.2, 0.1, "If 'Paris' comes back, layer 3 carries the fact: a causal test, not a correlation.", ha="center", fontsize=9.5)
    save(fig, "08_activation_patching")


def fig_vlm():
    fig, ax = canvas(12.5, 4.9, (0, 12.5), (-0.3, 4.8))
    heat = np.zeros((4, 4)); heat[1, 2] = 1.0; heat[2, 2] = 0.55; heat[1, 1] = 0.35; heat[2, 1] = 0.15
    thumb(ax, np.ones((4, 4)) * 0.85, 0.2, 1.4, 2.0, 2.0, vmin=0, vmax=1)
    for i in range(1, 4):
        ax.plot([0.2 + i * 0.5, 0.2 + i * 0.5], [1.4, 3.4], color="white", lw=1.5, zorder=5)
        ax.plot([0.2, 2.2], [1.4 + i * 0.5, 1.4 + i * 0.5], color="white", lw=1.5, zorder=5)
    ax.imshow(heat, cmap="Reds", extent=(0.2, 2.2, 1.4, 3.4), alpha=0.55, vmin=0, vmax=1, zorder=6)
    ax.text(1.2, 1.05, "X-ray as 16 patches", ha="center", fontsize=9)
    box(ax, 2.8, 1.9, 1.7, 1.0, "image encoder\n(ViT, from CLIP)", ORANGE, fill=LORANGE, fs=9)
    box(ax, 5.0, 1.9, 1.3, 1.0, "projector", ORANGE, fill=LORANGE, fs=9)
    box(ax, 6.9, 1.6, 2.2, 1.6, "language model\n(reads image tokens\n+ text tokens)", ORANGE, fill=LORANGE, fs=9)
    box(ax, 6.9, 0.1, 2.2, 0.7, "text: 'What is wrong?'", BLUE, fill=LBLUE, fs=9)
    box(ax, 9.7, 2.0, 2.6, 0.8, "'Fracture of the\nleft radius.'", GREEN, fill=LGREEN, fs=9.5)
    arrow(ax, 2.2, 2.4, 2.8, 2.4); arrow(ax, 4.5, 2.4, 5.0, 2.4); arrow(ax, 6.3, 2.4, 6.9, 2.4)
    arrow(ax, 8.0, 0.8, 8.0, 1.6); arrow(ax, 9.1, 2.4, 9.7, 2.4)
    ax.annotate("", xy=(1.6, 3.45), xytext=(10.4, 2.85), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6, ls="--",
                                                                         connectionstyle="arc3,rad=0.28"))
    ax.text(6.0, 4.45, "grounding: which image patches supported the word 'fracture'?", ha="center", fontsize=9.5, color=RED)
    save(fig, "08_vlm")


# ---------------------------------------------------------------- 11 generative
def fig_autoencoder():
    from sklearn.datasets import load_digits
    d = load_digits(); img = d.images[7] / 16.0
    rng = np.random.RandomState(0); noisy = np.clip(img + rng.normal(0, 0.35, img.shape), 0, 1)
    fig, ax = canvas(12, 5.0, (0, 12), (0, 5.2))

    def row(y, inp, label, target_note):
        thumb(ax, inp, 0.3, y, 1.4, 1.4, cmap="gray_r", vmin=0, vmax=1)
        ax.text(1.0, y - 0.3, label, ha="center", fontsize=9)
        ax.add_patch(Polygon([[2.2, y - 0.1], [3.8, y + 0.45], [3.8, y + 0.95], [2.2, y + 1.5]], closed=True, fc=LORANGE, ec=ORANGE, lw=1.6))
        ax.text(3.0, y + 0.7, "encoder", ha="center", va="center", fontsize=9)
        box(ax, 4.1, y + 0.4, 0.8, 0.6, "z", PURPLE, fill=LPURPLE, fs=11, weight="bold")
        ax.add_patch(Polygon([[5.2, y + 0.45], [6.8, y - 0.1], [6.8, y + 1.5], [5.2, y + 0.95]], closed=True, fc=LORANGE, ec=ORANGE, lw=1.6))
        ax.text(6.0, y + 0.7, "decoder", ha="center", va="center", fontsize=9)
        thumb(ax, img, 7.3, y, 1.4, 1.4, cmap="gray_r", vmin=0, vmax=1)
        ax.text(8.0, y - 0.3, "reconstruction x̂", ha="center", fontsize=9, color=GREEN)
        arrow(ax, 1.7, y + 0.7, 2.2, y + 0.7); arrow(ax, 3.8, y + 0.7, 4.1, y + 0.7)
        arrow(ax, 4.9, y + 0.7, 5.2, y + 0.7); arrow(ax, 6.8, y + 0.7, 7.3, y + 0.7)
        ax.text(9.0, y + 0.7, target_note, fontsize=9, va="center")

    row(3.3, img, "input x", "loss = ‖x − x̂‖²\nplain autoencoder")
    row(0.6, noisy, "noisy input x̃", "loss = ‖x − x̂‖²  (clean x!)\ndenoising autoencoder:\ncannot just copy the input")
    ax.text(4.5, 2.6, "small bottleneck", ha="center", fontsize=8.5, color=PURPLE)
    save(fig, "11_autoencoder")


def fig_anomaly():
    rng = np.random.RandomState(0)
    norm_val = rng.gamma(2.0, 0.05, 600); norm = rng.gamma(2.0, 0.05, 900); abn = rng.gamma(4.0, 0.06, 300) + 0.06
    th = norm_val.mean() + 2 * norm_val.std()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
    bins = np.linspace(0, 0.8, 60)
    axes[0].hist(norm, bins=bins, color=BLUE, alpha=0.7, label="normal beats (test)")
    axes[0].hist(abn, bins=bins, color=RED, alpha=0.6, label="abnormal beats (test)")
    axes[0].axvline(th, color=PURPLE, lw=2, ls="--")
    axes[0].text(th + 0.01, axes[0].get_ylim()[1] * 0.8, f"threshold = mean + 2·std\nof NORMAL validation errors\n= {th:.2f}", color=PURPLE, fontsize=9)
    axes[0].set_xlabel("reconstruction error"); axes[0].set_ylabel("count"); axes[0].legend(fontsize=9, loc="center right")
    axes[0].set_title("Trained on normal data only: anomalies reconstruct badly")
    ts = np.linspace(0, 1, 400)
    tpr = np.array([(abn >= t).mean() for t in ts]); fpr = np.array([(norm >= t).mean() for t in ts])
    auc = np.trapezoid(tpr[::-1], fpr[::-1])
    axes[1].plot(fpr, tpr, color=GREEN, lw=2.4, label=f"autoencoder, AUC = {auc:.2f}")
    axes[1].plot([0, 1], [0, 1], color=GREY, ls="--", lw=1.4, label="random, AUC = 0.50")
    axes[1].scatter([(norm >= th).mean()], [(abn >= th).mean()], color=PURPLE, zorder=5)
    axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate")
    axes[1].set_title("ROC over all thresholds"); axes[1].legend(fontsize=9, loc="lower right"); axes[1].set_aspect("equal")
    save(fig, "11_anomaly")


def fig_cfg():
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    o = np.array([0, 0]); eu = np.array([2.0, 0.5]); ec = np.array([2.3, 1.4]); w = 3
    eg = eu + w * (ec - eu)
    for v, col, lab in [(eu, GREY, "ε(x_t, t, ∅): no condition"), (ec, BLUE, "ε(x_t, t, y): with the prompt y"),
                        (eg, GREEN, f"guided, w = {w}: ε(∅) + w·[ε(y) − ε(∅)]")]:
        ax.annotate("", xy=v, xytext=o, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4))
        ax.scatter([], [], color=col, label=lab)
    ax.plot([eu[0], eg[0]], [eu[1], eg[1]], color=PURPLE, ls="--", lw=1.4)
    ax.text(2.55, 1.0, "difference\nε(y) − ε(∅)", color=PURPLE, fontsize=9)
    ax.scatter(*o, color=INK, zorder=5); ax.text(-0.15, -0.3, "x_t", fontsize=10)
    ax.set_xlim(-0.5, 4); ax.set_ylim(-0.6, 3.8); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    ax.set_title("Classifier-free guidance: push further in the\ndirection the prompt adds", fontsize=11)
    save(fig, "11_cfg")


def fig_text_generation():
    fig, ax = canvas(11.5, 4.6, (0, 11.5), (0, 4.8))
    ins = ["h", "e", "l", "l"]; outs = ["e", "l", "l", "o"]
    for t in range(4):
        x = 1.2 + t * 2.5
        box(ax, x, 0.3, 0.9, 0.6, f"x{t + 1} = '{ins[t]}'", BLUE, fill=LBLUE, fs=8.5)
        box(ax, x, 1.7, 0.9, 0.7, f"a{t + 1}", ORANGE, fill=LORANGE, fs=10)
        box(ax, x - 0.15, 3.1, 1.2, 0.9, f"softmax over\ncharacters\n→ sample '{outs[t]}'", GREEN, fill=LGREEN, fs=7.5)
        arrow(ax, x + 0.45, 0.9, x + 0.45, 1.7); arrow(ax, x + 0.45, 2.4, x + 0.45, 3.1)
        if t < 3:
            arrow(ax, x + 0.9, 2.05, x + 2.5, 2.05, color=ORANGE)
            ax.annotate("", xy=(x + 2.95, 0.9), xytext=(x + 1.05, 3.55), arrowprops=dict(arrowstyle="-|>", color=RED, ls="--", lw=1.4,
                                                                                         connectionstyle="arc3,rad=-0.2"))
    ax.text(0.2, 2.05, "a₀", fontsize=10); arrow(ax, 0.5, 2.05, 1.2, 2.05, color=ORANGE)
    ax.text(5.75, 4.45, "the same weights W_aa, W_ax, W_ya at every step  ·  red dashed = the sampled character becomes the next input",
            ha="center", fontsize=9, color=SOFT)
    save(fig, "11_text_generation")


def fig_wavenet():
    fig, ax = canvas(12, 4.6, (0, 12), (-1.0, 4.6))
    n = 16; dil = [1, 2, 4, 8]
    xs = lambda i: 0.6 + i * 0.68
    active = {(4, n - 1)}
    for l in range(4, 0, -1):
        d = dil[l - 1]
        for (ll, i) in list(active):
            if ll == l:
                for j in (i, i - d):
                    if j >= 0:
                        ax.plot([xs(i), xs(j)], [l, l - 1], color=ORANGE, lw=1.8, zorder=1); active.add((l - 1, j))
    for l in range(5):
        for i in range(n):
            on = (l, i) in active
            col = (GREEN if l == 4 else (BLUE if l == 0 else ORANGE)) if on else "#cbd5e1"
            ax.add_patch(Circle((xs(i), l), 0.13, color=col, zorder=3))
    for l, d in enumerate(dil):
        ax.text(11.6, l + 0.5, f"dilation {d}", fontsize=9, color=SOFT, va="center")
    ax.text(xs(0), -0.45, "past audio samples", fontsize=9, color=BLUE)
    ax.text(xs(n - 1), 4.3, "next sample", fontsize=9, color=GREEN, ha="center")
    ax.text(6.0, -0.8, "each output sees 16 past samples with only 4 layers (receptive field doubles per layer); never the future (causal)",
            fontsize=8.8, color=SOFT, ha="center", va="top")
    save(fig, "11_wavenet")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("fig_") and callable(fn):
            fn()
