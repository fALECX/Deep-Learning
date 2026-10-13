"""Figures for the learning docs. Run:  python Lernen/_figures/make_figures.py
Writes PNGs into Lernen/img/. Colour code matches the formulas:
blue = input/data, orange = learned weights, green = model output, red = true label, purple = setting you choose.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = pathlib.Path(__file__).resolve().parent.parent / "img"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#3b82f6", "#e8833a", "#22a06b", "#e5484d", "#a855f7", "#64748b"
INK, SOFT = "#1f2937", "#6b7280"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": INK, "xtick.color": SOFT, "ytick.color": SOFT, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})
torch.manual_seed(0); np.random.seed(0)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=140, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, color, fill=None, fs=10, weight="normal", alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.6, edgecolor=color, facecolor=fill or "white", alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight=weight)


def arrow(ax, x1, y1, x2, y2, color=SOFT, lw=1.6, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12, color=color,
                                 linewidth=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off"); ax.set_aspect("equal")
    return fig, ax


# ---------------------------------------------------------------- 01 basics
def fig_activations():
    z = np.linspace(-5, 5, 400)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, (name, y, c, rng) in zip(axes, [("Sigmoid  σ(z)", 1 / (1 + np.exp(-z)), GREEN, (-0.1, 1.1)),
                                            ("tanh(z)", np.tanh(z), BLUE, (-1.15, 1.15)),
                                            ("ReLU  max(0, z)", np.maximum(0, z), ORANGE, (-0.5, 5.2))]):
        ax.axhline(0, color="#e2e8f0", lw=1); ax.axvline(0, color="#e2e8f0", lw=1)
        ax.plot(z, y, color=c, lw=2.6); ax.set_title(name); ax.set_ylim(*rng); ax.set_xlabel("z")
    axes[0].text(2.2, 0.55, "output 0…1\n= a probability", fontsize=9, color=SOFT)
    axes[1].text(1.2, -0.7, "output −1…1", fontsize=9, color=SOFT)
    axes[2].text(-4.6, 3.5, "0 for negative z,\nidentity for positive z", fontsize=9, color=SOFT)
    save(fig, "01_activations")


def fig_xor():
    n = 120
    centers = [(-1, -1, 0), (1, 1, 0), (-1, 1, 1), (1, -1, 1)]
    X = np.concatenate([np.random.randn(n, 2) * 0.35 + [cx, cy] for cx, cy, _ in centers])
    y = np.concatenate([np.full(n, lab) for _, _, lab in centers])
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def train(model, steps=1500):
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        for _ in range(steps):
            opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xt).squeeze(1), yt)
            loss.backward(); opt.step()
        return model

    lin = train(torch.nn.Linear(2, 1))
    mlp = train(torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1)))
    gx, gy = np.meshgrid(np.linspace(-2.2, 2.2, 300), np.linspace(-2.2, 2.2, 300))
    G = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, model, title in [(axes[0], lin, "Linear model: one straight line"),
                             (axes[1], mlp, "1 hidden layer with ReLU: bent boundary")]:
        with torch.no_grad():
            p = torch.sigmoid(model(G)).numpy().reshape(gx.shape)
        ax.contourf(gx, gy, p, levels=[0, 0.5, 1], colors=["#dbeafe", "#fee2e2"], alpha=0.9)
        ax.contour(gx, gy, p, levels=[0.5], colors=[INK], linewidths=2)
        ax.scatter(*X[y == 0].T, s=10, color=BLUE, label="class 0  (x₁, x₂ same sign)")
        ax.scatter(*X[y == 1].T, s=10, color=RED, label="class 1  (x₁, x₂ opposite sign)")
        acc = ((p.ravel() > 0.5) == 0).mean()
        with torch.no_grad():
            tr_acc = ((torch.sigmoid(model(Xt)).squeeze(1) > 0.5).float() == yt).float().mean().item()
        ax.set_title(f"{title}\ntraining accuracy {tr_acc:.0%}"); ax.set_xlabel("x₁"); ax.set_ylabel("x₂")
        ax.set_aspect("equal")
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.15), ncol=2, fontsize=9)
    save(fig, "01_xor")


def fig_gradient_descent():
    L = lambda w: (w - 2) ** 2 + 0.5
    dL = lambda w: 2 * (w - 2)
    w = np.linspace(-1.5, 5.5, 300)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, (eta, title, c) in zip(axes, [(0.05, "η too small: slow", PURPLE), (0.35, "η about right", GREEN),
                                          (0.97, "η too large: zig-zag", RED)]):
        ax.plot(w, L(w), color="#94a3b8", lw=2)
        x = -1.0; xs = [x]
        for _ in range(8):
            x = x - eta * dL(x); xs.append(x)
        xs = np.array(xs)
        ax.plot(xs, L(xs), "-o", color=c, ms=5, lw=1.4)
        ax.scatter([2], [L(2)], marker="*", s=180, color=INK, zorder=5)
        ax.set_title(title); ax.set_xlabel("weight w"); ax.set_ylim(0, 13)
        ax.text(2.1, 0.9, "minimum", fontsize=9, color=SOFT)
    axes[0].set_ylabel("loss L(w)")
    save(fig, "01_gradient_descent")


def fig_fitting():
    rng = np.random.RandomState(3)
    x = np.sort(rng.uniform(0, 1, 18)); f = lambda t: np.sin(2 * np.pi * t)
    y = f(x) + rng.randn(18) * 0.2
    t = np.linspace(0, 1, 400)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for ax, (deg, title) in zip(axes, [(1, "Underfitting (too simple)"), (4, "Good fit"), (15, "Overfitting (memorises noise)")]):
        coef = np.polyfit(x, y, deg)
        ax.plot(t, f(t), color="#cbd5e1", lw=6, label="true pattern")
        ax.plot(t, np.clip(np.polyval(coef, t), -2, 2), color=GREEN, lw=2.2, label="model")
        ax.scatter(x, y, color=BLUE, s=22, zorder=5, label="training data")
        ax.set_title(title); ax.set_ylim(-2, 2); ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(fontsize=9, loc="lower left")
    save(fig, "01_fitting")


# ---------------------------------------------------------------- 02 text
def fig_tokenisation():
    fig, ax = canvas(11, 3.2, (0, 11), (0, 3.2))
    rows = [("word level", ["the", "un­believably", "LOSER"], [BLUE, RED, RED], "rare words → [UNK]"),
            ("character level", list("the un…"), [BLUE] * 7, "very long sequences"),
            ("sub-word (BPE)", ["the", "un", "believ", "ably", "LOS", "ER"], [BLUE] * 6, "fixed vocabulary, no OOV")]
    for r, (name, toks, cols, note) in enumerate(rows):
        yy = 2.55 - r * 1.0
        ax.text(0.1, yy, name, va="center", fontsize=10.5, weight="bold")
        x = 2.2
        for tok, c in zip(toks, cols):
            w = 0.18 + 0.115 * len(tok)
            box(ax, x, yy - 0.28, w, 0.56, tok, c, fill="#eff6ff" if c == BLUE else "#fef2f2", fs=10)
            x += w + 0.12
        ax.text(8.6, yy, note, va="center", fontsize=9.5, color=SOFT)
    save(fig, "02_tokenisation")


def fig_embedding_space():
    pts = {"king": (3.0, 3.2), "queen": (3.1, 1.2), "man": (0.9, 3.0), "woman": (1.0, 1.0),
           "apple": (5.3, 0.6), "banana": (5.8, 1.1), "cat": (4.9, 3.8), "dog": (5.5, 3.5)}
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for wd, (x, y) in pts.items():
        ax.scatter(x, y, s=60, color=BLUE, zorder=3); ax.text(x + 0.1, y + 0.12, wd, fontsize=11)
    for a, b in [("man", "woman"), ("king", "queen")]:
        arrow(ax, *pts[a], *pts[b], color=ORANGE, lw=2)
    ax.text(1.95, 2.0, "same direction:\n'male → female'", color=ORANGE, fontsize=9.5, ha="center")
    ax.set_title("Embeddings: similar words lie close, relations become directions")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(0, 6.8); ax.set_ylim(0, 4.6)
    save(fig, "02_embedding_space")


def fig_padding_mask():
    toks = ["[CLS]", "I", "love", "it", "[SEP]", "[PAD]", "[PAD]"]
    n = len(toks); real = np.array([1, 1, 1, 1, 1, 0, 0])
    M = np.tile(real, (n, 1)).astype(float)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(M, cmap=matplotlib.colors.ListedColormap(["#f1f5f9", "#bbf7d0"]))
    for i in range(n):
        for j in range(n):
            ax.text(j, i, "score" if M[i, j] else "−∞", ha="center", va="center", fontsize=8, color=INK if M[i, j] else RED)
    ax.set_xticks(range(n), toks, rotation=45, ha="right"); ax.set_yticks(range(n), toks)
    ax.set_xlabel("key token j (attended to)"); ax.set_ylabel("query token i")
    ax.set_title("Padding mask: nobody looks at [PAD]")
    for s in ax.spines.values(): s.set_visible(False)
    save(fig, "02_padding_mask")


# ---------------------------------------------------------------- 03 RNN
def fig_task_shapes():
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    specs = [("one-to-many", [1, 0, 0], [1, 1, 1], "caption an image"),
             ("many-to-one", [1, 1, 1], [0, 0, 1], "sentiment · A2 Part A"),
             ("many-to-many (aligned)", [1, 1, 1], [1, 1, 1], "NER tagging · A2 Part B"),
             ("encoder–decoder", [1, 1, 1, 0, 0], [0, 0, 0, 1, 1], "translation")]
    for ax, (title, ins, outs, ex) in zip(axes, specs):
        ax.set_xlim(-0.5, 5); ax.set_ylim(-0.3, 3.4); ax.axis("off")
        k = len(ins)
        for t in range(k):
            box(ax, t * 1.0, 1.15, 0.7, 0.7, "", GREY, fill="#f8fafc")
            if t: arrow(ax, (t - 1) * 1.0 + 0.7, 1.5, t * 1.0, 1.5, color=GREY)
            if ins[t]:
                box(ax, t * 1.0 + 0.1, 0.05, 0.5, 0.5, "x", BLUE, fill="#eff6ff", fs=9)
                arrow(ax, t * 1.0 + 0.35, 0.55, t * 1.0 + 0.35, 1.15, color=BLUE)
            if outs[t]:
                box(ax, t * 1.0 + 0.1, 2.45, 0.5, 0.5, "ŷ", GREEN, fill="#f0fdf4", fs=9)
                arrow(ax, t * 1.0 + 0.35, 1.85, t * 1.0 + 0.35, 2.45, color=GREEN)
        ax.set_title(title, fontsize=11); ax.text(k * 0.5 - 0.15, -0.25, ex, ha="center", fontsize=9, color=SOFT)
    save(fig, "03_task_shapes")


def fig_unrolled_rnn():
    fig, ax = canvas(11, 3.6, (-0.6, 10.8), (-0.4, 3.6))
    words = ["The", "keys", "are", "gone"]
    box(ax, -0.5, 1.2, 0.9, 0.8, "h₀", GREY, fill="#f8fafc")
    for t, wd in enumerate(words):
        x = 1.0 + t * 2.4
        box(ax, x, 1.1, 1.3, 1.0, f"RNN cell\nh{t + 1}", GREEN, fill="#f0fdf4")
        box(ax, x + 0.2, -0.3, 0.9, 0.6, wd, BLUE, fill="#eff6ff")
        arrow(ax, x + 0.65, 0.3, x + 0.65, 1.1, color=BLUE)
        arrow(ax, x - (1.1 if t else 0.6), 1.6, x, 1.6, color=GREEN)
        ax.text(x - (0.55 if t else 0.3), 1.75, "W_h", color=ORANGE, fontsize=9, ha="center")
        ax.text(x + 0.85, 0.65, "W_x", color=ORANGE, fontsize=9)
    box(ax, 8.4, 2.75, 1.5, 0.6, "ŷ (label)", GREEN, fill="#f0fdf4")
    arrow(ax, 8.85, 2.1, 9.1, 2.75, color=GREEN)
    ax.text(5.2, 3.35, "the same weights W_h and W_x are reused at every step", ha="center", color=ORANGE, fontsize=10)
    save(fig, "03_unrolled_rnn")


def fig_vanishing():
    t = np.arange(0, 51)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(t, 0.9 ** t, color=BLUE, lw=2.4, label="factor 0.9 per step → vanishes (0.9⁵⁰ ≈ 0.005)")
    ax.plot(t, 1.1 ** t, color=RED, lw=2.4, label="factor 1.1 per step → explodes (1.1⁵⁰ ≈ 117)")
    ax.axhline(1, color="#cbd5e1", lw=1)
    ax.set_yscale("log"); ax.set_xlabel("number of time steps the gradient travels back")
    ax.set_ylabel("gradient size (log scale)"); ax.legend(fontsize=9.5, loc="upper left")
    ax.set_title("Why RNNs forget: a product of many factors")
    save(fig, "03_vanishing")


def fig_lstm():
    fig, ax = canvas(11, 4.4, (0, 11), (0, 4.4))
    ax.add_patch(FancyBboxPatch((1.2, 0.5), 8.4, 3.3, boxstyle="round,pad=0.02,rounding_size=0.2",
                                edgecolor="#cbd5e1", facecolor="#f8fafc", lw=1.5))
    arrow(ax, 0.2, 3.3, 10.6, 3.3, color=GREEN, lw=3)
    ax.text(0.2, 3.72, "cell state c (conveyor belt)", color=GREEN, fontsize=10, weight="bold")
    for x, sym, name, c in [(2.6, "×", "forget gate f", PURPLE), (5.0, "+", "input gate i × candidate c̃", PURPLE)]:
        ax.add_patch(plt.Circle((x, 3.3), 0.28, color="white", ec=c, lw=2)); ax.text(x, 3.3, sym, ha="center", va="center", fontsize=14, color=c)
        ax.text(x, 2.6, name, ha="center", fontsize=9.5, color=c)
        arrow(ax, x, 1.4, x, 3.02, color=c)
    box(ax, 6.9, 1.7, 2.0, 0.7, "output gate o\n→ h = o · tanh(c)", PURPLE, fill="white", fs=9)
    arrow(ax, 7.9, 3.3, 7.9, 2.4, color=GREEN)
    arrow(ax, 8.9, 2.05, 10.6, 2.05, color=GREEN); ax.text(9.4, 2.2, "h (output)", fontsize=9.5, color=GREEN)
    box(ax, 0.2, 0.9, 1.6, 0.8, "h(t−1), x(t)", BLUE, fill="#eff6ff", fs=9.5)
    arrow(ax, 1.8, 1.3, 2.4, 1.3, color=BLUE)
    ax.plot([2.4, 7.9], [1.3, 1.3], color=BLUE, lw=1.5); arrow(ax, 7.9, 1.3, 7.9, 1.7, color=BLUE)
    ax.text(5.6, 0.75, "all gates are small sigmoid layers that read h(t−1) and x(t)", ha="center", fontsize=9.5, color=SOFT)
    ax.text(5.6, 4.15, "LSTM: the memory is updated by + (not by repeated ×W), so gradients survive", ha="center", fontsize=10.5, weight="bold")
    save(fig, "03_lstm")


# ---------------------------------------------------------------- 04 attention
def fig_attention_map():
    toks = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "tired"]
    w = np.array([0.02, 0.62, 0.02, 0.04, 0.01, 0.12, 0.03, 0.08, 0.02, 0.04])
    fig, ax = plt.subplots(figsize=(10.5, 2.2))
    ax.bar(range(len(toks)), w, color=[GREEN if i == 1 else "#cbd5e1" for i in range(len(toks))])
    ax.set_xticks(range(len(toks)), toks); ax.set_ylabel("attention weight")
    ax.set_title("Attention weights of the query token \"it\" (they sum to 1)")
    ax.get_xticklabels()[7].set_color(PURPLE); ax.get_xticklabels()[7].set_weight("bold")
    save(fig, "04_attention_it")


def fig_qkv_flow():
    fig, ax = canvas(12, 3.6, (0, 12), (0, 3.6))
    box(ax, 0.1, 1.4, 1.2, 0.8, "token\nvectors X", BLUE, fill="#eff6ff", fs=9.5)
    for k, (name, y) in enumerate([("Q = X·W_Q", 2.7), ("K = X·W_K", 1.5), ("V = X·W_V", 0.3)]):
        box(ax, 2.0, y, 1.6, 0.7, name, ORANGE, fill="#fff7ed", fs=9.5)
        arrow(ax, 1.3, 1.8, 2.0, y + 0.35, color=BLUE)
    box(ax, 4.4, 1.9, 1.7, 0.8, "scores\nQ·Kᵀ", GREY, fs=9.5)
    arrow(ax, 3.6, 3.05, 4.4, 2.45, color=ORANGE); arrow(ax, 3.6, 1.85, 4.4, 2.15, color=ORANGE)
    box(ax, 6.7, 1.9, 1.3, 0.8, "÷ √d_k", PURPLE, fs=10)
    arrow(ax, 6.1, 2.3, 6.7, 2.3)
    box(ax, 8.6, 1.9, 1.3, 0.8, "softmax\n(rows)", GREY, fs=9.5)
    arrow(ax, 8.0, 2.3, 8.6, 2.3)
    box(ax, 10.4, 1.85, 1.5, 0.9, "weights × V\n= output", GREEN, fill="#f0fdf4", fs=9.5)
    arrow(ax, 9.9, 2.3, 10.4, 2.3); arrow(ax, 3.6, 0.65, 11.15, 1.85, color=ORANGE)
    ax.text(6, 0.1, "every token gets a weighted average of all value vectors", ha="center", fontsize=9.5, color=SOFT)
    save(fig, "04_qkv_flow")


def fig_softmax_saturation():
    s = np.array([2.0, 1.0, 0.5, 0.2])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3))
    for ax, scale, title in [(axes[0], 1.0, "scores scaled by 1/√d_k"), (axes[1], 8.0, "unscaled scores (×8)")]:
        p = np.exp(s * scale) / np.exp(s * scale).sum()
        ax.bar(range(4), p, color=[GREEN, "#94a3b8", "#94a3b8", "#94a3b8"])
        ax.set_ylim(0, 1.05); ax.set_xticks(range(4), ["tok 1", "tok 2", "tok 3", "tok 4"]); ax.set_title(title)
        for i, v in enumerate(p): ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    axes[0].set_ylabel("softmax weight")
    axes[1].text(1.6, 0.6, "almost one-hot →\ngradients ≈ 0", color=RED, fontsize=9.5)
    save(fig, "04_softmax_scaling")


def fig_positional():
    pos = np.arange(50)[:, None]; i = np.arange(64)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / 64)
    pe = np.where(i % 2 == 0, np.sin(angle), np.cos(angle))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("embedding dimension"); ax.set_ylabel("position in sentence")
    ax.set_title("Sinusoidal positional encoding: each row is a unique pattern")
    plt.colorbar(im, ax=ax, fraction=0.03)
    save(fig, "04_positional_encoding")


def fig_masks():
    n = 6; toks = ["I", "love", "this", "meme", "[PAD]", "[PAD]"]
    causal = np.tril(np.ones((n, n))); pad = np.tile([1, 1, 1, 1, 0, 0], (n, 1))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    cm = matplotlib.colors.ListedColormap(["#f1f5f9", "#bbf7d0"])
    for ax, M, title in [(axes[0], pad, "Padding mask (encoder)\nhide [PAD]"), (axes[1], causal, "Causal mask (decoder)\nhide the future")]:
        ax.imshow(M, cmap=cm)
        ax.set_xticks(range(n), toks if ax is axes[0] else [f"t{k + 1}" for k in range(n)], rotation=45, ha="right")
        ax.set_yticks(range(n), toks if ax is axes[0] else [f"t{k + 1}" for k in range(n)])
        ax.set_title(title); [s.set_visible(False) for s in ax.spines.values()]
        for a in range(n):
            for b in range(n):
                if not M[a, b]: ax.text(b, a, "−∞", ha="center", va="center", color=RED, fontsize=8)
    axes[0].set_ylabel("query (looks)"); axes[0].set_xlabel("key (is looked at)"); axes[1].set_xlabel("key")
    save(fig, "04_masks")


def fig_block():
    fig, ax = canvas(5, 6, (0, 5), (0, 6.2))
    box(ax, 1.3, 0.1, 2.4, 0.6, "token vectors x", BLUE, fill="#eff6ff")
    box(ax, 1.3, 1.3, 2.4, 0.8, "multi-head\nself-attention", ORANGE, fill="#fff7ed")
    box(ax, 1.3, 2.5, 2.4, 0.55, "add + LayerNorm", GREY)
    box(ax, 1.3, 3.5, 2.4, 0.8, "feed-forward\n(per token)", ORANGE, fill="#fff7ed")
    box(ax, 1.3, 4.7, 2.4, 0.55, "add + LayerNorm", GREY)
    box(ax, 1.3, 5.6, 2.4, 0.55, "contextual vectors", GREEN, fill="#f0fdf4")
    for y1, y2 in [(0.7, 1.3), (2.1, 2.5), (3.05, 3.5), (4.3, 4.7), (5.25, 5.6)]:
        arrow(ax, 2.5, y1, 2.5, y2)
    ax.plot([1.3, 0.6, 0.6, 1.3], [0.95, 0.95, 2.78, 2.78], color=GREEN, lw=1.6, ls="--")
    ax.plot([1.3, 0.6, 0.6, 1.3], [3.25, 3.25, 4.98, 4.98], color=GREEN, lw=1.6, ls="--")
    ax.text(0.05, 1.9, "residual", rotation=90, color=GREEN, fontsize=9, va="center")
    ax.text(0.05, 4.1, "residual", rotation=90, color=GREEN, fontsize=9, va="center")
    ax.text(4.0, 3.1, "× 12 blocks\n(BERT-base,\nDeBERTa-base)", fontsize=9, color=SOFT)
    save(fig, "04_encoder_block")


# ---------------------------------------------------------------- 05 adaptation
def fig_adaptation():
    fig, ax = canvas(12, 3.6, (0, 12), (0, 3.6))
    specs = [("Zero-shot", 0, "prompt only"), ("Feature extraction", 1, "train head"),
             ("LoRA", 2, "head + tiny adapters"), ("Full fine-tuning", 3, "train everything")]
    for k, (name, lvl, note) in enumerate(specs):
        x = 0.2 + k * 3.0
        ax.text(x + 1.25, 3.35, name, ha="center", weight="bold", fontsize=10.5)
        for layer in range(4):
            trained = (lvl == 3) or (lvl >= 1 and layer == 3)
            col, fill = (ORANGE, "#fff7ed") if trained else (GREY, "#e2e8f0")
            label = "head" if layer == 3 else f"layer {layer + 1}"
            box(ax, x, 0.45 + layer * 0.68, 2.5, 0.55, label, col, fill=fill, fs=9)
            if lvl == 2 and layer < 3:
                box(ax, x + 1.95, 0.5 + layer * 0.68, 0.45, 0.45, "BA", ORANGE, fill="#fff7ed", fs=8)
        ax.text(x + 1.25, 0.12, note, ha="center", fontsize=9, color=SOFT)
    ax.text(6, -0.25, "grey = frozen   ·   orange = trained   ·   cost and flexibility grow from left to right", ha="center", fontsize=9.5, color=SOFT)
    save(fig, "05_adaptation_spectrum")


def fig_lora():
    fig, ax = canvas(9, 4, (0, 9), (0, 4.2))
    box(ax, 0.2, 1.7, 1.2, 0.8, "input x\n(d = 768)", BLUE, fill="#eff6ff", fs=9.5)
    box(ax, 3.0, 2.6, 3.0, 1.2, "W₀ (768 × 768)\nfrozen, 589,824 values", GREY, fill="#e2e8f0", fs=9.5)
    box(ax, 2.6, 0.4, 1.4, 1.2, "A\n(16 × 768)", ORANGE, fill="#fff7ed", fs=9.5)
    box(ax, 5.0, 0.4, 1.4, 1.2, "B\n(768 × 16)", ORANGE, fill="#fff7ed", fs=9.5)
    ax.text(4.5, 1.25, "r = 16", ha="center", color=PURPLE, fontsize=10, weight="bold")
    arrow(ax, 4.0, 1.0, 5.0, 1.0, color=ORANGE)
    arrow(ax, 1.4, 2.3, 3.0, 3.1); arrow(ax, 1.4, 1.9, 2.6, 1.0, color=ORANGE)
    ax.add_patch(plt.Circle((7.2, 2.1), 0.3, color="white", ec=INK, lw=1.6)); ax.text(7.2, 2.1, "+", ha="center", va="center", fontsize=15)
    arrow(ax, 6.0, 3.2, 7.05, 2.35); arrow(ax, 6.4, 1.0, 7.05, 1.85, color=ORANGE)
    box(ax, 7.8, 1.7, 1.1, 0.8, "output\ny", GREEN, fill="#f0fdf4", fs=9.5)
    arrow(ax, 7.5, 2.1, 7.8, 2.1)
    ax.text(4.5, 0.05, "trainable: 16 × (768 + 768) = 24,576 values (≈ 4 %)", ha="center", fontsize=9.5, color=ORANGE)
    save(fig, "05_lora")


# ---------------------------------------------------------------- 06 framing
def fig_softmax_vs_sigmoid():
    z = np.array([2.2, 1.9, 1.5, -1.0]); names = ["Name calling", "Smears", "Loaded lang.", "Slogans"]
    sm = np.exp(z) / np.exp(z).sum(); sg = 1 / (1 + np.exp(-z))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), sharey=True)
    for ax, p, title in [(axes[0], sm, "softmax: probabilities compete (sum = 1)"), (axes[1], sg, "sigmoid: each class on its own")]:
        ax.bar(names, p, color=GREEN); ax.axhline(0.5, color=PURPLE, ls="--", lw=1.4)
        ax.set_ylim(0, 1.05); ax.set_title(title, fontsize=11)
        for i, v in enumerate(p): ax.text(i, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
        ax.tick_params(axis="x", labelsize=9)
    axes[1].text(3.05, 0.52, "threshold τ", color=PURPLE, fontsize=9)
    fig.suptitle("Same logits, a meme that really uses the first three techniques", fontsize=11, y=1.03)
    save(fig, "06_softmax_vs_sigmoid")


def fig_overlap_tagging():
    toks = ["CORRUPT", "CROOKED", "HILLARY", "wants", "your", "guns"]
    rows = {"Loaded Language": [1, 1, 0, 0, 0, 0], "Name calling": [1, 1, 1, 0, 0, 0],
            "Smears": [1, 1, 1, 1, 1, 1], "Appeal to fear": [0, 0, 0, 1, 1, 1]}
    cols = [ORANGE, RED, BLUE, PURPLE]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    for r, ((name, row), c) in enumerate(zip(rows.items(), cols)):
        for t, v in enumerate(row):
            ax.add_patch(Rectangle((t, -r - 0.42), 0.94, 0.84, color=c if v else "#f1f5f9", alpha=0.85 if v else 1))
        ax.text(-0.15, -r, name, ha="right", va="center", fontsize=10)
    for t, tok in enumerate(toks): ax.text(t + 0.47, 0.75, tok, ha="center", fontsize=10, weight="bold")
    ax.set_xlim(-2.8, 6.2); ax.set_ylim(-3.6, 1.1); ax.axis("off")
    ax.set_title("Multi-label tagging: one yes/no row per technique, rows may overlap (BIO could give each word only one tag)", fontsize=10.5)
    save(fig, "06_overlap_tagging")


# ---------------------------------------------------------------- 07 evaluation
def fig_learning_curves():
    e = np.arange(1, 31)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    specs = [("Underfitting", 1.2 - 0.25 * (1 - np.exp(-e / 8)), 1.25 - 0.25 * (1 - np.exp(-e / 8))),
             ("Good fit", 1.2 * np.exp(-e / 6) + 0.25, 1.2 * np.exp(-e / 6) + 0.33),
             ("Overfitting", 1.2 * np.exp(-e / 5) + 0.05, 1.0 * np.exp(-e / 5) + 0.35 + 0.012 * np.maximum(0, e - 8)),
             ("Learning rate too high", 0.7 + 0.25 * np.sin(e * 1.3) + 0.1 * np.random.RandomState(1).randn(30),
              0.8 + 0.3 * np.sin(e * 1.3 + 0.6) + 0.1 * np.random.RandomState(2).randn(30))]
    for ax, (title, tr, va) in zip(axes, specs):
        ax.plot(e, tr, color=BLUE, lw=2.2, label="training loss"); ax.plot(e, va, color=ORANGE, lw=2.2, label="validation loss")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylim(0, 1.5)
    axes[2].axvline(8, color=PURPLE, ls="--", lw=1.3); axes[2].text(8.5, 1.3, "stop here\n(early stopping)", color=PURPLE, fontsize=8.5)
    axes[0].set_ylabel("loss"); axes[0].legend(fontsize=9)
    save(fig, "07_learning_curves")


def fig_kfold():
    fig, ax = canvas(9, 3.4, (0, 9), (0, 3.5))
    for r in range(5):
        for f in range(5):
            val = f == r
            ax.add_patch(Rectangle((1.6 + f * 1.4, 2.9 - r * 0.62), 1.32, 0.5, color=ORANGE if val else "#dbeafe"))
            ax.text(1.6 + f * 1.4 + 0.66, 3.15 - r * 0.62, "validate" if val else "train", ha="center", va="center", fontsize=8.5)
        ax.text(1.4, 3.15 - r * 0.62, f"run {r + 1}", ha="right", va="center", fontsize=9.5)
    ax.text(5.1, 0.05, "every meme is validated exactly once, by a model that never saw it (out-of-fold)", ha="center", fontsize=9.5, color=SOFT)
    save(fig, "07_kfold")


def fig_precision_recall():
    fig, ax = canvas(9, 4.2, (0, 9), (0, 4.3))
    ax.add_patch(plt.Circle((3.0, 2.2), 1.7, color=RED, alpha=0.18)); ax.text(1.6, 3.95, "truly Smears (10)", color=RED, fontsize=10)
    ax.add_patch(plt.Circle((4.6, 2.2), 1.45, color=GREEN, alpha=0.22)); ax.text(4.6, 3.8, "flagged by model (8)", color=GREEN, fontsize=10)
    ax.text(2.2, 2.2, "FN = 4\n(missed)", ha="center", fontsize=10); ax.text(3.75, 2.2, "TP = 6", ha="center", fontsize=11, weight="bold")
    ax.text(5.45, 2.2, "FP = 2\n(false\nalarm)", ha="center", fontsize=10)
    ax.text(6.9, 3.0, "Precision = TP / flagged\n= 6 / 8 = 0.75", fontsize=10.5, color=GREEN)
    ax.text(6.9, 1.6, "Recall = TP / truly yes\n= 6 / 10 = 0.60", fontsize=10.5, color=RED)
    save(fig, "07_precision_recall")


def fig_threshold():
    rng = np.random.RandomState(0)
    pos = np.clip(rng.normal(0.65, 0.18, 400), 0, 1); neg = np.clip(rng.normal(0.3, 0.17, 1200), 0, 1)
    th = np.linspace(0.05, 0.95, 100)
    P = [((pos >= t).sum()) / max((pos >= t).sum() + (neg >= t).sum(), 1) for t in th]
    R = [(pos >= t).mean() for t in th]; F = [2 * p * r / (p + r) if p + r else 0 for p, r in zip(P, R)]
    fig, ax = plt.subplots(figsize=(8, 3.3))
    ax.plot(th, P, color=GREEN, lw=2.2, label="precision"); ax.plot(th, R, color=RED, lw=2.2, label="recall")
    ax.plot(th, F, color=INK, lw=2.2, ls="--", label="F1")
    best = th[int(np.argmax(F))]; ax.axvline(best, color=PURPLE, lw=1.3); ax.text(best + 0.01, 0.1, "best F1 threshold\n(tune on validation!)", color=PURPLE, fontsize=9)
    ax.set_xlabel("threshold τ"); ax.set_ylim(0, 1.02); ax.legend(fontsize=9.5, loc="lower left")
    ax.set_title("Higher threshold → fewer labels → higher precision, lower recall")
    save(fig, "07_threshold")


# ---------------------------------------------------------------- 08 interpretability
def fig_saturation_ig():
    z = np.linspace(-8, 8, 400); s = 1 / (1 + np.exp(-z))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.8))
    ax = axes[0]; ax.plot(z, s, color=GREEN, lw=2.5)
    for x0, c, lab in [(0, BLUE, "uncertain: steep → big gradient"), (6, RED, "confident: flat → gradient ≈ 0")]:
        g = np.exp(-x0) / (1 + np.exp(-x0)) ** 2; y0 = 1 / (1 + np.exp(-x0))
        ax.plot([x0 - 1.8, x0 + 1.8], [y0 - 1.8 * g, y0 + 1.8 * g], color=c, lw=2); ax.scatter([x0], [y0], color=c, zorder=5)
        ax.text(2.3 if x0 else -7.8, 0.78 if x0 else 0.15, lab, color=c, fontsize=9)
    ax.set_title("Saturation: flat curve, tiny gradient"); ax.set_xlabel("input strength"); ax.set_ylabel("output")
    ax = axes[1]; a = np.linspace(0, 1, 200); f = 1 / (1 + np.exp(-(12 * a - 3)))
    ax.plot(a, f, color=GREEN, lw=2.5); ax.fill_between(a, 0, np.gradient(f, a) / 12, color=ORANGE, alpha=0.3, label="gradient along the path")
    ax.scatter([0, 1], [f[0], f[-1]], color=[PURPLE, BLUE], zorder=5, s=50)
    ax.text(0.02, f[0] + 0.08, "baseline x′", color=PURPLE, fontsize=9.5); ax.text(0.78, f[-1] - 0.15, "real input x", color=BLUE, fontsize=9.5)
    ax.set_title("Integrated Gradients: sum gradients along the path"); ax.set_xlabel("α (0 = baseline, 1 = input)")
    ax.legend(fontsize=9, loc="center right")
    save(fig, "08_saturation_ig")


def fig_shapley():
    fig, ax = canvas(10, 3.6, (0, 10), (0, 3.6))
    ax.text(0.1, 3.3, "order A → B", weight="bold"); ax.text(0.1, 1.5, "order B → A", weight="bold")
    for y, steps in [(2.4, [("∅", 0.1), ("+A", 0.5), ("+B", 0.9)]), (0.6, [("∅", 0.1), ("+B", 0.3), ("+A", 0.9)])]:
        for k, (lab, v) in enumerate(steps):
            box(ax, 0.3 + k * 2.4, y, 1.4, 0.6, f"{lab}: v = {v}", GREY, fs=9.5)
            if k:
                d = v - steps[k - 1][1]; who = lab[1]
                arrow(ax, 0.3 + (k - 1) * 2.4 + 1.4, y + 0.3, 0.3 + k * 2.4, y + 0.3, color=ORANGE if who == "A" else BLUE)
                ax.text(0.3 + k * 2.4 - 0.5, y + 0.45, f"{who} adds {d:.1f}", ha="center", fontsize=9, color=ORANGE if who == "A" else BLUE)
    ax.text(7.6, 2.4, "φ_A = ½(0.4 + 0.6) = 0.5", color=ORANGE, fontsize=11, weight="bold")
    ax.text(7.6, 1.8, "φ_B = ½(0.2 + 0.4) = 0.3", color=BLUE, fontsize=11, weight="bold")
    ax.text(7.6, 1.2, "sum = 0.8 = v(A,B) − v(∅)", color=INK, fontsize=10)
    save(fig, "08_shapley")


def fig_deletion():
    k = np.arange(0, 11)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(k, 0.92 * np.exp(-k / 2.2) + 0.05, color=GREEN, lw=2.4, marker="o", ms=4, label="faithful explanation: drops fast")
    ax.plot(k, 0.92 - 0.06 * k + 0.02 * np.sin(k), color="#94a3b8", lw=2.4, marker="o", ms=4, label="random order")
    ax.set_xlabel("tokens removed (most important first)"); ax.set_ylabel("model score"); ax.legend(fontsize=9.5)
    ax.set_title("Deletion test")
    save(fig, "08_deletion")


# ---------------------------------------------------------------- 09 bias / shift
def fig_shift():
    x = np.linspace(-4, 6, 400); g = lambda m, s: np.exp(-((x - m) / s) ** 2 / 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.1))
    ax = axes[0]; ax.fill_between(x, g(0, 1), color=BLUE, alpha=0.35, label="training data (US, 2020)")
    ax.fill_between(x, g(2.2, 1.3), color=ORANGE, alpha=0.35, label="deployment (AU, 2026)")
    ax.set_title("Covariate shift: the inputs look different"); ax.set_yticks([]); ax.set_xticks([]); ax.legend(fontsize=9, loc="upper right")
    ax = axes[1]; cats = ["'Liberals' in\nUS memes", "'Liberals' in\nAU memes"]
    ax.bar(cats, [0.9, 0.1], color=BLUE, label="refers to the left"); ax.bar(cats, [0.1, 0.9], bottom=[0.9, 0.1], color=RED, label="refers to the right")
    ax.set_title("Concept shift: same word, new meaning"); ax.set_yticks([]); ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    save(fig, "09_shift")


def fig_counterfactual():
    fig, ax = canvas(10, 2.6, (0, 10), (0, 2.6))
    box(ax, 0.1, 1.5, 3.6, 0.8, "\"CROOKED BIDEN wants your guns\"", BLUE, fill="#eff6ff", fs=9.5)
    box(ax, 0.1, 0.2, 3.6, 0.8, "\"CROOKED TRUMP wants your guns\"", BLUE, fill="#eff6ff", fs=9.5)
    for y in (1.9, 0.6):
        box(ax, 4.6, y - 0.35, 1.6, 0.7, "same model", ORANGE, fill="#fff7ed", fs=9.5); arrow(ax, 3.7, y, 4.6, y)
        arrow(ax, 6.2, y, 7.0, y)
    ax.text(7.1, 1.9, "P(Name calling) = 0.91", color=GREEN, va="center"); ax.text(7.1, 0.6, "P(Name calling) = 0.88", color=GREEN, va="center")
    ax.text(5.4, 1.25, "only the name changes", ha="center", fontsize=9, color=SOFT)
    save(fig, "09_counterfactual")


# ---------------------------------------------------------------- 10 CNN
def fig_convolution():
    img = np.array([[10, 10, 10, 0, 0], [10, 10, 10, 0, 0], [10, 10, 10, 0, 0], [10, 10, 10, 0, 0], [10, 10, 10, 0, 0]])
    ker = np.array([[1, 0, -1]] * 3)
    out = np.array([[sum((img[i:i + 3, j:j + 3] * ker).ravel()) for j in range(3)] for i in range(3)])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [5, 3, 3]})
    for ax, M, title, cmap in [(axes[0], img, "input image (5 × 5)", "Blues"), (axes[1], ker, "filter K (3 × 3)", "Oranges"),
                               (axes[2], out, "feature map (3 × 3)", "Greens")]:
        ax.imshow(np.abs(M), cmap=cmap, vmin=-5, vmax=np.abs(M).max() + 5)
        for (a, b), v in np.ndenumerate(M): ax.text(b, a, int(v), ha="center", va="center", fontsize=10)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
    axes[0].add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=ORANGE, lw=3))
    axes[2].add_patch(Rectangle((-0.5, -0.5), 1, 1, fill=False, ec=ORANGE, lw=3))
    fig.suptitle("Slide the filter, multiply, sum: the orange window gives 30 (a vertical edge)", fontsize=11, y=1.02)
    save(fig, "10_convolution")


def fig_pooling():
    M = np.array([[1, 3, 2, 4], [5, 9, 1, 0], [7, 2, 6, 8], [0, 1, 3, 2]])
    P = M.reshape(2, 2, 2, 2).max(axis=(1, 3))
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={"width_ratios": [4, 2]})
    cols = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3]])
    axes[0].imshow(cols, cmap=matplotlib.colors.ListedColormap(["#dbeafe", "#fee2e2", "#dcfce7", "#f3e8ff"]))
    for (a, b), v in np.ndenumerate(M): axes[0].text(b, a, v, ha="center", va="center", fontsize=11, weight="bold" if v == P[a // 2, b // 2] else "normal")
    axes[1].imshow(np.arange(4).reshape(2, 2), cmap=matplotlib.colors.ListedColormap(["#dbeafe", "#fee2e2", "#dcfce7", "#f3e8ff"]))
    for (a, b), v in np.ndenumerate(P): axes[1].text(b, a, v, ha="center", va="center", fontsize=12, weight="bold")
    axes[0].set_title("4 × 4 feature map"); axes[1].set_title("2 × 2 max pool")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    save(fig, "10_pooling")


def fig_unet():
    fig, ax = canvas(10, 4.2, (0, 10), (0, 4.4))
    enc = [(0.3, 3.2, 1.0, 0.9), (1.5, 2.2, 0.8, 0.7), (2.6, 1.3, 0.6, 0.55)]
    dec = [(7.0, 1.3, 0.6, 0.55), (7.8, 2.2, 0.8, 0.7), (8.7, 3.2, 1.0, 0.9)]
    for x, y, w, h in enc: box(ax, x, y, w, h, "", BLUE, fill="#eff6ff")
    for x, y, w, h in dec: box(ax, x, y, w, h, "", GREEN, fill="#f0fdf4")
    box(ax, 4.6, 0.3, 0.9, 0.5, "", PURPLE, fill="#f3e8ff")
    path = [(0.8, 3.2), (1.9, 2.9), (2.9, 1.85), (5.05, 0.8), (7.3, 1.85), (8.2, 2.9), (9.2, 3.2)]
    for (x1, y1), (x2, y2) in zip(path, path[1:]): arrow(ax, x1, y1, x2, y2, color=SOFT)
    for (x, y, w, h), (x2, y2, w2, h2) in zip(enc, reversed(dec)):
        arrow(ax, x + w, y + h / 2, x2, y2 + h2 / 2, color=ORANGE, ls="--")
    ax.text(1.4, 4.2, "encoder: shrink, learn WHAT", color=BLUE, fontsize=10, weight="bold")
    ax.text(6.9, 4.2, "decoder: grow back, say WHERE", color=GREEN, fontsize=10, weight="bold")
    ax.text(5.0, 2.6, "skip connections copy details across", color=ORANGE, fontsize=9.5, ha="center")
    ax.text(5.05, 0.05, "bottleneck", color=PURPLE, fontsize=9, ha="center")
    save(fig, "10_unet")


# ---------------------------------------------------------------- 11 generative
def fig_latent_spaces():
    rng = np.random.RandomState(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    cols = [BLUE, ORANGE, GREEN, RED, PURPLE]
    for c, (mx, my) in zip(cols, [(-3, 2), (2.5, 2.5), (-2, -2.5), (3, -1.5), (0, 0.5)]):
        axes[0].scatter(rng.randn(40) * 0.35 + mx, rng.randn(40) * 0.35 + my, s=9, color=c)
    axes[0].scatter([0.8], [-0.5], marker="x", s=120, color=INK); axes[0].text(1.0, -0.3, "random z → nonsense", fontsize=9)
    axes[0].set_title("Autoencoder: scattered islands with gaps")
    for c, (mx, my) in zip(cols, [(-0.8, 0.6), (0.7, 0.7), (-0.6, -0.7), (0.8, -0.5), (0, 0)]):
        axes[1].scatter(rng.randn(40) * 0.45 + mx, rng.randn(40) * 0.45 + my, s=9, color=c, alpha=0.8)
    axes[1].add_patch(plt.Circle((0, 0), 2, fill=False, ls="--", ec=SOFT)); axes[1].text(-1.9, 2.1, "N(0, I)", color=SOFT, fontsize=9)
    axes[1].set_title("VAE: smooth, filled space you can sample")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    axes[0].set_xlim(-5, 5); axes[0].set_ylim(-4, 4); axes[1].set_xlim(-3, 3); axes[1].set_ylim(-2.6, 2.6)
    save(fig, "11_latent_spaces")


def fig_gan():
    fig, ax = canvas(10, 3.3, (0, 10), (-0.4, 2.8))
    box(ax, 0.1, 1.0, 1.4, 0.8, "noise z", PURPLE, fill="#f3e8ff")
    box(ax, 2.2, 1.0, 1.6, 0.8, "Generator G", ORANGE, fill="#fff7ed")
    box(ax, 4.5, 1.0, 1.4, 0.8, "fake image", GREEN, fill="#f0fdf4")
    box(ax, 4.5, 2.0, 1.4, 0.7, "real image", BLUE, fill="#eff6ff")
    box(ax, 6.7, 1.3, 1.7, 0.9, "Discriminator D", ORANGE, fill="#fff7ed")
    box(ax, 8.9, 1.35, 1.0, 0.8, "real?\nfake?", RED, fill="#fef2f2", fs=9)
    arrow(ax, 1.5, 1.4, 2.2, 1.4); arrow(ax, 3.8, 1.4, 4.5, 1.4); arrow(ax, 5.9, 1.4, 6.7, 1.6); arrow(ax, 5.9, 2.35, 6.7, 2.0)
    arrow(ax, 8.4, 1.75, 8.9, 1.75)
    ax.annotate("", xy=(3.0, 1.0), xytext=(9.4, 1.35), arrowprops=dict(arrowstyle="-|>", color=RED, ls="--", connectionstyle="arc3,rad=-0.25"))
    ax.text(6.2, -0.15, "G learns from D's verdict: make fakes that D calls real", color=RED, fontsize=9.5, ha="center")
    save(fig, "11_gan")


def fig_diffusion():
    yy, xx = np.mgrid[0:48, 0:48]
    img = np.exp(-(((xx - 24) ** 2 + (yy - 20) ** 2) / 120.0)) + 0.8 * ((np.abs(xx - 24) < 14) & (np.abs(yy - 36) < 4))
    img = img / img.max()
    rng = np.random.RandomState(0); steps = [0, 0.25, 0.5, 0.75, 0.95, 1.0]
    fig, axes = plt.subplots(1, len(steps), figsize=(12, 2.6))
    for ax, a in zip(axes, steps):
        ab = 1 - a
        ax.imshow(np.sqrt(ab) * img + np.sqrt(1 - ab) * rng.randn(*img.shape) * 0.5, cmap="gray", vmin=-1, vmax=1.4)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(f"t = {int(a * 1000)}", fontsize=10)
    fig.suptitle("Forward: add noise step by step (fixed)  →→→      ←←←  Reverse: a U-Net removes it step by step (learned)", fontsize=11, y=1.04)
    save(fig, "11_diffusion")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("fig_") and callable(fn):
            fn()
