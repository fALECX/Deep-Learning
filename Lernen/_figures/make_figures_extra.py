"""Extra figures and animations for chapter 01 (run after make_figures.py).
python Lernen/_figures/make_figures_extra.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from make_figures import (OUT, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, INK, SOFT, save, box, arrow, canvas)


# A loss surface with a long, narrow valley (like real networks): steep across, shallow along.
def loss(w1, w2):
    return 0.15 * (w1 - 3) ** 2 + 2.0 * (w2 - 1) ** 2 + 0.3 * np.sin(1.5 * w1) * np.cos(1.2 * w2)


def grad(w):
    w1, w2 = w
    return np.array([0.3 * (w1 - 3) + 0.45 * np.cos(1.5 * w1) * np.cos(1.2 * w2),
                     4.0 * (w2 - 1) - 0.36 * np.sin(1.5 * w1) * np.sin(1.2 * w2)])


def run(kind, steps=60, start=(-3.5, 2.6), lr=0.06, seed=0):
    rng = np.random.RandomState(seed)
    w = np.array(start, float); path = [w.copy()]; v = np.zeros(2); m = np.zeros(2); s = np.zeros(2)
    for t in range(1, steps + 1):
        g = grad(w)
        if kind == "SGD (noisy mini-batches)":
            g = g + rng.randn(2) * 1.2
            w = w - lr * g
        elif kind == "plain gradient descent":
            w = w - lr * g
        elif kind == "momentum":
            v = 0.85 * v + g; w = w - lr * 0.6 * v
        elif kind == "Adam":
            m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g ** 2
            mh, sh = m / (1 - 0.9 ** t), s / (1 - 0.999 ** t)
            w = w - 0.25 * mh / (np.sqrt(sh) + 1e-8)
        path.append(w.copy())
    return np.array(path)


KINDS = [("plain gradient descent", BLUE), ("SGD (noisy mini-batches)", SOFT), ("momentum", ORANGE), ("Adam", PURPLE)]
g1, g2 = np.meshgrid(np.linspace(-4.5, 6.5, 300), np.linspace(-1.5, 3.2, 300))
Z = loss(g1, g2)


def fig_loss_landscape_3d():
    fig = plt.figure(figsize=(12, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(g1, g2, Z, cmap="viridis", alpha=0.75, linewidth=0, antialiased=True, rstride=6, cstride=6)
    p = run("plain gradient descent")
    ax.plot(p[:, 0], p[:, 1], loss(p[:, 0], p[:, 1]) + 0.05, color=RED, lw=2.5, marker="o", ms=3)
    ax.set_xlabel("weight w₁"); ax.set_ylabel("weight w₂"); ax.set_zlabel("loss")
    ax.set_title("The loss is a landscape over the weights"); ax.view_init(elev=35, azim=-60)
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contourf(g1, g2, Z, levels=30, cmap="viridis", alpha=0.85)
    ax2.contour(g1, g2, Z, levels=30, colors="white", linewidths=0.4, alpha=0.5)
    ax2.plot(p[:, 0], p[:, 1], color=RED, lw=2.2, marker="o", ms=3)
    ax2.scatter([p[0, 0]], [p[0, 1]], color="white", ec=INK, s=70, zorder=5); ax2.text(p[0, 0] + 0.2, p[0, 1] + 0.15, "start", color="white")
    ax2.set_title("Seen from above (contour lines = equal loss)"); ax2.set_xlabel("weight w₁"); ax2.set_ylabel("weight w₂")
    save(fig, "01_loss_landscape")


def gif_optimisers():
    paths = {k: run(k) for k, _ in KINDS}
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.contourf(g1, g2, Z, levels=30, cmap="Greys", alpha=0.35)
    ax.contour(g1, g2, Z, levels=20, colors="#94a3b8", linewidths=0.5)
    ax.scatter([3], [1], marker="*", s=220, color=GREEN, zorder=6); ax.text(3.15, 1.2, "minimum", color=GREEN)
    lines = {k: ax.plot([], [], color=c, lw=2.2, label=k)[0] for k, c in KINDS}
    dots = {k: ax.plot([], [], "o", color=c, ms=7)[0] for k, c in KINDS}
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9); ax.set_xticks([]); ax.set_yticks([])
    title = ax.set_title("")

    def update(i):
        for k, _ in KINDS:
            p = paths[k][: i + 1]; lines[k].set_data(p[:, 0], p[:, 1]); dots[k].set_data([p[-1, 0]], [p[-1, 1]])
        title.set_text(f"Four optimisers on the same loss landscape · step {i}")
        return list(lines.values()) + list(dots.values()) + [title]

    anim = FuncAnimation(fig, update, frames=61, interval=120, blit=False)
    anim.save(OUT / "01_optimisers.gif", writer=PillowWriter(fps=8), dpi=90)
    plt.close(fig); print("wrote 01_optimisers.gif")


def fig_neuron():
    fig, ax = canvas(10, 3.6, (0, 10), (0, 3.6))
    for k, (y, lab, w) in enumerate([(2.9, "x₁", "w₁"), (1.8, "x₂", "w₂"), (0.7, "x₃", "w₃")]):
        ax.add_patch(plt.Circle((0.6, y), 0.32, color="#eff6ff", ec=BLUE, lw=2)); ax.text(0.6, y, lab, ha="center", va="center")
        arrow(ax, 0.95, y, 3.55, 1.8 + (y - 1.8) * 0.25, color=ORANGE, lw=1.4 + k * 0.6)
        ax.text(2.1, y - 0.05 + (1.8 - y) * 0.35, w, color=ORANGE, fontsize=10)
    ax.add_patch(plt.Circle((4.2, 1.8), 0.65, color="#f8fafc", ec=GREY, lw=2)); ax.text(4.2, 1.8, "Σ + b", ha="center", va="center", fontsize=11)
    ax.text(4.2, 0.85, "weighted sum z", ha="center", fontsize=9, color=SOFT)
    arrow(ax, 4.85, 1.8, 5.8, 1.8)
    box(ax, 5.8, 1.35, 1.6, 0.9, "f(z)\nactivation", GREY, fs=10)
    arrow(ax, 7.4, 1.8, 8.4, 1.8, color=GREEN)
    ax.add_patch(plt.Circle((8.9, 1.8), 0.42, color="#f0fdf4", ec=GREEN, lw=2)); ax.text(8.9, 1.8, "a", ha="center", va="center", fontsize=12)
    ax.text(5.0, 3.35, "One neuron: thicker arrow = larger weight = that input counts more", ha="center", fontsize=10.5, weight="bold")
    save(fig, "01_neuron")


def fig_mlp():
    fig, ax = canvas(9, 4.2, (0, 9), (0, 4.4))
    layers = [(1.0, 3, BLUE, "input layer"), (3.4, 5, GREY, "hidden layer 1"), (5.8, 5, GREY, "hidden layer 2"), (8.0, 2, GREEN, "output layer")]
    pos = []
    for x, n, c, name in layers:
        ys = np.linspace(0.6, 3.6, n) if n > 1 else [2.1]
        if n == 2: ys = [1.5, 2.7]
        if n == 3: ys = [1.2, 2.1, 3.0]
        pos.append([(x, y) for y in ys])
        ax.text(x, 4.15, name, ha="center", fontsize=9.5, color=c if c != GREY else INK)
    for a, b in zip(pos, pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color=ORANGE, lw=0.7, alpha=0.5)
    for (x, n, c, _), ps in zip(layers, pos):
        for (xx, yy) in ps:
            ax.add_patch(plt.Circle((xx, yy), 0.22, color="white", ec=c, lw=2, zorder=3))
    ax.text(4.6, 0.05, "every orange line is one weight · each layer: h = f(W·x + b)", ha="center", fontsize=9.5, color=SOFT)
    save(fig, "01_mlp")


def fig_dropout():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    rng = np.random.RandomState(4)
    for ax, drop, title in [(axes[0], False, "training step without dropout"), (axes[1], True, "training step with dropout (p = 0.5)")]:
        ax.set_xlim(0, 6); ax.set_ylim(0, 4); ax.axis("off"); ax.set_aspect("equal"); ax.set_title(title, fontsize=11)
        layers = [[(0.8, y) for y in (1, 2, 3)], [(2.6, y) for y in (0.5, 1.25, 2.0, 2.75, 3.5)], [(4.4, y) for y in (0.5, 1.25, 2.0, 2.75, 3.5)], [(5.6, 2.0)]]
        off = [set(), set(rng.choice(5, 2, replace=False)) if drop else set(), set(rng.choice(5, 3, replace=False)) if drop else set(), set()]
        for li, (a, b) in enumerate(zip(layers, layers[1:])):
            for i, (x1, y1) in enumerate(a):
                for j, (x2, y2) in enumerate(b):
                    if i in off[li] or j in off[li + 1]: continue
                    ax.plot([x1, x2], [y1, y2], color=ORANGE, lw=0.8, alpha=0.6)
        for li, L in enumerate(layers):
            for i, (x, y) in enumerate(L):
                dead = i in off[li]
                ax.add_patch(plt.Circle((x, y), 0.2, color="#f1f5f9" if dead else "white", ec="#cbd5e1" if dead else INK, lw=1.6, zorder=3))
                if dead: ax.text(x, y, "×", ha="center", va="center", color=RED, fontsize=12, zorder=4)
    save(fig, "01_dropout")


def fig_backprop():
    fig, ax = canvas(11, 3.0, (0, 11), (0, 3.0))
    nodes = [(0.2, "w", ORANGE, "#fff7ed"), (2.5, "z = w·x + b", GREY, "white"), (5.0, "a = σ(z)", GREY, "white"), (7.5, "L = loss(a, y)", RED, "#fef2f2")]
    for x, t, c, f in nodes: box(ax, x, 1.6, 1.9, 0.75, t, c, fill=f, fs=10)
    for (x1, *_), (x2, *_) in zip(nodes, nodes[1:]): arrow(ax, x1 + 1.9, 2.15, x2, 2.15, color=BLUE)
    for (x1, *_), (x2, *_) in zip(nodes, nodes[1:]): arrow(ax, x2, 1.8, x1 + 1.9, 1.8, color=RED, ls="--")
    ax.text(5.4, 2.65, "forward pass: compute the loss", color=BLUE, ha="center", fontsize=10)
    ax.text(5.4, 1.2, "backward pass: multiply local slopes  ∂L/∂a · ∂a/∂z · ∂z/∂w", color=RED, ha="center", fontsize=10)
    ax.text(5.4, 0.55, "the product tells w which way to move (gradient), then w ← w − η·gradient", ha="center", fontsize=9.5, color=SOFT)
    save(fig, "01_backprop")


if __name__ == "__main__":
    fig_neuron(); fig_mlp(); fig_loss_landscape_3d(); gif_optimisers(); fig_dropout(); fig_backprop()


def fig_activation_applied():
    z = np.array([-2.0, 0.5, 3.0, -0.7])
    relu = np.maximum(0, z); sig = 1 / (1 + np.exp(-z))
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9), sharey=False)
    labels = ["neuron 1", "neuron 2", "neuron 3", "neuron 4"]
    for ax, vals, title, c in [(axes[0], z, "weighted sums z (before)", GREY), (axes[1], relu, "after ReLU: negatives → 0", ORANGE),
                               (axes[2], sig, "after sigmoid: squashed to 0…1", GREEN)]:
        ax.bar(labels, vals, color=c); ax.axhline(0, color="#94a3b8", lw=1); ax.set_title(title, fontsize=11)
        for i, v in enumerate(vals): ax.text(i, v + (0.12 if v >= 0 else -0.35), f"{v:.2f}", ha="center", fontsize=9)
        ax.set_ylim(-2.6, 3.6); ax.tick_params(axis="x", labelsize=8.5)
    save(fig, "01_activation_applied")


if __name__ == "__main__":
    fig_activation_applied()
