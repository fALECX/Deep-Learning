"""Figures for the topic-flow cheat sheet (CHEAT_SHEET_BY_TOPIC.md). Run:
    python Lernen/_figures/make_figures_cheatsheet.py
Writes PNGs into Lernen/img/ with the prefix cs_. Same style and colour code as make_figures.py.
"""
import numpy as np
import matplotlib.pyplot as plt

from make_figures import BLUE, GREEN, INK, ORANGE, PURPLE, RED, SOFT, arrow, box, canvas, save


def fig_loss_pairs():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))

    p = np.linspace(0.01, 1, 300)
    a1.plot(p, -np.log(p), color=RED, lw=2.6, label="−ln p   (CE / BCE)")
    a1.plot(p, (1 - p) ** 2, color=SOFT, lw=2.2, ls="--", label="(1 − p)²   (MSE)")
    a1.set_xlabel("p = probability the model gives the TRUE answer")
    a1.set_ylabel("loss")
    a1.set_title("Loss for one example")
    a1.set_ylim(0, 4.6)
    a1.annotate("confident and wrong:\n−ln p explodes", xy=(0.03, 3.5), xytext=(0.25, 3.6),
                fontsize=9, color=RED, arrowprops=dict(arrowstyle="-|>", color=RED))
    a1.annotate("MSE stays ≤ 1", xy=(0.05, 0.9), xytext=(0.3, 1.6),
                fontsize=9, color=SOFT, arrowprops=dict(arrowstyle="-|>", color=SOFT))
    a1.legend(loc="upper right", fontsize=9)

    z = np.linspace(-7, 7, 400)
    s = 1 / (1 + np.exp(-z))
    a2.plot(z, np.abs(s - 1), color=RED, lw=2.6, label="sigmoid + BCE:  p − y")
    a2.plot(z, np.abs(2 * (s - 1) * s * (1 - s)), color=SOFT, lw=2.2, ls="--",
            label="sigmoid + MSE:  2(p − y)·p(1 − p)")
    a2.set_xlabel("raw score z   (true label y = 1)")
    a2.set_ylabel("size of the learning signal")
    a2.set_title("Learning signal at z")
    a2.axvline(-5, color="#e2e8f0", lw=1)
    a2.text(-6.8, 0.55, "z = −5, p = 0.007\nBCE: 0.99\nMSE: 0.013\n→ 75× weaker", fontsize=9, color=INK)
    a2.legend(loc="upper right", fontsize=9)
    save(fig, "cs_loss_pairs")


def fig_learning_loop():
    fig, ax = canvas(12, 3.4, (0, 12.4), (0, 3.4))
    steps = [
        (0.1, "input x", BLUE, "#eff6ff"),
        (2.1, "layers\nΣ w·x → f", ORANGE, "#fff7ed"),
        (4.1, "raw scores z\n(logits)", GREEN, "#f0fdf4"),
        (6.1, "output function\nsigmoid / softmax", GREEN, "#f0fdf4"),
        (8.1, "prediction p", GREEN, "#f0fdf4"),
        (10.1, "loss\n−ln p(true)", RED, "#fef2f2"),
    ]
    for x, text, c, fill in steps:
        box(ax, x, 1.5, 1.7, 0.9, text, c, fill=fill, fs=9)
    for x, _, _, _ in steps[:-1]:
        arrow(ax, x + 1.75, 1.95, x + 1.95, 1.95, color=BLUE)
    ax.text(5.0, 2.95, "① forward: compute the prediction and the loss", ha="center", color=BLUE, fontsize=10)
    box(ax, 10.1, 2.75, 1.7, 0.45, "label y", RED, fill="#fef2f2", fs=9)
    arrow(ax, 10.95, 2.75, 10.95, 2.45, color=RED)
    # backward path runs below the boxes: loss -> z -> layers
    ax.plot([10.95, 10.95], [1.45, 1.15], color=RED, lw=1.6, ls="--")
    arrow(ax, 10.95, 1.15, 5.0, 1.15, color=RED, ls="--")
    arrow(ax, 5.0, 1.15, 5.0, 1.45, color=RED)
    ax.text(8.0, 0.8, "② signal at z = p − y  (output function + loss together)", ha="center", color=RED, fontsize=9)
    arrow(ax, 4.9, 1.15, 3.0, 1.15, color=RED, ls="--")
    arrow(ax, 3.0, 1.15, 3.0, 1.45, color=RED)
    ax.text(3.0, 0.65, "③ backward: gradient of a weight\n= signal × value it multiplied", ha="center", color=RED, fontsize=9)
    ax.text(6.2, 0.1, "④ update every weight:  w ← w − η · gradient   → repeat with the next batch",
            ha="center", color=PURPLE, fontsize=10)
    save(fig, "cs_learning_loop")


if __name__ == "__main__":
    fig_loss_pairs()
    fig_learning_loop()
