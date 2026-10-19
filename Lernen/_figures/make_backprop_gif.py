"""Animated walk-through of the Week 4 tutorial Q1 backprop example (forward, backward, update, check).
python Lernen/_figures/make_backprop_gif.py  ->  Lernen/img/01_backprop_example.gif + one still per step (01_bp_*.png)
"""
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from PIL import Image

from make_figures import OUT, BLUE, ORANGE, GREEN, RED, PURPLE, GREY, INK, SOFT

# ----- the numbers (computed, not typed) -----
x = [1.0, 2.0]; y = 3.0; lr = 0.1
W = [[2.0, 1.0], [1.0, -2.0], [1.0, 2.0]]      # hidden unit k: weights from x1, x2
v = [-1.0, 3.0, 2.0]                            # output weights
z = [W[k][0] * x[0] + W[k][1] * x[1] for k in range(3)]
h = [max(0.0, zk) for zk in z]
yhat = sum(v[k] * h[k] for k in range(3))
L = (yhat - y) ** 2
d_out = 2 * (yhat - y)                          # dL/dyhat
g_v = [d_out * h[k] for k in range(3)]
d_h = [d_out * v[k] for k in range(3)]
relu_slope = [1.0 if zk > 0 else 0.0 for zk in z]
d_z = [d_h[k] * relu_slope[k] for k in range(3)]
g_W = [[d_z[k] * x[0], d_z[k] * x[1]] for k in range(3)]


def step(rate):
    v2 = [v[k] - rate * g_v[k] for k in range(3)]
    b_o = -rate * d_out
    W2 = [[W[k][i] - rate * g_W[k][i] for i in range(2)] for k in range(3)]
    b_h = [-rate * d_z[k] for k in range(3)]
    z2 = [W2[k][0] * x[0] + W2[k][1] * x[1] + b_h[k] for k in range(3)]
    h2 = [max(0.0, zk) for zk in z2]
    y2 = sum(v2[k] * h2[k] for k in range(3)) + b_o
    return v2, W2, y2, (y2 - y) ** 2, h2


v_new, W_new, y_big, L_big, h_big = step(0.1)
_, _, y_small, L_small, _ = step(0.001)
print("z", z, "h", h, "yhat", yhat, "L", L)
print("g_v", g_v, "d_h", d_h, "d_z", d_z, "g_W", g_W)
print("after lr=0.1:", round(y_big, 2), round(L_big, 1), " after lr=0.001:", round(y_small, 3), round(L_small, 2))

f = lambda a: f"{a:g}"

# ----- layout -----
POS = {"x1": (1.2, 5.0), "x2": (1.2, 2.2), "h1": (5.4, 6.0), "h2": (5.4, 3.6), "h3": (5.4, 1.2), "y": (9.4, 3.6)}
HID = ["h1", "h2", "h3"]


def frame(title, text, vals=None, hi=(), red=None, green=None, dead=False, gate=False, lossbox=None, colour=BLUE):
    vals = vals or {}; red = red or {}; green = green or {}
    fig, ax = plt.subplots(figsize=(12.4, 9.2))
    ax.set_xlim(0, 13); ax.set_ylim(-2.3, 7.3); ax.axis("off"); ax.set_aspect("equal")
    # edges
    for k, hn in enumerate(HID):
        for i, xn in enumerate(["x1", "x2"]):
            key = f"w{k + 1}{i + 1}"
            (x0, y0), (x1, y1) = POS[xn], POS[hn]
            on = key in hi
            ax.plot([x0, x1], [y0, y1], color=ORANGE if on else "#cbd5e1", lw=3 if on else 1.6, zorder=1)
            mx, my = x0 + 0.3 * (x1 - x0), y0 + 0.3 * (y1 - y0) + 0.15
            lab = green.get(key, f"w{k + 1}{i + 1} = {W[k][i]:g}")
            ax.text(mx, my, lab, fontsize=11, ha="center", color=GREEN if key in green else INK,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=3)
            if key in red:
                ax.text(mx + 0.95, my, red[key], fontsize=10.5, ha="left", color=RED, weight="bold",
                        bbox=dict(boxstyle="round,pad=0.15", fc="#fef2f2", ec="none"), zorder=3)
        (x0, y0), (x1, y1) = POS[hn], POS["y"]
        key = f"v{k + 1}"; on = key in hi
        ax.plot([x0, x1], [y0, y1], color=ORANGE if on else "#cbd5e1", lw=3 if on else 1.6, zorder=1)
        mx, my = x0 + 0.5 * (x1 - x0), y0 + 0.5 * (y1 - y0) + 0.22
        lab = green.get(key, f"v{k + 1} = {v[k]:g}")
        ax.text(mx, my, lab, fontsize=11, ha="center", color=GREEN if key in green else INK,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"), zorder=3)
        if key in red:
            ax.text(mx, my - 0.45, red[key], fontsize=11, ha="center", color=RED, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#fef2f2", ec="none"), zorder=3)
    # nodes
    names = {"x1": "x₁", "x2": "x₂", "h1": "h₁", "h2": "h₂", "h3": "h₃", "y": "ŷ"}
    for n, (px, py) in POS.items():
        col = BLUE if n.startswith("x") else (GREEN if n == "y" else ORANGE)
        fill = {"x": "#eff6ff", "h": "#fff7ed", "y": "#f0fdf4"}[n[0]]
        if dead and n == "h2":
            col, fill = GREY, "#e2e8f0"
        ax.add_patch(Circle((px, py), 0.62, fc=fill, ec=col, lw=2.4, zorder=4))
        ax.text(px, py + 0.18, names[n], ha="center", va="center", fontsize=12, color=SOFT, zorder=5)
        if n in vals:
            ax.text(px, py - 0.2, vals[n], ha="center", va="center", fontsize=13, weight="bold", zorder=5)
        if n in red:
            ax.text(px + (0.0 if n == "y" else -0.05), py - 1.0, red[n], ha="center", fontsize=11, color=RED, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="#fef2f2", ec=RED, lw=1), zorder=6)
    if gate:
        for k, hn in enumerate(HID):
            px, py = POS[hn]
            ok = relu_slope[k] > 0
            ax.text(px + 1.05, py + 0.55, "ReLU slope 1" if ok else "ReLU slope 0", fontsize=10,
                    color=GREEN if ok else GREY, weight="bold", zorder=6)
    # target and loss
    ax.add_patch(FancyBboxPatch((10.75, 3.0), 2.0, 1.2, boxstyle="round,pad=0.05", fc="#fef2f2", ec=RED, lw=1.6))
    ax.text(11.75, 3.85, "target y = 3", ha="center", fontsize=11, color=RED)
    ax.text(11.75, 3.3, lossbox or "loss L = ?", ha="center", fontsize=11.5, weight="bold")
    ax.plot([10.02, 10.75], [3.6, 3.6], color="#cbd5e1", lw=1.6)
    # caption
    ax.add_patch(FancyBboxPatch((0.2, -2.2), 12.6, 1.95, boxstyle="round,pad=0.05", fc="#f8fafc", ec=colour, lw=2))
    ax.text(0.45, -0.55, title, fontsize=13, weight="bold", color=colour, va="center")
    ax.text(0.45, -1.35, text, fontsize=11.5, va="center", color=INK, linespacing=1.6)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=90, facecolor="white"); plt.close(fig)
    buf.seek(0); return Image.open(buf).convert("RGB")


V = {"x1": "1", "x2": "2"}
frames, dur = [], []


def add(img, seconds):
    frames.append(img); dur.append(int(seconds * 1000))


add(frame("The network", "Inputs x = (1, 2), target y = 3. Numbers on the lines are the weights. All biases are 0.", V), 4)
V1 = dict(V, h1=f(h[0]))
add(frame("Forward 1 · hidden neuron 1", f"z₁ = 2·1 + 1·2 = {f(z[0])}   →   h₁ = ReLU({f(z[0])}) = {f(h[0])}", V1, hi={"w11", "w12"}), 4.5)
V2 = dict(V1, h2=f(h[1]))
add(frame("Forward 2 · hidden neuron 2", f"z₂ = 1·1 + (−2)·2 = {f(z[1])}   →   h₂ = ReLU({f(z[1])}) = 0\nNegative in, 0 out: this neuron is 'dead' for this example (grey).",
          V2, hi={"w21", "w22"}, dead=True), 5)
V3 = dict(V2, h3=f(h[2]))
add(frame("Forward 3 · hidden neuron 3", f"z₃ = 1·1 + 2·2 = {f(z[2])}   →   h₃ = ReLU({f(z[2])}) = {f(h[2])}", V3, hi={"w31", "w32"}, dead=True), 4.5)
V4 = dict(V3, y=f(yhat))
add(frame("Forward 4 · output", f"ŷ = (−1)·4 + 3·0 + 2·5 = −4 + 0 + 10 = {f(yhat)}", V4, hi={"v1", "v2", "v3"}, dead=True), 4.5)
LB = f"L = (6 − 3)² = {f(L)}"
add(frame("Forward 5 · loss", f"L = (ŷ − y)² = (6 − 3)² = {f(L)}\nThe prediction 6 is too big; the target is 3. Now: which weights should change, and how?", V4, dead=True, lossbox=LB), 5)

R1 = {"y": "∂L/∂ŷ = 6"}
add(frame("Backward 1 · start at the loss", f"∂L/∂ŷ = 2·(ŷ − y) = 2·(6 − 3) = {f(d_out)}   (the slope of the loss at ŷ = 6)\nPositive: a bigger ŷ makes the loss bigger, so ŷ must go down. This 6 now travels backwards.",
          V4, red=R1, dead=True, lossbox=LB, colour=RED), 6)
R2 = dict(R1, v1=f"grad {f(g_v[0])}", v2=f"grad {f(g_v[1])}", v3=f"grad {f(g_v[2])}")
add(frame("Backward 2 · output weights", f"ŷ = v₁h₁ + v₂h₂ + v₃h₃, so v_k changes ŷ by h_k.   grad of v_k = 6 · h_k\nv₁: 6·4 = {f(g_v[0])}     v₂: 6·0 = {f(g_v[1])}     v₃: 6·5 = {f(g_v[2])}     output bias: 6·1 = 6",
          V4, hi={"v1", "v2", "v3"}, red=R2, dead=True, lossbox=LB, colour=RED), 7)
R3 = dict(R2, h1=f"signal {f(d_h[0])}", h2=f"signal {f(d_h[1])}", h3=f"signal {f(d_h[2])}")
add(frame("Backward 3 · send the 6 back to each hidden neuron", f"h_k changes ŷ by v_k, so the signal at h_k = 6 · v_k\nh₁: 6·(−1) = {f(d_h[0])}     h₂: 6·3 = {f(d_h[1])}     h₃: 6·2 = {f(d_h[2])}",
          V4, hi={"v1", "v2", "v3"}, red=R3, dead=True, lossbox=LB, colour=RED), 7)
R4 = dict(R2, h1=f"{f(d_h[0])}·1 = {f(d_z[0])}", h2=f"{f(d_h[1])}·0 = 0", h3=f"{f(d_h[2])}·1 = {f(d_z[2])}")
add(frame("Backward 4 · through the ReLU", "Multiply by the ReLU slope: 1 if z was positive, 0 if z was negative.\nh₁: −6·1 = −6     h₂ had z = −3: 18·0 = 0, blocked     h₃: 12·1 = 12",
          V4, red=R4, dead=True, gate=True, lossbox=LB, colour=RED), 7)
R5 = dict(R4, **{f"w{k + 1}{i + 1}": f"grad {f(g_W[k][i])}" for k in range(3) for i in range(2)})
add(frame("Backward 5 · hidden weights", "grad of a hidden weight = signal at its neuron × input on that line (x₁ = 1, x₂ = 2)\nto h₁: −6·1 = −6, −6·2 = −12     to h₂: 0, 0     to h₃: 12·1 = 12, 12·2 = 24",
          V4, hi={"w11", "w12", "w31", "w32"}, red=R5, dead=True, lossbox=LB, colour=RED), 8)
G = {f"w{k + 1}{i + 1}": f"w{k + 1}{i + 1} = {W_new[k][i]:.1f}" for k in range(3) for i in range(2)}
G.update({f"v{k + 1}": f"v{k + 1} = {v_new[k]:.1f}" for k in range(3)})
add(frame("Update · new weight = old weight − 0.1 · grad   (green = new values)", "v₁: −1 − 0.1·24 = −3.4     v₃: 2 − 0.1·30 = −1     w₁₁: 2 − 0.1·(−6) = 2.6     w₃₂: 2 − 0.1·24 = −0.4\nNegative grad → weight goes up. Positive grad → weight goes down. Grad 0 → no change (h₂'s weights).",
          V4, green=G, dead=True, lossbox=LB, colour=GREEN), 9)
add(frame("Check · run the forward pass again with the new weights", f"Every change points the right way, but η = 0.1 is far too big: ŷ jumps from 6 to {y_big:.1f} (overshoot).\nWith η = 0.001 instead: ŷ = {y_small:.2f} and the loss drops from 9 to {L_small:.1f}. The gradient gives the direction, η the step size.",
          dict(V, y=f"{y_big:.1f}", h1=f"{h_big[0]:.1f}", h2=f"{h_big[1]:g}", h3=f"{h_big[2]:g}"), green=G, lossbox=f"L = {L_big:.0f}  (was 9)", colour=PURPLE), 10)

frames[0].save(OUT / "01_backprop_example.gif", save_all=True, append_images=frames[1:], duration=dur, loop=0, optimize=True)
frames[-4].save(OUT / "01_backprop_example_still.png")
# one still per step, placed next to the matching step in doc 01
for idx, name in [(3, "a3"), (5, "a5"), (6, "b1"), (7, "b2"), (8, "b3"), (9, "b4"), (10, "b5"), (11, "c"), (12, "d")]:
    frames[idx].save(OUT / f"01_bp_{name}.png", optimize=True)
print("wrote 01_backprop_example.gif with", len(frames), "frames,", sum(dur) / 1000, "s")
