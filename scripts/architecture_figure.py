"""LeNet-style schematic of the pixel-graph-time CNN (docs/pixel_graph_time_cnn.png).

    uv run --with matplotlib python scripts/architecture_figure.py docs/pixel_graph_time_cnn.png
"""

import sys

import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

GRAY, DARK, LINE = "#8c8c8c", "#5f5f5f", "#b8b8b8"
PIXEL_Y, NODE_Y = 62, 36  # row centres of the 375 m and 0.75 km stages

fig = plt.figure(figsize=(19, 7))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 304)
ax.set_ylim(0, 110.5)
ax.axis("off")
ax.text(150, 104, "Pixel-graph-time CNN", ha="center", va="center", fontsize=26, fontweight="bold", color="#111")


def stack(x, yc, size, n, label, shade=GRAY, dx=2.4, dy=2.4):
    """n slabs of side `size`, front slab at (x, yc - size / 2). Returns (front corner, back corner, size)."""

    corners = []
    for i in reversed(range(n)):
        corner = (x + i * dx, yc - size / 2 + i * dy)
        ax.add_patch(Rectangle(corner, size, size, facecolor=shade, edgecolor="white", linewidth=1.6))
        corners.append(corner)
    ax.text(x + size / 2 + (n - 1) * dx / 2, yc + size / 2 + (n - 1) * dy + 1.8, label,
            ha="center", va="bottom", fontsize=10.5, color="#111", linespacing=1.25)
    return corners[-1], corners[0], size


def connect(src, dst, window=None):
    """A receptive window on the source front slab, joined by two thin lines to the destination back slab."""

    (fx, fy), _, fs = src
    _, (bx, by), bs = dst
    w = window or fs * 0.28
    wx, wy = fx + fs * 0.55, fy + fs * 0.35
    ax.add_patch(Rectangle((wx, wy), w, w, facecolor="none", edgecolor="white", linewidth=1.2))
    for (px, py), (qx, qy) in (((wx + w, wy), (bx, by)), ((wx + w, wy + w), (bx, by + bs))):
        ax.add_line(Line2D([px, qx], [py, qy], color=LINE, linewidth=0.9, zorder=0))


def op(x, y, text):
    ax.text(x, y, text, ha="center", va="top", fontsize=10.5, color="#222", linespacing=1.25)


# Inputs
px_in = stack(4, PIXEL_Y, 28, 5, "PIXEL INPUT\n5 days · 7 @ 2H×2W", shade=DARK)
ax.text(18, PIXEL_Y - 9, "375 m", ha="center", va="center", fontsize=20, color="white", fontweight="bold")
nd_in = stack(4, 14, 14, 5, "NODE INPUT\n5 days · 45 @ H×W", shade=DARK)
ax.text(11, 10, "0.75 km", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

# Pixel encoder
p1 = stack(50, PIXEL_Y, 28, 5, "P1: pixel maps\n16 @ 2H×2W")
connect(px_in, p1)
op(41, 45, "2 × conv 3×3")

# Pool + concat -> node maps -> 1×1
n0 = stack(94, NODE_Y, 14, 5, "N0\n61 @ H×W")
connect(p1, n0, window=4)
connect(nd_in, n0)
op(74, 20, "pool 2×2\n+ concat")
n1 = stack(116, NODE_Y, 14, 5, "N1\n64 @ H×W")
connect(n0, n1, window=2)
op(107, 20, "1×1")

# Graph-time layers
g = []
for i, x in enumerate((138, 156, 174, 192)):
    g.append(stack(x, NODE_Y, 14, 5, f"G{i + 1}\n64 @ H×W"))
for a, b in zip([n1] + g[:-1], g):
    connect(a, b)
ax.text(170, 77, "4 × graph-time layers", ha="center", va="bottom", fontsize=11.5, color="#111", fontweight="bold")
ax.text(170, 73.5, "S: attention over the 8 neighbours, per node and day  ·  S_T: previous day  ·  4 heads  ·  3 taps",
        ha="center", va="bottom", fontsize=9.5, color="#444")
ax.plot([137, 212], [72, 72], color="#444", linewidth=0.8)
op(170, 20, "graph-time convolutions")

# Last-day node states, upsample + skip from P1
last = stack(214, NODE_Y, 14, 1, "last day\n64 @ H×W")
connect(g[3], last, window=2)
u = stack(232, PIXEL_Y, 28, 2, "U: 64 + 16 @ 2H×2W", dx=1.6, dy=1.6)
connect(last, u, window=3)
op(223, 20, "upsample ×2\n+ skip from P1")
ax.add_line(Line2D([p1[1][0] + 28, u[1][0]], [p1[1][1] + 28, u[1][1] + 28], color=LINE, linewidth=0.9, linestyle="--", zorder=0))

# Decoder
d = stack(268, PIXEL_Y, 14, 2, "D: 32 @ 2H×2W", dx=1.6, dy=1.6)
connect(u, d, window=3)
op(263, 45, "conv 3×3")
out = stack(290, PIXEL_Y, 7, 1, "OUTPUT\n1 @ 2H×2W", shade=DARK)
connect(d, out, window=2)
op(286, 45, "1×1")

fig.savefig(sys.argv[1], dpi=150, facecolor="white")
