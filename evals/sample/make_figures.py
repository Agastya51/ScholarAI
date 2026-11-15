import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Figure 1: architecture block diagram
fig, ax = plt.subplots(figsize=(6, 2.2), dpi=200)
ax.axis("off")
boxes = [("Traffic\nreadings", 0.02), ("Diffusion\nconvolution\n(K = 2)", 0.22), ("Gated temporal\nattention\n(4 heads)", 0.44),
         ("Residual\nfusion", 0.68), ("12-step\nforecast", 0.86)]
for label, x in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.25), 0.13, 0.5, boxstyle="round,pad=0.01", fc="#dbe7f5", ec="#2f5d8a", lw=1.2))
    ax.text(x + 0.065, 0.5, label, ha="center", va="center", fontsize=7)
for (_, x0), (_, x1) in zip(boxes, boxes[1:]):
    ax.annotate("", xy=(x1 - 0.005, 0.5), xytext=(x0 + 0.14, 0.5), arrowprops=dict(arrowstyle="->", color="#2f5d8a"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig("fig_arch.png", bbox_inches="tight")

# Figure 2: error vs horizon
h = [15, 30, 45, 60]
fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=200)
ax.plot(h, [2.71, 3.08, 3.36, 3.62], marker="o", label="RecurNet")
ax.plot(h, [2.66, 2.99, 3.24, 3.47], marker="s", label="WaveGraph")
ax.plot(h, [2.58, 2.85, 3.02, 3.17], marker="^", label="FlowNet-X")
ax.set_xlabel("Prediction horizon (minutes)"); ax.set_ylabel("MAE (mph)")
ax.set_xticks(h); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.savefig("fig_horizon.png", bbox_inches="tight")
