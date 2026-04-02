"""
SimBA Hyperparameter Sweep — Results Visualisation
Reads sweeps/run_v1/trials.json + best_inference/metrics.json
Saves all plots to sweeps/run_v1/plots/
"""
import json, pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats

ROOT    = pathlib.Path("/scratch/gent/vo/000/gvo00017/vsc21162/simba/sweeps/run_v1")
PLOTDIR = ROOT / "plots"
PLOTDIR.mkdir(exist_ok=True)

with open(ROOT / "trials.json") as f:
    trials = json.load(f)
with open(ROOT / "best_inference" / "metrics.json") as f:
    inf = json.load(f)

ids     = [t["trial_number"] for t in trials]
values  = [t["value"]        for t in trials]
params  = [t["params"]       for t in trials]

BASELINE_IDX = 0
BEST_IDX     = int(np.argmin(values))
running_best = [min(values[:i+1]) for i in range(len(values))]

PARAM_KEYS  = list(params[0].keys())
PARAM_SHORT = ["lr", "wd", "d_model", "n_layers", "dropout"]

cmap = plt.cm.RdYlGn_r
norm = Normalize(vmin=min(values), vmax=max(values))

print(f"Baseline (trial {BASELINE_IDX}): val_loss = {values[BASELINE_IDX]:.4f}")
print(f"Best     (trial {BEST_IDX}):     val_loss = {values[BEST_IDX]:.4f}")
print(f"Saving plots to {PLOTDIR}\n")

# ── 1. Trial overview ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SimBA Hyperparameter Sweep — run_v1", fontsize=13, fontweight="bold")

colours = ["#888888"] * len(ids)
colours[BASELINE_IDX] = "#2196F3"
colours[BEST_IDX]     = "#4CAF50"

ax = axes[0]
ax.bar(ids, values, color=colours, edgecolor="white", linewidth=0.5, zorder=3)
ax.axhline(0, color="black", linewidth=0.6, linestyle="--", alpha=0.4)
ax.set_xlabel("Trial #"); ax.set_ylabel("val_loss  (lower = better)")
ax.set_title("val_loss per trial"); ax.set_xticks(ids); ax.grid(axis="y", alpha=0.3, zorder=0)
for idx, color in [(BASELINE_IDX, "#2196F3"), (BEST_IDX, "#4CAF50")]:
    ax.annotate(f"#{idx}\n{values[idx]:.3f}", xy=(idx, values[idx]),
                xytext=(0, 8 if values[idx] >= 0 else -24), textcoords="offset points",
                ha="center", fontsize=8, color=color, fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(color="#2196F3", label=f"Baseline (trial {BASELINE_IDX})"),
    mpatches.Patch(color="#4CAF50", label=f"Best (trial {BEST_IDX})"),
    mpatches.Patch(color="#888888", label="Other"),
], fontsize=8)

ax2 = axes[1]
ax2.plot(ids, values, "o--", color="#aaaaaa", alpha=0.6, linewidth=1, label="val_loss")
ax2.plot(ids, running_best, "s-", color="#E91E63", linewidth=2, label="running best", zorder=4)
ax2.scatter([BASELINE_IDX], [values[BASELINE_IDX]], s=120, color="#2196F3", zorder=5, label=f"Baseline (#{BASELINE_IDX})")
ax2.scatter([BEST_IDX], [values[BEST_IDX]], s=180, color="#4CAF50", marker="*", zorder=5, label=f"Best (#{BEST_IDX})")
ax2.set_xlabel("Trial #"); ax2.set_ylabel("val_loss")
ax2.set_title("Optimisation path (running best)"); ax2.set_xticks(ids)
ax2.grid(alpha=0.3); ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(PLOTDIR / "1_trial_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 1_trial_overview.png")

# ── 2. Hyperparameter scatter plots ──────────────────────────────────────────
param_values = {s: [p[k] for p in params] for s, k in zip(PARAM_SHORT, PARAM_KEYS)}

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Hyperparameter vs val_loss", fontsize=13, fontweight="bold")

for ax, short in zip(axes, PARAM_SHORT):
    xs = param_values[short]
    ax.scatter(xs, values, c=values, cmap=cmap, norm=norm,
               s=60, edgecolors="white", linewidth=0.5, zorder=3)
    for idx, marker, color in [(BASELINE_IDX, "D", "#2196F3"), (BEST_IDX, "*", "#4CAF50")]:
        ax.scatter([xs[idx]], [values[idx]], marker=marker, s=180,
                   color=color, zorder=5, edgecolors="black", linewidth=0.8)
        ax.annotate(f"#{idx}", xy=(xs[idx], values[idx]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=color, fontweight="bold")
    ax.set_xlabel(short, fontsize=10)
    ax.set_ylabel("val_loss" if ax is axes[0] else "")
    ax.grid(alpha=0.3)
    if short in ("lr", "wd"):
        ax.set_xscale("log")

fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes[-1], label="val_loss", shrink=0.8)
axes[0].legend(handles=[
    mpatches.Patch(color="#2196F3", label=f"Baseline (#{BASELINE_IDX})"),
    mpatches.Patch(color="#4CAF50", label=f"Best (#{BEST_IDX})"),
], fontsize=7)

plt.tight_layout()
plt.savefig(PLOTDIR / "2_hyperparam_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 2_hyperparam_scatter.png")

# ── 3. Parallel coordinates ───────────────────────────────────────────────────
def norm_col(arr):
    arr = np.array(arr, dtype=float)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)

cols_raw = {s: np.array(param_values[s], dtype=float) for s in PARAM_SHORT}
cols_raw["val_loss"] = np.array(values)
col_order = PARAM_SHORT + ["val_loss"]
cols_norm = {k: norm_col(v) for k, v in cols_raw.items()}
for s in ("lr", "wd"):
    cols_norm[s] = norm_col(np.log10(cols_raw[s]))

x_pos = np.arange(len(col_order))
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle("Parallel coordinates — hyperparameter landscape", fontsize=13, fontweight="bold")

for i in range(len(trials)):
    ys    = [cols_norm[c][i] for c in col_order]
    color = cmap(norm(values[i]))
    lw    = 0.8 if i not in (BASELINE_IDX, BEST_IDX) else 2.5
    alpha = 0.4 if i not in (BASELINE_IDX, BEST_IDX) else 1.0
    ax.plot(x_pos, ys, color=color, linewidth=lw, alpha=alpha)

for idx, color, label in [
    (BASELINE_IDX, "#2196F3", f"Baseline (#{BASELINE_IDX}, val={values[BASELINE_IDX]:.3f})"),
    (BEST_IDX,     "#4CAF50", f"Best     (#{BEST_IDX}, val={values[BEST_IDX]:.3f})"),
]:
    ys = [cols_norm[c][idx] for c in col_order]
    ax.plot(x_pos, ys, color=color, linewidth=3, label=label, zorder=5)

ax.set_xticks(x_pos)
ax.set_xticklabels(["lr (log)", "wd (log)", "d_model", "n_layers", "dropout", "val_loss"], fontsize=10)
ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["min", "mid", "max"])
ax.set_ylabel("Normalised value"); ax.grid(axis="x", alpha=0.4); ax.legend(fontsize=9)
fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="val_loss", shrink=0.8, pad=0.01)

plt.tight_layout()
plt.savefig(PLOTDIR / "3_parallel_coords.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 3_parallel_coords.png")

# ── 4. Parity plots (best model) ──────────────────────────────────────────────
def parity_ax(ax, fig, true, pred, r, title, color):
    true, pred = np.array(true), np.array(pred)
    from matplotlib.colors import LogNorm
    hb = ax.hexbin(true, pred, gridsize=40, cmap="Blues", mincnt=1, linewidths=0.2,
                   norm=LogNorm())
    fig.colorbar(hb, ax=ax, label="count (log)")
    lo = min(true.min(), pred.min()); hi = max(true.max(), pred.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo-margin, hi+margin], [lo-margin, hi+margin], "r--", linewidth=1.5, label="y = x")
    slope, intercept, *_ = stats.linregress(true, pred)
    xs = np.array([lo-margin, hi+margin])
    ax.plot(xs, slope*xs + intercept, "-", color=color, linewidth=2, label=f"fit  (r={r:.3f})")
    ax.set_xlim(lo-margin, hi+margin); ax.set_ylim(lo-margin, hi+margin)
    ax.set_xlabel("True similarity"); ax.set_ylabel("Predicted similarity")
    ax.set_title(title, fontsize=11); ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=0.6)

bp = params[BEST_IDX]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle(
    f"Best model  (trial #{BEST_IDX} · lr={bp['optimizer.lr']:.2e} · "
    f"d_model={bp['model.transformer.d_model']} · n_layers={bp['model.transformer.n_layers']})",
    fontsize=11, fontweight="bold"
)
parity_ax(axes[0], fig, inf["ed_true"],   inf["ed_pred"],
          inf["ed_correlation"],   f"Edit Distance (ED)  ·  Pearson r = {inf['ed_correlation']:.4f}", "#E91E63")
parity_ax(axes[1], fig, inf["mces_true"], inf["mces_pred"],
          inf["mces_correlation"], f"MCES  ·  Pearson r = {inf['mces_correlation']:.4f}", "#FF9800")

plt.tight_layout()
plt.savefig(PLOTDIR / "4_best_model_parity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 4_best_model_parity.png")

# ── 5. Summary table ──────────────────────────────────────────────────────────
print("\n── Summary (sorted by val_loss) ─────────────────────────────────────────")
header = f"{'#':>4}  {'val_loss':>10}  {'lr':>12}  {'wd':>12}  {'d_model':>7}  {'n_lay':>5}  {'drop':>6}  note"
print(header)
print("-" * len(header))
for t in sorted(trials, key=lambda x: x["value"]):
    i = t["trial_number"]; p = t["params"]
    note = ("◀ baseline" if i == BASELINE_IDX else "") + ("★ BEST" if i == BEST_IDX else "")
    print(f"{i:>4}  {t['value']:>10.4f}  {p['optimizer.lr']:>12.2e}  {p['optimizer.weight_decay']:>12.2e}  "
          f"{p['model.transformer.d_model']:>7}  {p['model.transformer.n_layers']:>5}  "
          f"{p['model.transformer.dropout']:>6.3f}  {note}")
print(f"\nBest model validation metrics:")
print(f"  ED   Pearson r = {inf['ed_correlation']:.4f}")
print(f"  MCES Pearson r = {inf['mces_correlation']:.4f}")
print(f"\nAll done — plots in {PLOTDIR}")
