"""Figures for the report (static PNG, light surface)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RES, ROOT, jload

FIG = f"{ROOT}/figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                     "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e8e7e2", "grid.linewidth": 0.6, "axes.axisbelow": True})
B = jload(f"{RES}/behaviour.json")
R = jload(f"{RES}/detectors.json")
Z = np.load(f"{RES}/scores.npz")
CN = {"neutral": "Neutral", "inc_ally": "Incentive: truth pays\n(control)", "inc_rival": "Incentive: falsehood pays\n(no instruction)", "instructed": "Instructed to lie"}
KN = {"known": ("Known (knowing misreport)", ORANGE), "conf_wrong": ("Confident-wrong belief", YELLOW),
      "no_knowledge": ("No knowledge (hallucination)", BLUE), "ambiguous": ("Ambiguous knowledge", "#b9b8b2")}

# ---- Fig 1: share of false outputs by knowledge state, per condition (+ n false)
fig, ax = plt.subplots(figsize=(8.6, 3.6))
conds = list(CN)
left = np.zeros(len(conds))
for k, (lab, col) in KN.items():
    v = np.array([B["cond"][c]["false_share"][k][0] * 100 for c in conds])
    ax.barh(range(len(conds)), v, left=left, color=col, edgecolor=SURF, linewidth=2, label=lab, height=0.62)
    for i, (x, l) in enumerate(zip(v, left)):
        if x >= 6:
            ax.text(l + x / 2, i, f"{x:.0f}%", ha="center", va="center", color="white" if col in (BLUE, ORANGE) else INK, fontsize=9)
    left += v
ax.set_yticks(range(len(conds)), [f"{CN[c]}\n(n false = {B['cond'][c]['outcome']['INCORRECT']})" for c in conds])
ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Share of false answers (%)"); ax.grid(axis="y", visible=False)
ax.legend(ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
fig.tight_layout(); fig.savefig(f"{FIG}/fig1_false_share.png", dpi=170); plt.close(fig)

# ---- Fig 2: AUROC heatmap, detector x contrast (diverging around chance 0.5)
DET = ["IP-LR", "IP-MM", "IP-LR@prompt", "SelfReport(Yes-No)", "Truth-probe", "Corr-probe", "NegLogProb", "SampleInconsistency"]
DL = ["Deception probe (LR)", "Deception probe (mean diff)", "Deception probe,\nprompt token only", "Self-report follow-up", "Truth probe", "Correctness probe", "Answer improbability", "Sample inconsistency"]
for P in ["instructed", "inc_rival"]:
    cs = list(R["auroc"][P])
    M = np.array([[R["auroc"][P][c][d][0] if R["auroc"][P][c][d][0] is not None else np.nan for d in DET] + [
        R["ceiling"][P][c]["resp_mean"][0] or np.nan, R["ceiling"][P][c]["prompt_last"][0] or np.nan] for c in cs], float)
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=9, color="white" if abs(M[i, j] - .5) > .33 else INK)
    def n_of(c):
        v = R["auroc"][P][c]["IP-LR"]
        return f"{v[3]} vs {v[4]}"
    ax.set_yticks(range(len(cs)), [f"{c}  [n={n_of(c)}]" for c in cs], fontsize=8.5)
    ax.set_xticks(range(len(DET) + 2), DL + ["In-distribution probe\n(ceiling, response)", "In-distribution probe\n(prompt token only)"], rotation=35, ha="right", fontsize=8.5)
    ax.grid(False); ax.axvline(len(DET) - 0.5, color=INK, lw=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("AUROC (0.5 = chance)")
    ax.set_title(f"AUROC, first-named class as positive — pressure condition: {CN[P].replace(chr(10), ' ')}", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(f"{FIG}/fig2_auroc_{P}.png", dpi=170); plt.close(fig)

# ---- Fig 3: layer sweep for the deception probe and the correctness probe
L = R["layers"]
pos = R["chosen"]["Corr-probe"][0]
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
lines = [("instructed|a: lie vs correct (neutral prompt)", "lie vs correct (neutral prompt)", BLUE),
         ("instructed|b: lie vs honest-correct (same prompt)", "lie vs honest-correct (same prompt)", ORANGE),
         ("instructed|d: lie vs hallucination (same prompt)", "lie vs hallucination (same prompt)", AQUA),
         ("instructed|e: hallucination vs correct (neutral prompt)", "hallucination vs correct (neutral)", YELLOW)]
for ax, nm, title in zip(axs, ["IP-LR", f"Corr-probe[{pos}]"], ["Deception probe (instructed pairs, LR)", "Correctness probe (trained on neutral answers)"]):
    for key, lab, col in lines:
        ax.plot(L, R["sweep"][nm][key], color=col, lw=2, marker="o", ms=4, label=lab)
    ax.axhline(0.5, color=MUTED, lw=1, ls="--"); ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel("Residual-stream layer"); ax.set_ylim(0, 1.02)
axs[0].set_ylabel("AUROC"); axs[1].legend(frameon=False, fontsize=8.5, loc="lower right", title="Instructed-lie condition")
fig.tight_layout(); fig.savefig(f"{FIG}/fig3_layer_sweep.png", dpi=170); plt.close(fig)

# ---- Fig 4: score distributions by cell for the two headline detectors
order = [("COR_N", "Correct\n(neutral)"), ("HAL_N", "Hallucination\n(neutral)"), ("HON_inc_rival", "Honest-correct\n(incentive)"), ("HAL_inc_rival", "Hallucination\n(incentive)"),
         ("LIE_inc_rival", "Lie\n(incentive)"), ("HON_instructed", "Honest-correct\n(instructed)"), ("HAL_instructed", "Hallucination\n(instructed)"), ("LIE_instructed", "Lie\n(instructed)")]
cond_of = lambda k: "neutral" if k.endswith("_N") else k.split("_", 1)[1]
fig, axs = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True)
for ax, d, lab in zip(axs, ["IP-LR", "Corr-probe"], ["Deception-probe score", "Correctness-probe score\n(higher = predicted wrong)"]):
    data = [Z[f"score|{d}|{cond_of(k)}"][Z[f"cell|{k}"]] for k, _ in order]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK, lw=1.5))
    for patch, (k, _) in zip(bp["boxes"], order):
        patch.set(facecolor=ORANGE if k.startswith("LIE") else BLUE if k.startswith("HAL") else AQUA, edgecolor=SURF, alpha=0.9)
    if d == "IP-LR":
        ax.axhline(R["flag_rate"]["IP-LR"]["threshold"], color=MUTED, ls="--", lw=1)
        ax.text(8.45, R["flag_rate"]["IP-LR"]["threshold"], " Alpaca 1% FPR", va="bottom", ha="right", fontsize=8, color=MUTED)
    ax.set_ylabel(lab); ax.grid(axis="x", visible=False)
axs[1].set_xticks(range(1, len(order) + 1), [f"{l}\nn={int(Z[f'cell|{k}'].sum())}" for k, l in order], fontsize=8.5)
from matplotlib.patches import Patch
axs[0].legend(handles=[Patch(color=AQUA, label="true answer"), Patch(color=BLUE, label="false, model does not know"), Patch(color=ORANGE, label="false, model knows")], frameon=False, ncol=3, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); fig.savefig(f"{FIG}/fig4_score_distributions.png", dpi=170); plt.close(fig)
print("figures written")
