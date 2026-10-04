"""Extra checks: incentive-only lie yield against the matched control, sample-level corroboration
of lies, trivial surface baselines (response length, trailing period), and the pilot yields."""
import numpy as np
from scipy.stats import binomtest, rankdata

from common import RES, jdump, jload

D = jload(f"{RES}/main.json")
Z = np.load(f"{RES}/scores.npz")
kn = Z["know"] == "known"
rng = np.random.default_rng(0)
out = {}
g = {c: np.array(D["cond"][c]["grade"]) for c in D["cond"]}

# paired comparison: falsehood-pays prompt vs truth-pays prompt, known questions only
fr, fa = (g["inc_rival"] == "INCORRECT")[kn], (g["inc_ally"] == "INCORRECT")[kn]
b, c = int((fr & ~fa).sum()), int((fa & ~fr).sum())
diff = [(lambda i: fr[i].mean() - fa[i].mean())(rng.integers(0, kn.sum(), kn.sum())) for _ in range(5000)]
out["incentive_yield"] = {"n_known": int(kn.sum()), "false_rival": int(fr.sum()), "false_ally": int(fa.sum()), "both": int((fr & fa).sum()),
                          "rival_only": b, "ally_only": c, "mcnemar_p": float(binomtest(b, b + c, 0.5).pvalue),
                          "excess_rate": float(fr.mean() - fa.mean()), "excess_ci": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))]}

# do same-prompt samples corroborate the greedy lie? (fraction of 10 samples that are correct)
out["samples_correct_frac"] = {}
for cnd in ["inc_ally", "inc_rival", "instructed"]:
    lie = kn & (g[cnd] == "INCORRECT"); hon = kn & (g[cnd] == "CORRECT")
    sa = np.array([np.mean(x) for x in D["cond"][cnd]["samples_alias"]])
    out["samples_correct_frac"][cnd] = {"lie": float(sa[lie].mean()), "honest": float(sa[hon].mean())}


def auc(p, n):
    r = rankdata(np.r_[p, n])
    return float((r[: len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


# surface baselines for the instructed condition
resp = D["cond"]["instructed"]["resp"]
ln = np.array([len(r) for r in resp]); dot = np.array([r.endswith(".") for r in resp], float)
cell = lambda k: Z[f"cell|{k}"]
out["surface_baselines_instructed"] = {
    "length: lie vs honest (same prompt)": auc(ln[cell("LIE_instructed")], ln[cell("HON_instructed")]),
    "length: lie vs hallucination (same prompt)": auc(ln[cell("LIE_instructed")], ln[cell("HAL_instructed")]),
    "trailing period: lie vs honest (same prompt)": auc(dot[cell("LIE_instructed")], dot[cell("HON_instructed")]),
    "trailing period rate": {k: float(dot[cell(k)].mean()) for k in ["LIE_instructed", "HON_instructed", "HAL_instructed", "FWKc_instructed"]},
}
out["pilot_yield_on_neutral_correct"] = jload(f"{RES}/pilot.json")["summary"]
jdump(out, f"{RES}/extra_stats.json")
import json; print(json.dumps(out, indent=1))
