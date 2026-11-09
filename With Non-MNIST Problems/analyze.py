import json, os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/claude/bench/out"
os.makedirs(OUT, exist_ok=True)
rows = [json.loads(l) for l in open("/home/claude/bench/results.jsonl")]

TASKS = ["wdbc", "wine", "digits_0v8", "digits_3v5", "digits_4v9", "digits_all"]
LABEL = {"wdbc": "WDBC\n(2-class)", "wine": "Wine\n(3-class)", "digits_0v8": "Digits\n0 vs 8",
         "digits_3v5": "Digits\n3 vs 5", "digits_4v9": "Digits\n4 vs 9", "digits_all": "Digits\n10-class"}
METHODS = ["IaPCE_sat16", "fixed_sat16", "IaPCE_mod4", "fixed_mod4", "AE_4q", "AE_8q",
           "classical_LR_12f"]
NAME = {"IaPCE_sat16": "IαPCE (α→16)", "fixed_sat16": "PCE fixed α=16",
        "IaPCE_mod4": "IαPCE (α→4)", "fixed_mod4": "PCE fixed α=4",
        "AE_4q": "Angle enc., 4 qubits (4 feat.)", "AE_8q": "Angle enc., 8 qubits (8 feat.)",
        "classical_LR_12f": "Classical LR (12 feat.)"}


def col(task, m, key="acc"):
    rs = sorted([r for r in rows if r["task"] == task], key=lambda r: r["seed"])
    return np.array([r[m][key] for r in rs])


# ---------- summary table
summ = []
for t in TASKS:
    n = len([r for r in rows if r["task"] == t])
    q = [r for r in rows if r["task"] == t][0]["n_qubits_pce"]
    for m in METHODS:
        a = col(t, m)
        summ.append({"task": t, "method": m, "mean": a.mean(), "std": a.std(ddof=1) if len(a) > 1 else 0,
                     "n_seeds": n, "pce_qubits": q})
S = pd.DataFrame(summ)
S.to_csv(f"{OUT}/summary.csv", index=False)

# ---------- paired comparisons
def paired(t, a, b):
    d = col(t, a) - col(t, b)
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    ci = stats.t.ppf(0.975, n - 1) * se
    p_t = stats.ttest_rel(col(t, a), col(t, b)).pvalue if sd > 0 else np.nan
    try:
        p_w = stats.wilcoxon(d, zero_method="zsplit").pvalue
    except Exception:
        p_w = np.nan
    return {"task": t, "A": a, "B": b, "mean_diff": mean, "ci95": ci, "p_t": p_t, "p_wilcoxon": p_w,
            "wins": int((d > 0).sum()), "ties": int((d == 0).sum()), "losses": int((d < 0).sum())}

pairs = [("IaPCE_sat16", "fixed_sat16"), ("IaPCE_mod4", "fixed_mod4"),
         ("IaPCE_sat16", "AE_4q"), ("IaPCE_sat16", "AE_8q"),
         ("IaPCE_mod4", "AE_4q"), ("IaPCE_mod4", "AE_8q"),
         ("IaPCE_sat16", "classical_LR_12f")]
P = pd.DataFrame([paired(t, a, b) for t in TASKS for (a, b) in pairs])
# Holm correction within the primary family (IaPCE_sat16 vs fixed_sat16 across tasks)
fam = P[(P.A == "IaPCE_sat16") & (P.B == "fixed_sat16")].copy()
order = fam.p_wilcoxon.argsort().values
m = len(fam)
holm = np.empty(m)
run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m - rank) * fam.p_wilcoxon.values[i]))
    holm[i] = run
P.loc[fam.index, "p_wilcoxon_holm"] = holm
P.to_csv(f"{OUT}/paired.csv", index=False)

pd.set_option("display.width", 200)
print("=== accuracy mean ± std (10 seeds) ===")
piv = S.pivot(index="task", columns="method", values="mean").loc[TASKS, METHODS]
pivs = S.pivot(index="task", columns="method", values="std").loc[TASKS, METHODS]
print((piv.round(3).astype(str) + " ± " + pivs.round(3).astype(str)).to_string())
print("\nPCE qubits:", S.groupby("task").pce_qubits.first().to_dict())
print("\n=== paired differences (A − B) ===")
print(P.round(4).to_string(index=False))

# ---------- figure (dot + whisker, fixed categorical order)
C = {"IaPCE_sat16": "#2a78d6", "fixed_sat16": "#eb6834", "AE_4q": "#1baf7a", "AE_8q": "#eda100"}
HATCH = {"IaPCE_sat16": None, "fixed_sat16": "///", "AE_4q": "...", "AE_8q": "xxx"}
MARK = {"IaPCE_sat16": "o", "fixed_sat16": "s", "AE_4q": "^", "AE_8q": "D"}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, ax = plt.subplots(figsize=(7.1, 2.9), dpi=220)
ms = ["IaPCE_sat16", "fixed_sat16", "AE_4q", "AE_8q"]
off = np.linspace(-0.27, 0.27, len(ms))
x = np.arange(len(TASKS))
for j, mth in enumerate(ms):
    mu = [S[(S.task == t) & (S.method == mth)]["mean"].item() for t in TASKS]
    sd = [S[(S.task == t) & (S.method == mth)]["std"].item() for t in TASKS]
    ax.errorbar(x + off[j], mu, yerr=sd, fmt=MARK[mth], color=C[mth], ms=5.5, mec="#fcfcfb", mew=0.8,
                elinewidth=1.4, capsize=2.5, label=NAME[mth], zorder=3)
cl = [S[(S.task == t) & (S.method == "classical_LR_12f")]["mean"].item() for t in TASKS]
ax.scatter(x, cl, marker="_", s=420, color="#0b0b0b", linewidths=1.4, zorder=4,
           label=NAME["classical_LR_12f"] + " reference")
ax.set_xticks(x)
ax.set_xticklabels([LABEL[t] for t in TASKS])
ax.set_ylabel("Test accuracy (mean ± s.d., 10 seeds)")
ax.set_ylim(0.2, 1.02)
ax.grid(axis="y", color="#e5e4df", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.legend(ncol=3, frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0.0, 1.0), handletextpad=0.3,
          columnspacing=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/benchmark_accuracy.png", bbox_inches="tight")
fig.savefig(f"{OUT}/benchmark_accuracy.pdf", bbox_inches="tight")
print("\nfigure saved")
