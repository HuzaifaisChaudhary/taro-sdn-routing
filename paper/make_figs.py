"""Generates the paper's result figures (PDF and PNG) from the saved experiment records.

Usage (from the repository root or this folder):  python paper/make_figs.py [figure names...]
Figures: targeted (Fig. 2), qualitative (Fig. 3), matrix (Fig. 4), tradeoff (Fig. 5), ablation (Fig. 6),
         robust and ppo (supporting figures, not in the paper).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as tdist

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results"
TARGETED = ROOT / "artifacts" / "targeted_burst" / "results.json"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

W2, W1 = 6.30, 3.00            # double- and single-column widths (inches) of the sn-jnl two-column layout
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.linewidth": .6, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.labelcolor": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .5, "axes.axisbelow": True,
    "xtick.major.width": .5, "ytick.major.width": .5, "xtick.major.size": 2, "ytick.major.size": 2,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": .02, "legend.frameon": False})
ORDER = ["taro-stable", "taro-heuristic", "ecmp", "ospf-style", "load-aware", "dqn-style", "ddpg-style", "graph-embedding"]
LBL = {"taro-stable": "TARO", "taro-heuristic": "TARO-NT", "ecmp": "ECMP", "ospf-style": "OSPF-style",
       "load-aware": "Load-aware", "dqn-style": "DQN-style", "ddpg-style": "DDPG-style", "graph-embedding": "BC-style"}
# colour-blind-checked categorical palette in fixed order: a method keeps its colour in every figure
COL = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
MRK = dict(zip(ORDER, ["o", "s", "^", "v", "D", "P", "X", "*"]))


def ci(x):
    x = np.asarray(x, float)
    return 0.0 if len(x) < 2 else float(tdist.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); fig.savefig(OUT / f"{name}.png", dpi=220); plt.close(fig)


def targeted():
    df = pd.DataFrame(json.load(open(TARGETED))["runs"]); ms = ORDER[:5]
    tick = {"taro-stable": "TARO", "taro-heuristic": "TARO-\nNT", "ecmp": "ECMP", "ospf-style": "OSPF-\nstyle", "load-aware": "Load-\naware"}
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.7))
    for ax, (m, ttl, yl, kind) in zip(axs, [("delivery_fraction", "(a) Delivery fraction", "delivered / offered", "dot"),
                                            ("mean_delay", "(b) Mean delay", "model time units", "dot"),
                                            ("changed_entries", "(c) Changed routing entries", "entries per run", "bar")]):
        for i, k in enumerate(ms):
            v = df[df.method == k][m].to_numpy()
            if kind == "bar":
                ax.bar(i, v.mean(), width=.62, color=COL[k], edgecolor=INK if k == "taro-stable" else "none", linewidth=.7)
                ax.text(i, v.mean() + 6, f"{v.mean():.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
            else:
                ax.scatter(np.full(len(v), i) + np.linspace(-.16, .16, len(v)), v, s=5, color=COL[k], alpha=.45, linewidths=0)
                ax.errorbar(i, v.mean(), yerr=ci(v), fmt=MRK[k], ms=4.5, color=COL[k], mec=INK, mew=.6, ecolor=INK, elinewidth=.8, capsize=2.5)
        ax.set_xticks(range(5)); ax.set_xticklabels([tick[k] for k in ms])
        ax.set_title(ttl, loc="left"); ax.set_ylabel(yl); ax.grid(axis="x", visible=False)
        if kind == "bar":
            ax.set_ylim(0, 320)
    fig.tight_layout(w_pad=1.2); save(fig, "fig_targeted")


def qualitative():
    q = json.load(open(EXP / "qual.json")); tr = q["trace"]; T = len(tr["weights"]["taro-stable"]); x = np.arange(T)
    ser = [("taro-heuristic", "TARO-NT ($\\delta$=0.1, $\\tau$=0)", COL["taro-heuristic"], "-"),
           ("taro-cap02", "$\\delta$=0.2, $\\tau$=0.1", "#1baf7a", "--"),
           ("taro-stable", "TARO ($\\delta$=$\\tau$=0.1)", COL["taro-stable"], "-")]
    fig, (a, b) = plt.subplots(2, 1, figsize=(W1, 2.75), sharex=True, gridspec_kw={"height_ratios": [1.15, 1]})
    for k, lab, c, ls in ser:
        a.step(x, tr["weights"][k], where="post", color=c, lw=1.3, ls=ls, label=lab)
        b.step(x, q["methods"][k]["updates"], where="post", color=c, lw=1.3, ls=ls)
    a.axvspan(24, T - 1, color="#efeee9", lw=0); b.axvspan(24, T - 1, color="#efeee9", lw=0)
    a.text(28.5, .9, "drain", ha="center", fontsize=6, color=MUTED)
    a.set_ylabel(f"split weight on\nlink {tr['edge'][0]}$\\to${tr['edge'][1]}"); a.set_ylim(0, 1)
    a.set_title(f"(a) One routing entry (node {tr['node']}, destination {tr['dest']})", loc="left")
    h_, l_ = a.get_legend_handles_labels()
    b.set_ylabel("changed entries\nper interval"); b.set_xlabel("control interval (s)")
    b.set_title("(b) Routing changes across all 56 entries", loc="left"); b.set_xlim(0, T - 1)
    fig.tight_layout(h_pad=.6, rect=(0, 0, 1, .85))
    fig.legend(h_, l_, loc="upper center", ncol=1, handlelength=2.4, bbox_to_anchor=(.55, 1.0), labelspacing=.25)
    save(fig, "fig_qualitative")


def robust():
    r = pd.DataFrame(json.load(open(EXP / "robust.json")))
    pv = r.groupby(["graph_seed", "method"]).delivery.mean().unstack()
    fig, ax = plt.subplots(figsize=(W2, 1.9)); x = np.arange(len(pv))
    for k, off in [("ecmp", -.27), ("ospf-style", 0), ("load-aware", .27)]:
        ax.bar(x + off, (pv["taro-stable"] - pv[k]) * 100, width=.25, color=COL[k], label=f"vs {LBL[k]}")
    ax.axhline(0, color=INK, lw=.6); ax.set_xticks(x); ax.set_xticklabels(pv.index, fontsize=6)
    ax.set_xlabel("graph seed"); ax.set_ylabel("delivery difference (pp)"); ax.grid(axis="x", visible=False)
    ax.set_xlim(-.7, len(pv) - .3); ax.set_ylim(-1, 14.5)
    d4 = (pv.loc[4, "taro-stable"] - pv.loc[4, "ecmp"]) * 100
    ax.annotate("graph used in the\ntargeted experiment", xy=(4 - .27, d4), xytext=(5.6, 10.3), fontsize=6.2, color=INK,
                arrowprops=dict(arrowstyle="-", lw=.5, color=MUTED), va="center")
    ax.legend(ncol=3, loc="upper right", title="TARO minus baseline (positive favours TARO)", title_fontsize=6.5,
              columnspacing=1.2, handlelength=1.2)
    fig.tight_layout(); save(fig, "fig_robust")


def load_matrix():
    return pd.concat([pd.DataFrame(json.load(open(EXP / f))) for f in ["m_nsf.json", "m_geant.json", "m_att.json"]])


TNAME = {"NSFNET": "N14 (14 nodes, 27 links)", "GEANT": "N22 (22 nodes, 56 links)", "AT&T": "N25 (25 nodes, 57 links)"}


def matrix():
    df = load_matrix(); sc = ["normal", "heavy", "burst", "failure-05", "failure-10", "failure-15"]
    sl = ["normal", "heavy", "burst", "fail 5%", "fail 10%", "fail 15%"]
    fig, axs = plt.subplots(1, 3, figsize=(W2, 1.95), sharey=True)
    for ax, tp in zip(axs, ["NSFNET", "GEANT", "AT&T"]):
        d = df[df.topology == tp]
        for i, k in enumerate(ORDER):
            m = [d[(d.scenario == s) & (d.method == k)].delivery_total.mean() for s in sc]
            e = [ci(d[(d.scenario == s) & (d.method == k)].delivery_total) for s in sc]
            star = k == "taro-stable"
            ax.errorbar(np.arange(6) + (i - 3.5) * .095, m, yerr=e, fmt=MRK[k], ms=5 if star else 3.6, color=COL[k],
                        mec=INK if star else COL[k], mew=.7 if star else .3, ecolor=COL[k], elinewidth=.6, capsize=0,
                        label=LBL[k], zorder=4 if star else 3)
        ax.set_xticks(range(6)); ax.set_xticklabels(sl, rotation=35, ha="right"); ax.set_title(TNAME[tp], loc="left")
        ax.grid(axis="x", visible=False)
        for xx in np.arange(.5, 5.5, 1):
            ax.axvline(xx, color=GRID, lw=.5)
    axs[0].set_ylabel("delivery fraction")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=8, loc="upper center", bbox_to_anchor=(.5, 1.06), columnspacing=1.1, handletextpad=.2)
    fig.tight_layout(w_pad=.8); save(fig, "fig_matrix")


def tradeoff():
    df = load_matrix(); fig, ax = plt.subplots(figsize=(W1, 2.2))
    g = df.groupby("method").mean(numeric_only=True)
    off = {"taro-stable": (-5, 6), "taro-heuristic": (-2, -11), "ecmp": (6, -2), "ospf-style": (6, -3), "load-aware": (2, -11),
           "dqn-style": (-6, 4), "ddpg-style": (-6, -3), "graph-embedding": (6, -2)}
    for k in ORDER:
        star = k == "taro-stable"
        ax.scatter(g.loc[k, "updates_total"], g.loc[k, "delivery_total"], s=46 if star else 28, marker=MRK[k], color=COL[k],
                   edgecolor=INK if star else "white", linewidth=.8 if star else .6, zorder=4 if star else 3)
        ha = "left" if off[k][0] > 0 else "right"
        ax.annotate(LBL[k], (g.loc[k, "updates_total"], g.loc[k, "delivery_total"]), xytext=off[k], textcoords="offset points",
                    fontsize=6.3, ha=ha, color=INK)
    ax.set_xscale("log"); ax.set_xlabel("changed routing entries per run (log scale)"); ax.set_ylabel("delivery fraction")
    ax.annotate("better", xy=(.05, .95), xytext=(.2, .84), xycoords="axes fraction", fontsize=6.2, color=MUTED,
                arrowprops=dict(arrowstyle="->", lw=.6, color=MUTED))
    ax.set_xlim(55, 5500); ax.set_ylim(.59, .72)
    fig.tight_layout(); save(fig, "fig_tradeoff")


AB = [("no stability threshold", "No threshold ($\\tau$=0)"), ("threshold 0.05", "$\\tau$=0.05"), ("threshold 0.20", "$\\tau$=0.20"),
      ("threshold 0.30", "$\\tau$=0.30"), ("cap 0.05", "$\\delta$=0.05"), ("cap 0.20", "$\\delta$=0.20"), ("no change cap", "No change cap"),
      ("no capacity/delay term", "No capacity/delay term"), ("no utilisation term", "No utilisation term"), ("no queue term", "No queue term"),
      ("single path (argmax)", "Single path (no split)")]


def ablation():
    a = pd.DataFrame(json.load(open(EXP / "abl.json"))); FULL = "full (Stable TARO)"
    scen = [("normal", "N14 normal", "#2a78d6", "o"), ("heavy", "N14 heavy", "#eb6834", "s"), ("burst (selected graph)", "G8 burst", "#1baf7a", "^")]
    fig, (l, r) = plt.subplots(1, 2, figsize=(W2, 2.2), sharey=True); y = np.arange(len(AB))
    for j, (s, lab, c, mk) in enumerate(scen):
        base = a[(a.variant == FULL) & (a.scenario == s)].set_index("seed")
        dd, uu = [], []
        for v, _ in AB:
            d = a[(a.variant == v) & (a.scenario == s)].set_index("seed")
            dd.append(((d.delivery - base.delivery) * 100).mean()); uu.append(d.updates.mean() / base.updates.mean())
        l.scatter(dd, y + (j - 1) * .2, s=16, color=c, marker=mk, label=lab, zorder=3, edgecolor="white", linewidth=.4)
        r.scatter(uu, y + (j - 1) * .2, s=16, color=c, marker=mk, zorder=3, edgecolor="white", linewidth=.4)
    l.axvline(0, color=INK, lw=.6); r.axvline(1, color=INK, lw=.6); r.set_xscale("log")
    l.set_yticks(y); l.set_yticklabels([n for _, n in AB]); l.invert_yaxis()
    l.set_xlabel("change in delivery vs. full TARO (pp)"); r.set_xlabel("changed entries relative to full TARO (log scale)")
    l.set_title("(a) Routing quality", loc="left"); r.set_title("(b) Routing churn", loc="left")
    for ax in (l, r):
        ax.grid(axis="y", visible=False)
        for yy in np.arange(.5, len(AB) - .5, 1):
            ax.axhline(yy, color=GRID, lw=.4)
    h_, l_ = l.get_legend_handles_labels()
    fig.tight_layout(w_pad=1, rect=(0, 0, 1, .94))
    fig.legend(h_, l_, loc="upper center", ncol=3, handletextpad=.2, bbox_to_anchor=(.55, 1.0))
    save(fig, "fig_ablation")


def ppo():
    r = json.load(open(EXP / "ppo_check.json"))
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    sty = {"original lr=0.0003": ("#2a78d6", "-", "as implemented, lr 3e-4"), "original lr=0.03": ("#2a78d6", "--", "as implemented, lr 3e-2"),
           "sign-corrected lr=0.0003": ("#eb6834", "-", "sign-corrected, lr 3e-4"), "sign-corrected lr=0.03": ("#eb6834", "--", "sign-corrected, lr 3e-2")}
    for k, v in r.items():
        ax.plot(pd.Series(v["hist"]).rolling(4, min_periods=1).mean(), color=sty[k][0], ls=sty[k][1], lw=1.1, label=sty[k][2])
    ax.set_xlabel("training episode"); ax.set_ylabel("episode reward\n(4-episode mean)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(.45, 1.42), columnspacing=1)
    fig.tight_layout(); save(fig, "fig_ppo")


if __name__ == "__main__":
    for f in (sys.argv[1:] or ["targeted", "qualitative", "robust", "matrix", "tradeoff", "ablation", "ppo"]):
        globals()[f]()
        print("wrote", f)
