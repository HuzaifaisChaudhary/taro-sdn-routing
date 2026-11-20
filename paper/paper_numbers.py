"""Computes every number quoted in the paper, and the bodies of its result tables, from the saved records.

Usage: python paper/paper_numbers.py            -> prints all numbers and table rows (table rows are LaTeX)
The assertions at the end check the prose statements made in the paper against the data.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, t as tdist

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "results"
TARGETED = ROOT / "artifacts" / "targeted_burst" / "results.json"
ORDER = ["taro-stable", "taro-heuristic", "ecmp", "ospf-style", "load-aware", "dqn-style", "ddpg-style", "graph-embedding"]
LBL = {"taro-stable": "TARO", "taro-heuristic": "TARO-NT", "ecmp": "ECMP", "ospf-style": "OSPF-style",
       "load-aware": "Load-aware", "dqn-style": "DQN-style", "ddpg-style": "DDPG-style", "graph-embedding": "BC-style"}
TOPOS = ["NSFNET", "GEANT", "AT&T"]; TN = {"NSFNET": "N14", "GEANT": "N22", "AT&T": "N25"}   # synthetic graphs of matching size
S = "taro-stable"


def ci(x):
    x = np.asarray(x, float); return float(tdist.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def wp(a, b):
    try:
        return float(wilcoxon(a, b, zero_method="zsplit").pvalue)
    except ValueError:
        return float("nan")


def pfmt(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def pm(p):
    return "<0.001" if p < 0.001 else f"={p:.3f}"


N, T = {}, {}   # scalar numbers (formatted strings) and table bodies

# ------------------------------------------------------------------ targeted experiment on G8 (Table 3, Fig. 2)
tg = pd.DataFrame(json.load(open(TARGETED))["runs"])
M5 = ORDER[:5]
pv = {m: tg.pivot(index="seed", columns="method", values=m) for m in ["delivery_fraction", "mean_delay", "changed_entries", "mean_route_change", "offered"]}
rows = []
for k in M5:
    d, dl, ch = pv["delivery_fraction"][k], pv["mean_delay"][k], pv["changed_entries"][k]
    if k == S:
        cmp_ = "---"
    else:
        diff = pv["delivery_fraction"][S] - d
        cmp_ = f"${diff.mean()*100:+.2f}$ ({pfmt(wp(pv['delivery_fraction'][S], d))})"
    rows.append(f"{LBL[k]} & {d.mean():.4f} $\\pm$ {ci(d):.4f} & {d.std(ddof=1):.4f} & {dl.mean():.3f} $\\pm$ {ci(dl):.3f} & "
                f"{ch.mean():.0f} & {pv['mean_route_change'][k].mean()*1000:.2f} & {cmp_} \\\\")
T["targeted"] = "\n".join(rows)
for k in M5:
    N[f"tg_del_{k}"] = f"{pv['delivery_fraction'][k].mean():.4f}"; N[f"tg_delay_{k}"] = f"{pv['mean_delay'][k].mean():.3f}"
    N[f"tg_chg_{k}"] = f"{pv['changed_entries'][k].mean():.0f}"
for k in M5[1:]:
    diff = pv["delivery_fraction"][S] - pv["delivery_fraction"][k]
    N[f"tg_dpp_{k}"] = f"{diff.mean()*100:.2f}"; N[f"tg_wins_{k}"] = str(int((diff > 0).sum()))
    N[f"tg_p_{k}"] = pfmt(wp(pv["delivery_fraction"][S], pv["delivery_fraction"][k]))
    N[f"tg_drel_{k}"] = f"{diff.mean()/pv['delivery_fraction'][k].mean()*100:.1f}"
    N[f"tg_delayrel_{k}"] = f"{(pv['mean_delay'][S].mean()/pv['mean_delay'][k].mean()-1)*100:.1f}"
N["tg_red"] = f"{(1-pv['changed_entries'][S].mean()/pv['changed_entries']['taro-heuristic'].mean())*100:.1f}"
N["tg_fried_p"] = pfmt(friedmanchisquare(*[pv["delivery_fraction"][k] for k in M5]).pvalue)
N["tg_offered"] = f"{pv['offered'][S].mean():.0f}"
N["tg_nt_cost"] = f"{(pv['delivery_fraction']['taro-heuristic']-pv['delivery_fraction'][S]).mean()*100:.2f}"

# ------------------------------------------------------------------ robustness over 30 graphs (Table 4)
rb = pd.DataFrame(json.load(open(EXP / "robust.json")))
rp = rb.groupby(["graph_seed", "method"]).delivery.mean().unstack()
rows = []
for k in ["ospf-style", "load-aware", "ecmp", "taro-heuristic"]:
    d = rp[S] - rp[k]; w, ti, l = int((d > 1e-6).sum()), int((d.abs() <= 1e-6).sum()), int((d < -1e-6).sum()); p = wp(rp[S], rp[k])
    rows.append(f"{LBL[k]} & {w} & {ti} & {l} & ${d.mean()*100:+.2f}$ & {pfmt(p)} \\\\")
    N[f"rb_w_{k}"], N[f"rb_t_{k}"], N[f"rb_l_{k}"], N[f"rb_m_{k}"] = str(w), str(ti), str(l), f"{d.mean()*100:.2f}"
    N[f"rb_p_{k}"] = pfmt(p) if p < 0.001 else f"{p:.2f}"
T["robust"] = "\n".join(rows)
N["rb_alltie"] = str(int(((rp.max(axis=1) - rp.min(axis=1)) < 1e-9).sum()))
N["rb_rank4"] = str(int((rp[S] - rp["ecmp"]).rank(ascending=False)[4]))
ru = rb.groupby(["graph_seed", "method"]).updates.mean().unstack().mean()
N["rb_upd_stable"], N["rb_upd_nt"] = f"{ru[S]:.0f}", f"{ru['taro-heuristic']:.0f}"

# ------------------------------------------------------------------ scenario matrix on N14, N22, N25 (Tables 5 and 7)
mx = pd.concat([pd.DataFrame(json.load(open(EXP / f))) for f in ["m_nsf.json", "m_geant.json", "m_att.json"]])


def mean(tp, k, m):
    return mx[(mx.topology == tp) & (mx.method == k)][m].mean()


rows = []
for k in ORDER:
    r = [LBL[k]]
    for tp in TOPOS:
        vals = {kk: mean(tp, kk, "delivery_total") for kk in ORDER}
        v = vals[k]; r.append(f"\\textbf{{{v:.3f}}}" if v == max(vals.values()) else f"{v:.3f}")
    for tp in TOPOS:
        r.append(f"{mean(tp, k, 'delay_total'):.2f}")
    for tp in TOPOS:
        r.append(f"{mean(tp, k, 'updates_total'):.0f}")
    rows.append(" & ".join(r) + " \\\\")
T["matrix"] = "\n".join(rows)


def retention(tp, k):
    sel = (mx.topology == tp) & (mx.method == k)
    return mx[sel & (mx.scenario == "failure-15")].delivery_total.mean() / mx[sel & (mx.scenario == "normal")].delivery_total.mean() * 100


for tp in TOPOS:
    s = pd.Series({k: mean(tp, k, "delivery_total") for k in ORDER}).sort_values(ascending=False)
    N[f"mx_rank_{TN[tp]}"] = str(list(s.index).index(S) + 1); N[f"mx_best_{TN[tp]}"] = LBL[s.index[0]]
    N[f"mx_red_{TN[tp]}"] = f"{(1-mean(tp,S,'updates_total')/mean(tp,'taro-heuristic','updates_total'))*100:.0f}"
    N[f"mx_ntgain_{TN[tp]}"] = f"{(mean(tp,'taro-heuristic','delivery_total')-mean(tp,S,'delivery_total'))*100:.1f}"
    N[f"mx_ret_{TN[tp]}_stable"] = f"{retention(tp, S):.1f}"; N[f"mx_ret_{TN[tp]}_ospf"] = f"{retention(tp, 'ospf-style'):.1f}"
rets = [retention(tp, k) for tp in TOPOS for k in ORDER]
N["mx_ret_min"], N["mx_ret_max"] = f"{min(rets):.0f}", f"{max(rets):.0f}"

blk = mx.set_index(["topology", "scenario", "seed", "method"])
P = {m: blk[m].unstack("method") for m in ["delivery_total", "delay_total", "updates_total", "rerouting_ratio"]}
N["mx_blocks"] = str(len(P["delivery_total"]))
ranks = np.vstack([rankdata(-row.to_numpy()) for _, row in P["delivery_total"][ORDER].iterrows()]); mr = ranks.mean(axis=0)
N["fried_pm"] = pm(friedmanchisquare(*[P["delivery_total"][k] for k in ORDER]).pvalue)
kk, nb = len(ORDER), len(ranks); N["nemenyi_cd"] = f"{3.031*np.sqrt(kk*(kk+1)/(6*nb)):.2f}"   # q_0.05 for 8 methods = 3.031
for k, r in zip(ORDER, mr):
    N[f"mrank_{k}"] = f"{r:.2f}"
T["ranks_text"] = ", ".join(f"{r:.2f}" for r in mr)

d, dl, u, rr = P["delivery_total"], P["delay_total"], P["updates_total"], P["rerouting_ratio"]


def cmp_row(k, tag):
    dd = d[S] - d[k]; rel = dd.mean() / d[k].mean() * 100
    dld = (dl[S].mean() / dl[k].mean() - 1) * 100; ur = (u[S].mean() / u[k].mean() - 1) * 100
    N[f"bp_dpp_{k}"], N[f"bp_drel_{k}"], N[f"bp_p_{k}"] = f"{dd.mean()*100:+.2f}", f"{rel:+.1f}", pfmt(wp(d[S], d[k]))
    N[f"bp_delay_{k}"], N[f"bp_upd_{k}"], N[f"bp_wins_{k}"] = f"{dld:+.1f}", f"{ur:+.0f}", str(int((dd > 0).sum()))
    N[f"bp_pm_{k}"] = pm(wp(d[S], d[k]))
    return (f"{tag} & {d[k].mean():.3f} & {dl[k].mean():.2f} & {u[k].mean():.0f} & {rr[k].mean()*100:.2f} & "
            f"${dd.mean()*100:+.2f}$ (${rel:+.1f}$\\%) & {pfmt(wp(d[S], d[k]))} & ${dld:+.1f}$\\% & ${ur:+.0f}$\\% \\\\")


rows = [f"\\textbf{{TARO (ours)}} & {d[S].mean():.3f} & {dl[S].mean():.2f} & {u[S].mean():.0f} & {rr[S].mean()*100:.2f} & --- & --- & --- & --- \\\\\n\\midrule"]
for k, tag in [("ddpg-style", "DDPG-style$^{\\dagger}$ (P1 \\cite{qi2026clpe})"),
               ("graph-embedding", "BC-style$^{\\dagger}$ (P2 \\cite{wu2026irodbc}, Grace \\cite{wei2025grace})"),
               ("dqn-style", "DQN-style$^{\\dagger}$ (P3 \\cite{jaf2026hybrid})"),
               ("ospf-style", "OSPF-style (baseline in P1--P3)"), ("ecmp", "ECMP (baseline in Grace \\cite{wei2025grace})")]:
    rows.append(cmp_row(k, tag))
T["basecmp"] = "\n".join(rows)
for k in ORDER:
    N[f"pool_del_{k}"] = f"{d[k].mean():.3f}"; N[f"pool_upd_{k}"] = f"{u[k].mean():.0f}"; N[f"pool_delay_{k}"] = f"{dl[k].mean():.2f}"

# ------------------------------------------------------------------ light load (quoted in Sect. 5.4)
lt = pd.DataFrame(json.load(open(EXP / "m_light.json"))); l0 = lt[lt.scenario == "light"]
for k in ORDER:
    N[f"lt_delay_{k}"] = f"{l0[l0.method==k].delay_total.mean():.2f}"
N["lt_del"] = f"{l0.delivery_total.mean():.3f}"

# ------------------------------------------------------------------ ablation (Table 8, Fig. 6)
ab = pd.DataFrame(json.load(open(EXP / "abl.json"))); FULL = "full (Stable TARO)"; G8 = "burst (selected graph)"


def A(v, s, m):
    return ab[(ab.variant == v) & (ab.scenario == s)][m].mean()


AB = [(FULL, "Full \\taro{}"), ("no stability threshold", "No threshold ($\\tau{=}0$)"), ("threshold 0.05", "$\\tau{=}0.05$"),
      ("threshold 0.20", "$\\tau{=}0.20$"), ("threshold 0.30", "$\\tau{=}0.30$"), ("cap 0.05", "$\\delta{=}0.05$"), ("cap 0.20", "$\\delta{=}0.20$"),
      ("no change cap", "No change cap"), ("no capacity/delay term", "No capacity/delay"), ("no utilisation term", "No utilisation"),
      ("no queue term", "No queue"), ("single path (argmax)", "Single path")]
rows = []
for v, name in AB:
    cells = [name] + [f"{A(v,s,'delivery'):.4f}" for s in ["normal", "heavy", "failure-10", G8]]
    cells += [f"{A(v,'normal','delay'):.2f}", f"{A(v,G8,'delay'):.2f}", f"{A(v,'normal','updates'):.0f}", f"{A(v,'heavy','updates'):.0f}",
              f"{A(v,G8,'updates'):.0f}", f"{A(v,'normal','flaps'):.0f}", f"{A(v,G8,'flaps'):.0f}"]
    rows.append(" & ".join(cells) + " \\\\" + ("\n\\midrule" if v == FULL else ""))
T["ablation"] = "\n".join(rows)
for v, key in [(FULL, "full"), ("no stability threshold", "nt"), ("threshold 0.30", "t30"), ("threshold 0.20", "t20"), ("cap 0.20", "c20"),
               ("cap 0.05", "c05"), ("no change cap", "nocap"), ("no capacity/delay term", "nocd"), ("no utilisation term", "noutil"),
               ("no queue term", "noq"), ("single path (argmax)", "single")]:
    for s, sk in [("normal", "n"), ("heavy", "h"), (G8, "g")]:
        N[f"ab_{key}_del_{sk}"] = f"{A(v,s,'delivery'):.4f}"; N[f"ab_{key}_upd_{sk}"] = f"{A(v,s,'updates'):.0f}"
        N[f"ab_{key}_flap_{sk}"] = f"{A(v,s,'flaps'):.0f}"; N[f"ab_{key}_delay_{sk}"] = f"{A(v,s,'delay'):.2f}"
        N[f"ab_{key}_dpp_{sk}"] = f"{(A(v,s,'delivery')-A(FULL,s,'delivery'))*100:+.2f}"
N["c20_share_lo"] = f"{min(A('cap 0.20',s,'updates')/A('no stability threshold',s,'updates') for s in ['normal','heavy',G8])*100:.0f}"
N["c20_share_hi"] = f"{max(A('cap 0.20',s,'updates')/A('no stability threshold',s,'updates') for s in ['normal','heavy',G8])*100:.0f}"

# ------------------------------------------------------------------ learner prototype and qualitative example
ppo = json.load(open(EXP / "ppo_check.json")); ref = json.load(open(EXP / "ppo_ref.json")); o = ppo["original lr=0.0003"]
N["ppo_first"], N["ppo_last"], N["ppo_eval"] = f"{np.mean(o['hist'][:5]):.2f}", f"{np.mean(o['hist'][-5:]):.2f}", f"{o['eval']:.2f}"
N["ppo_ecmp"], N["ppo_stable"] = f"{ref['ecmp']:.2f}", f"{ref['taro-stable']:.2f}"
q = json.load(open(EXP / "qual.json")); fz = json.load(open(EXP / "freeze.json"))
w = q["trace"]["weights"]
N["q_edge"] = f"{q['trace']['edge'][0]}->{q['trace']['edge'][1]}"; N["q_w0"] = f"{w['taro-stable'][0]:.2f}"
N["q_nt_min"], N["q_nt_max"] = f"{min(w['taro-heuristic']):.2f}", f"{max(w['taro-heuristic']):.2f}"
for k, key in [("taro-stable", "q_upd_stable"), ("taro-heuristic", "q_upd_nt"), ("taro-cap02", "q_upd_c20")]:
    N[key] = str(q["methods"][k]["total_updates"])
N["q_multi"], N["q_entries"] = str(q["multi_entries"]), str(q["entries"])
N["fz_first8"], N["fz_multi14"], N["fz_first14"] = f"{fz['targeted_first']:.0f}", str(fz["multi14"]), f"{fz['normal']['first']:.0f}"
N["fz_perint_n"], N["fz_perint_h"] = f"{fz['normal']['after']/19:.1f}", f"{fz['heavy']['after']/19:.1f}"

# ------------------------------------------------------------------ checks of statements written as prose in the paper
assert N["bp_wins_ecmp"] == "90" and N["bp_wins_ospf-style"] == "90" and N["mx_blocks"] == "90"
assert all(N[f"tg_wins_{k}"] == "10" for k in ["ecmp", "ospf-style", "load-aware"])
cd = float(N["nemenyi_cd"]); r = {k: float(N[f"mrank_{k}"]) for k in ORDER}
assert r["ecmp"] - r[S] > cd and r["ospf-style"] - r[S] > cd and r[S] - r["taro-heuristic"] > cd and r[S] - r["dqn-style"] > cd
assert all(abs(r[S] - r[k]) < cd for k in ["load-aware", "ddpg-style", "graph-embedding"])
assert sorted(r, key=r.get).index(S) == 3
pu = {k: float(N[f"pool_upd_{k}"]) for k in ORDER}
assert pu[S] < min(pu[k] for k in ["load-aware", "dqn-style", "graph-embedding"]) and pu[S] > max(pu[k] for k in ["ecmp", "ospf-style", "ddpg-style"])
assert [N[f"mx_rank_{t}"] for t in ["N14", "N22", "N25"]] == ["5", "5", "4"]
assert [N[f"mx_best_{t}"] for t in ["N14", "N22", "N25"]] == ["Load-aware", "DQN-style", "DQN-style"]
reds = [float(N[f"mx_red_{t}"]) for t in ["N14", "N22", "N25"]] + [float(N["tg_red"]), (1 - float(N["rb_upd_stable"]) / float(N["rb_upd_nt"])) * 100]
assert 80 <= min(reds) and max(reds) <= 95.5
assert all(float(N[f"mx_ret_{t}_stable"]) < float(N[f"mx_ret_{t}_ospf"]) for t in ["N14", "N22", "N25"])
assert N["rb_rank4"] == "1" and N["fz_first8"] == N["q_multi"] == "12"
assert abs(A(FULL, G8, "delivery") - pv["delivery_fraction"][S].mean()) < 1e-12
assert abs(A("no capacity/delay term", G8, "delivery") - pv["delivery_fraction"]["ecmp"].mean()) < 1e-9

if __name__ == "__main__":
    for k in sorted(N):
        print(f"{k:28s} {N[k]}")
    for k, v in T.items():
        print("==", k); print(v)
    print("\nAll prose checks passed.")
