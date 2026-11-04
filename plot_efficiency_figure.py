# -*- coding: utf-8 -*-
"""
Regenerates the efficiency figure (emissions in grams of CO2 + number of parameters, log scale) for ETeR-X,
USEM, BERT, EmbeddingGemma and Qwen3-Embedding, using:
  * final-training emissions (last row of emissions_train.csv, kg -> g) of the checkpoints behind the paper tables:
    ETeR-X and Amazon USEM/BERT as listed in results/user_agg_summary_known.csv, TripAdvisor USEM/BERT as
    retrained on the ETeR-X split (results/tripadvisor_retrain_md5.csv);
  * encoding emissions of the zero-shot embedding baselines (results/embedding_results.csv);
  * parameter counts from output/param_results.xlsx (ETeR-X/USEM/BERT) and the fixed sizes of the embedding models.
Usage: python plot_efficiency_figure.py [out.pdf]
"""
import csv
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "output/bar_plot_co2_params_resplit.pdf"
rdm = lambda f: list(csv.DictReader(open(f)))
layout = [("AM", "amazon", [("FSH", "fashion"), ("MSC", "digital_music")]),
          ("POI", "pois", [("BCN", "barcelona"), ("LND", "london"), ("MDR", "madrid"), ("NYC", "newyorkcity"), ("PRS", "paris")]),
          ("RST", "restaurants", [("BCN", "barcelona"), ("GJN", "gijon"), ("MDR", "madrid"), ("NYC", "newyorkcity"), ("PRS", "paris")])]
sm = {(r["model"], r["dataset"], r["subset"]): r["md5"] for r in rdm("results/user_agg_summary_known.csv")}
nm = {(r["model"], r["dataset"], r["subset"]): r["new_md5"] for r in rdm("results/tripadvisor_retrain_md5.csv")}
emb = {(r["model"], r["dataset"], r["subset"]): float(r["emissions_g"]) for r in rdm("results/embedding_results.csv")}
co2 = lambda m, d, s: 1000 * float(rdm(f"models/{m}/{d}/{s}/{nm.get((m, d, s)) or sm[(m, d, s)]}/emissions_train.csv")[-1]["emissions"])

pr = pd.read_excel("output/param_results.xlsx", header=None).iloc[3:15, :].reset_index(drop=True)
pr.columns = ["set", "subset", "mostpop", "ETeR-X", "USEM", "BERT"]
pr["set"] = pr["set"].ffill()
params = {(r.set, r.subset): (r["ETeR-X"], r.USEM, r.BERT) for _, r in pr.iterrows()}

colors = {"ETeR-X": "#4CAF50", "USEM": "#03A9F4", "BERT": "#ffa726", "EmbGemma": "#9C27B0", "Qwen3Emb": "#795548"}
names = list(colors)
widths = [len(l[2]) for l in layout]
fig, ax = plt.subplots(2, 3, figsize=(sum(widths) * .6 + 1, 3.5), gridspec_kw={"width_ratios": widths}, sharey="row")
for c, (sname, d, subs) in enumerate(layout):
    em = {n: [] for n in names}; pa = {n: [] for n in names}
    for ab, s in subs:
        em["ETeR-X"].append(co2("BOW2ITM", d, s)); em["USEM"].append(co2("USEM2ITM", d, s)); em["BERT"].append(co2("BERT2ITM", d, s))
        em["EmbGemma"].append(emb[("embgemma", d, s)]); em["Qwen3Emb"].append(emb[("qwen3emb", d, s)])
        p = params[(sname, ab)]
        pa["ETeR-X"].append(float(p[0])); pa["USEM"].append(float(p[1])); pa["BERT"].append(float(p[2]))
        pa["EmbGemma"].append(300e6); pa["Qwen3Emb"].append(600e6)
    x = np.arange(len(subs)); bw = 0.7 / len(names)
    for r, vals in enumerate((em, pa)):
        for i, n in enumerate(names):
            ax[r, c].bar(x + i * bw - 0.02, vals[n], bw - 0.02, label=n, color=colors[n])
        ax[r, c].set_yscale("log"); ax[r, c].set_axisbelow(True); ax[r, c].grid(axis="y")
    ax[0, c].set_title(sname); ax[0, c].set_xticks([])
    ax[1, c].set_xticks(x + bw * (len(names) - 1) / 2); ax[1, c].set_xticklabels([a for a, _ in subs])
ax[0, 0].set_ylabel("Emissions (g CO$_2$)"); ax[1, 0].set_ylabel("Parameters")
h, l = ax[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.025), ncol=5)
plt.tight_layout(); fig.subplots_adjust(wspace=0.1, hspace=0.1, top=0.85, bottom=0.15)
plt.savefig(out, bbox_inches="tight")
print("saved", out)
