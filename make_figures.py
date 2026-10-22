"""Figures of the README, drawn from the EDF Corsica open data in mimo_mh/Data.csv.

    python make_figures.py

Writes figures/*.png and figures/*.pdf. Requires numpy, pandas and matplotlib.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dfe5ec",
    "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200,
})

raw = pd.read_csv(HERE / "mimo_mh" / "Data.csv")
raw["t"] = pd.to_datetime(raw["Date"], utc=True).dt.tz_convert("Europe/Paris")
cols = {"Thermique (MW)": "Thermal", "Importations (MW)": "Imports", "Hydraulique (MW)": "Hydropower",
        "Micro-hydraulique (MW)": "Micro-hydro", "Bioénergies (MW)": "Bioenergy", "Eolien (MW)": "Wind",
        "Solaire photovoltaïque (MW)": "Solar PV"}
colors = {"Thermal": "#6b7785", "Imports": "#9b6fb0", "Hydropower": "#2e86ab", "Micro-hydro": "#7fc4d8",
          "Bioenergy": "#3c9d5d", "Wind": "#a8c66c", "Solar PV": "#e0a03a"}
df = raw.set_index("t")[list(cols)].rename(columns=cols).clip(lower=0)
df = df[~df.index.duplicated()]
total = raw.set_index("t")["Production totale (MW)"]
total = total[~total.index.duplicated()]


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# 1. One summer week of the Corsican mix
week = df.loc["2022-07-04":"2022-07-10"]
fig, ax = plt.subplots(figsize=(10, 4))
ax.stackplot(week.index, [week[c] for c in cols.values()], labels=list(cols.values()),
             colors=[colors[c] for c in cols.values()], alpha=0.9, lw=0)
ax.plot(total.loc["2022-07-04":"2022-07-10"], color="#1b2a3a", lw=1.4, label="Total production (= demand)")
ax.set_ylabel("Power (MW)")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9)
ax.margins(x=0)
fig.autofmt_xdate()
save(fig, "fig1_week_mix")

# 2. Mean daily profile, winter and summer
prof = df.copy()
prof["hour"] = prof.index.hour
prof["season"] = prof.index.month.map(lambda m: "Winter (Dec–Feb)" if m in (12, 1, 2)
                                      else "Summer (Jun–Aug)" if m in (6, 7, 8) else None)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
for ax, season in zip(axes, ["Winter (Dec–Feb)", "Summer (Jun–Aug)"]):
    p = prof[prof.season == season].groupby("hour")[list(cols.values())].mean()
    ax.stackplot(p.index, [p[c] for c in cols.values()], colors=[colors[c] for c in cols.values()], alpha=0.9,
                 labels=list(cols.values()), lw=0)
    ax.set_title(season)
    ax.set_xlabel("Hour of the day")
    ax.set_xticks(range(0, 24, 3))
    ax.margins(x=0)
axes[0].set_ylabel("Mean power (MW)")
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, ncol=7, loc="lower center", bbox_to_anchor=(0.5, 0.98), fontsize=8.5)
fig.tight_layout()
save(fig, "fig2_daily_profiles")

# 3. SISO, MIMO and MIMO-MH: what one model predicts
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
specs = [("SISO", "one model per source\nand per horizon", 1, 1),
         ("MIMO", "one model per horizon,\nall sources at once", 7, 1),
         ("MIMO-MH", "one model, all sources\nand all horizons", 7, 6)]
for ax, (name, sub, n_src, n_h) in zip(axes, specs):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(f"{name}\n{sub}", fontsize=10)
    ax.add_patch(FancyBboxPatch((0.4, 3.6), 2.6, 2.8, boxstyle="round,pad=0.1", fc="#e5eef7", ec="#2e86ab"))
    txt = "past 48 h\nof one\nsource" if name == "SISO" else "past 48 h\nof all\nsources\n+ hour"
    ax.text(1.7, 5.0, txt, ha="center", va="center", fontsize=8.5)
    ax.add_patch(FancyBboxPatch((3.7, 4.0), 1.8, 2.0, boxstyle="round,pad=0.1", fc="#fff4e0", ec="#e0a03a"))
    ax.text(4.6, 5.0, "ELM", ha="center", va="center", fontsize=10, weight="bold")
    ax.add_patch(FancyArrowPatch((3.1, 5.0), (3.6, 5.0), arrowstyle="-|>", mutation_scale=12, color="#1b2a3a"))
    ax.add_patch(FancyArrowPatch((5.6, 5.0), (6.1, 5.0), arrowstyle="-|>", mutation_scale=12, color="#1b2a3a"))
    cell_w, cell_h = 3.4 / max(n_h, 1), 5.6 / 7
    for i in range(7):
        for j in range(6):
            on = i < n_src and j < n_h
            ax.add_patch(plt.Rectangle((6.3 + j * 3.4 / 6, 2.2 + i * 5.6 / 7), 3.4 / 6 * 0.92, 5.6 / 7 * 0.85,
                                       fc="#3c9d5d" if on else "#edf2f7", ec="white"))
    ax.text(8.0, 1.6, "horizons →", ha="center", fontsize=8)
    ax.text(9.95, 5.0, "sources", rotation=90, ha="left", va="center", fontsize=8)
fig.text(0.5, -0.02, "Green cells: values predicted by a single trained model (7 series, here 6 of the 1–24 h horizons shown).",
         ha="center", fontsize=9, color="#46586a")
save(fig, "fig3_siso_mimo_mimomh")
print("figures written to", FIG)
