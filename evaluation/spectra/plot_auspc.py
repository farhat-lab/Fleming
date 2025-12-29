import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import integrate

# ---- CONFIG ----
# Load your CSV file
final_df = pd.read_excel("/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/spectra_result_full.xlsx")  # Update with your actual filename
final_df['Spectra Parameter'] = 1-final_df['Spectra Parameter']/20

model_order = [
    "Evidential DMPNN", "DMPNN", "EGNN",  "FFNN", "Transformer-M", "Random Forest"
]
legend = {
    "DMPNN": "DMPNN",
    "Evidential DMPNN": "Evidential DMPNN",
    "EGNN": "EGNN",
    "Transformer-M": "Transformer-M",
    "FFNN": "FFNN",
    "Random Forest": "Random Forest",
}
metric = ["AUROC", "std"]

# ---- PLOTTING ----
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["legend.fontsize"] = 20

fig, ax = plt.subplots(figsize=(9.6, 7.2))
line_width = 5

colors = ["#D55E00", "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442"]
palette_dict = {legend.get(m, m): c for m, c in zip(model_order, colors)}

from pandas.api.types import CategoricalDtype
model_cat = CategoricalDtype(categories=model_order, ordered=True)
final_df['Model'] = final_df['Model'].astype(model_cat)
final_df = final_df.sort_values(['Model','Spectra Parameter'], ascending=False).reset_index(drop=True)

def calculate_aumc(y, x):
    return integrate.simps(y, x)

for key, group in final_df.groupby("Model"):
    if key not in legend:
        continue
    x = group["Spectra Parameter"]
    y = group[metric[0]]
    yerr = group[metric[1]]

    auspc = -1 * calculate_aumc(y, x)
    auspc_low = -1 * calculate_aumc(y - yerr, x)
    auspc_high = -1 * calculate_aumc(y + yerr, x)
    auspc_label = f" - {auspc:.2f} [{auspc_low:.2f}-{auspc_high:.2f}]"

    group.plot(
        "Spectra Parameter",
        metric[0],
        yerr=metric[1],
        label=legend[key] + auspc_label,
        ax=ax,
        marker='o',
        linestyle='',
        markersize=line_width * 2,
        lw=line_width,
        color=palette_dict[legend[key]],
    )

ax.set_xlabel('Spectra Parameter', 
              fontsize=30, 
              #fontweight='bold'
              )
ax.set_ylabel('AUROC', 
              fontsize=30, 
              # fontweight='bold'
              )
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
for spine in ax.spines.values():
    spine.set_linewidth(line_width)

legend_obj = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
for text in legend_obj.get_texts():
    text.set_fontweight("bold")
    text.set_fontsize(20)

plt.savefig("/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/auspc_new.png", bbox_inches="tight")
plt.savefig("/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/spectra/auspc_new.pdf", bbox_inches="tight")
