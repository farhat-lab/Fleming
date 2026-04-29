import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('evaluation/bench_validation/designed_mols/Physiochemical Properties of Generated Molecules.xlsx')

molecule_labels = [f"Molecule {i}" for i in df["Molecule ID"]]
data_columns = df.columns[1:]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12,16))
axes = axes.flatten()

n_molecules = len(molecule_labels)

for i, col in enumerate(data_columns):
    ax = axes[i]
    ax.plot(molecule_labels, df[col], 'o', markersize=8)
    ax.set_title(col)
    ax.set_ylabel("Value")
    ax.set_xticks(range(n_molecules))

    # Show x-tick labels only on the bottom row (row indices 2, i.e. axes 6,7,8)
    if i // 3 == 2:
        ax.set_xticklabels(molecule_labels, rotation=45, ha='right')
    else:
        ax.set_xticklabels([])  # hide x-tick labels for top and middle rows
        ax.tick_params(axis='x', length=0)  # also hide tick marks on x axis

# Hide unused subplots
for j in range(len(data_columns), len(axes)):
    axes[j].axis('off')

# Hide right and top spines for all subplots
for ax in axes:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('evaluation/bench_validation/designed_mols/output/physiochemical_properties.png')
