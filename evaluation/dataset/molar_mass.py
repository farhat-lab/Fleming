import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

dataset = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/tb_inhibition_A&M/final_data/CompoundFragment_Inhibition_114933_mols_inclusive_with_antibiotics.csv')
fragment_data = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/inhibition_training_data/fragment/binarized_fragment_inhibition_999_mols.csv')
all_gen_mols = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/generative/molecular_optimization_result_v2/all_generated_molecules.csv')
dragonfold_data = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/screening_result/screening_labels.csv')
dragonfold_predicted_six = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/screening_result/final_output.csv')
bench_validated_generated = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/generative_model/bench_validation/verified_six.csv')

# Apply to DataFrame
n = len(dataset)
dataset['label'] = 'compounds'
mask = dataset['SMILES'].isin(fragment_data['SMILES'])
dataset.loc[mask, 'label'] = 'fragments'
dataset.loc[n-14:, 'label'] = 'current_antibiotics'

all_gen_mols = all_gen_mols[['SMILES','source']].copy()
all_gen_mols = all_gen_mols.rename(columns={'source':'label'})
dataset = pd.concat([dataset, all_gen_mols], ignore_index=True)

dragonfold_data = dragonfold_data[['SMILES']].copy()
dragonfold_data['label'] = 'dragonfold'
dataset = pd.concat([dataset, dragonfold_data], ignore_index=True)

dragonfold_predicted_six = dragonfold_predicted_six[['SMILES']].copy()
dragonfold_predicted_six['label'] = 'dragonfold_predicted_six'
dataset = pd.concat([dataset, dragonfold_predicted_six], ignore_index=True)

bench_validated_generated = bench_validated_generated[['SMILES']].copy()
bench_validated_generated['label'] = 'bench_validated_generated'
dataset = pd.concat([dataset, bench_validated_generated], ignore_index=True)

# Function to compute molar mass
def calculate_molar_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Descriptors.MolWt(mol)
    else:
        return None  # or np.nan

dataset['molar_mass'] = dataset['SMILES'].apply(calculate_molar_mass)

# Display result
dataset.to_csv('evaluation/dataset/training_dataset_molar_mass_labeled.csv', index=False)

def iqr(x):
    return x.quantile(0.75) - x.quantile(0.25)

# Group by 'label' and compute stats
summary = dataset.groupby('label')['molar_mass'].agg(
    count='count',
    min='min',
    max='max',
    mean='mean',
    median='median',
    q1=lambda x: x.quantile(0.25),
    q3=lambda x: x.quantile(0.75),
    iqr=lambda x: x.quantile(0.75) - x.quantile(0.25)
).reset_index()

summary.to_csv('evaluation/dataset/molar_mass_summary.csv', index=False)
print(summary)
