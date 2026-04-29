import pandas as pd

batch_1_lumi = pd.read_excel('data/inhibition_training_data/tb_inhibition_A&M/raw_data/Batch1_lumi.xlsx')[['SMILES']]
batch_1_resa = pd.read_excel('data/inhibition_training_data/tb_inhibition_A&M/raw_data/Batch1_resazurin.xlsx')[['SMILES']]
batch_2_resa = pd.read_excel('data/inhibition_training_data/tb_inhibition_A&M/raw_data/Batch2_resazurin.xlsx')[['SMILES']]
batch_3_lumi = pd.read_excel('data/inhibition_training_data/tb_inhibition_A&M/raw_data/Sac1_WCA_forMaha.xlsx')[['SMILES']]

batch_1_lumi['readout'] = 'lumi'
batch_1_resa['readout'] = 'resa'
batch_2_resa['readout'] = 'resa'
batch_3_lumi['readout'] = 'lumi'

all_data = pd.concat([batch_1_lumi, batch_1_resa, batch_2_resa, batch_3_lumi], ignore_index=True)
print(len(all_data['SMILES'].unique()))

# Check Replicates across readouts
print("Replicates across readouts:")
dup_across_counts = all_data.groupby('SMILES')['readout'].nunique()
print(len(dup_across_counts[dup_across_counts > 1]))

# Check Replicates for each readout
for readout in ['lumi', 'resa']:
    readout_data = all_data[all_data['readout'] == readout]
    readout_dup_counts = readout_data['SMILES'].value_counts()
    print(f"Readout: {readout}")
    print(len(readout_data))
    print(len(readout_dup_counts[readout_dup_counts > 1]))

# Test Interbatch Molecule Duplicate
print("Interbatch duplicates:")
print(len(set(batch_1_lumi['SMILES']).intersection(set(batch_1_resa['SMILES']))))
print(len(set(batch_1_lumi['SMILES']).intersection(set(batch_2_resa['SMILES']))))
print(len(set(batch_1_lumi['SMILES']).intersection(set(batch_3_lumi['SMILES']))))
print(len(set(batch_1_resa['SMILES']).intersection(set(batch_2_resa['SMILES']))))
print(len(set(batch_1_resa['SMILES']).intersection(set(batch_3_lumi['SMILES']))))
print(len(set(batch_2_resa['SMILES']).intersection(set(batch_3_lumi['SMILES']))))