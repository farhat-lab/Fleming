import pandas as pd
import numpy as np
from descriptastorus.descriptors import rdNormalizedDescriptors
from tqdm import tqdm
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt

def fp_significance(screening_results_dir: str,
                    feature_dir: str,
                    method: str = 'SHAP',
                    confidence_cutoff: float = 0.7):

    predictions = pd.read_csv(screening_results_dir)

    ## Generate Features
    features = []
    try:
        feature_df = pd.read_csv(feature_dir)
    except:
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        for smiles in tqdm(predictions['SMILES']):
            features.append(generator.process(smiles)[1:])
    
        feature_df = pd.DataFrame(features)
        feature_df['SMILES'] = predictions['SMILES']
        feature_df = feature_df.fillna(0)
        feature_df.to_csv(feature_dir,
                          index = False)

    predictions = predictions.merge(feature_df, on='SMILES',how='inner')
    feature_names = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
                         'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v',
                         'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 'EState_VSA2',
                         'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
                         'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt',
                         'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
                         'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
                         'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex',
                         'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge',
                         'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
                         'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount',
                         'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
                         'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
                         'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
                         'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
                         'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons',
                         'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
                         'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5',
                         'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount',
                         'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
                         'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
                         'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4',
                         'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9',
                         'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
                         'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8',
                         'VSA_EState9', 'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN',
                         'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2',
                         'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0',
                         'fr_NH1', 'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
                         'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate', 'fr_alkyl_halide',
                         'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline', 'fr_aryl_methyl',
                         'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine',
                         'fr_bicyclic', 'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester',
                         'fr_ether', 'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
                         'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan', 'fr_ketone',
                         'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine',
                         'fr_nitrile', 'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho',
                         'fr_nitroso', 'fr_oxazole', 'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol',
                         'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine',
                         'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
                         'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole',
                         'fr_thiazole', 'fr_thiocyan', 'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed']
    
    predictions.columns = ['SMILES','inhibition','uncertainty'] + feature_names

    feature_significance_df = []

    for feature in tqdm(feature_names):

        if len(predictions[feature].unique()) < 2:
            print(feature, "only have one value")
            continue
        elif (len(predictions[feature].unique()) >=2) and ((len(predictions[feature].unique()) < 10)):
            too_similar = False
            for value in predictions[feature].unique():
                if (len(predictions[predictions[feature] == value]) > (0.9 * len(predictions))):
                    print("Values too similar")
                    too_similar = True

            if too_similar:
                continue

        low_uncertainty_group = predictions[predictions['uncertainty'] < 1 - confidence_cutoff][feature].to_list()
        high_uncertainty_group = predictions[predictions['uncertainty'] >= 1 - confidence_cutoff][feature].to_list()

        p_value =  ttest_ind(low_uncertainty_group, high_uncertainty_group).pvalue

        # feature_significance_df['feature'] = feature
        # feature_significance_df['pvalue'] = p_value
        
        # feature_significance_df['high_confidence_mean'] = round(sum(low_uncertainty_group) / len(low_uncertainty_group),4)
        # feature_significance_df['high_confidence_std'] = np.std(low_uncertainty_group)

        # feature_significance_df['low_confidence_mean'] = round(sum(high_uncertainty_group) / len(high_uncertainty_group),4)
        # feature_significance_df['low_confidence_std'] = np.std(high_uncertainty_group)

        effective_size = cohend(low_uncertainty_group, high_uncertainty_group)

        # Volcano Plot Stats
        log2_foldchange = np.mean(np.log2(low_uncertainty_group)) - np.mean(np.log2(high_uncertainty_group))

        row = [feature, 
               round(sum(low_uncertainty_group) / len(low_uncertainty_group),4), 
               np.std(low_uncertainty_group), 
               round(sum(high_uncertainty_group) / len(high_uncertainty_group),4), 
               np.std(high_uncertainty_group), 
               p_value, 
               effective_size,
               log2_foldchange]

        feature_significance_df.append(row)

    feature_significance_df = pd.DataFrame(feature_significance_df,
                                           columns=['feature','high_confidence_mean','high_confidence_std','low_confidence_mean','low_confidence_std','pvalue','cohend','log2FC'])

    up = feature_significance_df[(feature_significance_df['log2FC']>=2) & (feature_significance_df['pvalue'] < 0.01)].sort_values(by = ['log2FC'], ascending=False)
    down = feature_significance_df[(feature_significance_df['log2FC']<=-2) & (feature_significance_df['pvalue'] < 0.01)].sort_values(by = ['log2FC'])

    top_two_features = pd.concat([up[:1], down[:1]]).reset_index(drop=True)
    print(top_two_features)

    ## Volcano Plot
    plt.figure()
    ax = plt.subplot(111)

    plt.scatter(x = feature_significance_df['log2FC'],
                y = feature_significance_df['pvalue'].apply(lambda x: -np.log10(x)), 
                s = 10)
    plt.scatter(x = up['log2FC'],
                y = up['pvalue'].apply(lambda x: -np.log10(x)), 
                s = 15,
                color = "blue",
                label = "Higher in Unconfident")
    plt.scatter(x = down['log2FC'],
                y = down['pvalue'].apply(lambda x: -np.log10(x)), 
                s = 15,
                color = "red",
                label = "Higher in Confidenct")

    plt.annotate(list(top_two_features['feature'])[0],
                 (list(top_two_features['log2FC'])[0] - 4.25, list(top_two_features['pvalue'].apply(lambda x: -np.log10(x)))[0] + 1.2))
    
    plt.annotate(list(top_two_features['feature'])[1],
                 (list(top_two_features['log2FC'])[1] - 2.5, list(top_two_features['pvalue'].apply(lambda x: -np.log10(x)))[1] + 1.2))

    plt.xlabel("logFC")
    plt.ylabel("-log10 p-value")
    plt.axvline(-2, 
                color = 'grey',
                linestyle='--')
    plt.axvline(2, 
                color = 'grey',
                linestyle='--')
    plt.axhline(2,
                color = 'grey',
                linestyle='--')
    plt.legend()
    plt.xlim(-15,17.5)
    ax.spines[['right','top']].set_visible(False)

    plt.savefig('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/confidence_clustering/volcano_plot.png')
    exit()

    feature_significance_df = feature_significance_df[feature_significance_df['pvalue']<0.0001]

    feature_significance_df['abs_cohend'] = np.abs(feature_significance_df['cohend'])
    feature_significance_df = feature_significance_df.sort_values(by='abs_cohend', 
                                                                  ascending=False)
    feature_significance_df = feature_significance_df[['feature','high_confidence_mean','high_confidence_std','low_confidence_mean','low_confidence_std','pvalue','cohend']]

    feature_significance_df.to_csv('/home/aw274/Mtb_Inhibition_DL_Pipeline/evaluation/confidence_clustering/testset_significant_features.csv',
                                   index = False)

    

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s


if __name__ == '__main__':
    fp_significance(screening_results_dir = '/home/aw274/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/CompoundFragment_inclusive_with_antibiotics_evidential_classification_0.2_output/fold_0/test_predictions.csv',
                    feature_dir = '/home/aw274/Mtb_Inhibition_DL_Pipeline/model_checkpoints/classifier_checkpoints/CompoundFragment_inclusive_with_antibiotics_evidential_classification_0.2_output/fold_0/test_rdkit_features.csv',
                    confidence_cutoff = 0.55)