import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

from admet_model.current_antibiotic_threshold import get_admet_threshold

raw_prediction_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/confidence_clustering/dragonfold_prediction_dir.csv')
admet_prediction_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/final_output.csv')
admet_task_df = pd.read_csv('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/output/dragonfold/leadlike/raw_admet_prediction.csv')

threshold_antibiotic_dict, threshold_value_dict, threshold_type_dict = get_admet_threshold('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/admet_model/current_antibiotic_admet_predictions.csv')

## Plot Inhibition
count, bins_count = np.histogram(raw_prediction_df['inhibition'], bins=50)
pdf = count / sum(count)
cdf = np.cumsum(pdf)

plt.figure(figsize=(6.4,2.4))
ax = plt.subplot(111)
ax.plot(cdf, bins_count[1:], color='black', label = 'Inhibition Model Prediction Score')
plt.xlabel('Quantile')
plt.ylabel('M.tb Inhibition')
ax.spines[['right','top']].set_visible(False)
plt.savefig('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/prediction_distribution/prediction_distribution_inhibition.png')
plt.close()

## Plot Confidence
count, bins_count = np.histogram(1 - raw_prediction_df['inhibition_dirichlet_uncal_uncertainty'], bins=50)
pdf = count / sum(count)
cdf = np.cumsum(pdf)

plt.figure(figsize=(6.4,2.4))
ax = plt.subplot(111)
ax.plot(bins_count[1:], pdf, color='black', label = 'Inhibition Model Confidence Score')
plt.xlabel('Prediction Confidence')
plt.ylabel('% of Total Molecules')
plt.xlim(0.4,1.0)
ax.spines[['right','top']].set_visible(False)
plt.savefig('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/prediction_distribution/prediction_distribution_inhibition_confidence.png')
plt.close()

## Plot ADMET - Overall
count, bins_count = np.histogram(((37 - admet_prediction_df['fail_sum'])/37*100), bins=10)
pdf = count / sum(count)
cdf = np.cumsum(pdf)

plt.figure(figsize=(6.4,2.4))
ax = plt.subplot(111)
ax.plot(bins_count[1:], pdf, color='black', label = 'Total ADMET Pass Num Distribution')
# ax.hist(((37 - admet_prediction_df['fail_sum'])/37*100))
plt.xlabel('% of passed ADMET Tasks')
plt.ylabel('% of Total Molecules')
plt.xlim(35,100)
ax.spines[['right','top']].set_visible(False)
plt.savefig('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/prediction_distribution/prediction_distribution_admet.png')
plt.close()

## Plot ADMET - Individual Tasks
absorption_tasks = ['Bioavailability_Ma','HIA_Hou', 'PAMPA_NCATS','Pgp_Broccatelli','Solubility_AqSolDB',
                    'Caco2_Wang','Lipophilicity_AstraZeneca']
# absorption_tasks_pass = [f"pass_{item}" for item in absorption_tasks]
for item in absorption_tasks:
    plt.figure(figsize=(6.4,2.4))
    ax = plt.subplot(111)
    plt.hist(admet_task_df[item],
             alpha=0.7,
             density = True,
             color = 'grey',
             bins=20)
    plt.axvline(x=threshold_value_dict[item],
                color='blue',
                linewidth=2.0)
    ax.set_xlabel(f"Predicted_{item}")
    ax.set_ylabel('Percentage of Molecules')
    ax.spines[['right','top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/prediction_distribution/individual_pharmacokinetics_plot/absorption_{item}_distribution')
    plt.close()

distribution_tasks = ['BBB_Martins','PPBR_AZ','VDss_Lombardo']
# distribution_tasks = [f"pass_{item}" for item in distribution_tasks]
for item in distribution_tasks:

    plt.figure(figsize=(6.4,2.4))
    ax = plt.subplot(111)
    plt.hist(admet_task_df[item],
             alpha=0.7,
             density = True,
             color = 'grey',
             bins=20)
    plt.axvline(x=threshold_value_dict[item],
                color='blue',
                linewidth=2.0)
    ax.set_xlabel(f"Predicted_{item}")
    ax.set_ylabel('Percentage of Molecules')
    ax.spines[['right','top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/prediction_distribution/individual_pharmacokinetics_plot/distribution_{item}_distribution')
    plt.close()

metabolism_tasks = ['CYP1A2_Veith','CYP2C19_Veith','CYP2C9_Veith','CYP2D6_Veith','CYP3A4_Veith',]
# metabolism_tasks = [f"pass_{item}" for item in metabolism_tasks]
for item in metabolism_tasks:

    plt.figure(figsize=(6.4,2.4))
    ax = plt.subplot(111)
    plt.hist(admet_task_df[item],
             alpha=0.7,
             density = True,
             color = 'grey',
             bins=20)
    plt.axvline(x=threshold_value_dict[item],
                color='blue',
                linewidth=2.0)
    ax.set_xlabel(f"Predicted_{item}")
    ax.set_ylabel('Percentage of Molecules')
    ax.spines[['right','top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/prediction_distribution/individual_pharmacokinetics_plot/metabolism_{item}_distribution')
    plt.close()

excretion_tasks = ['Clearance_Hepatocyte_AZ','Clearance_Microsome_AZ','Half_Life_Obach']
# excretion_tasks = [f"pass_{item}" for item in excretion_tasks]
for item in excretion_tasks:

    plt.figure(figsize=(6.4,2.4))
    ax = plt.subplot(111)
    plt.hist(admet_task_df[item],
             alpha=0.7,
             density = True,
             color = 'grey',
             bins=20)
    plt.axvline(x=threshold_value_dict[item],
                color='blue',
                linewidth=2.0)
    ax.set_xlabel(f"Predicted_{item}")
    ax.set_ylabel('Percentage of Molecules')
    ax.spines[['right','top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/prediction_distribution/individual_pharmacokinetics_plot/excretion_{item}_distribution')
    plt.close()

toxicity_tasks = ['AMES','Carcinogens_Lagunin','ClinTox','DILI',
                  'NR-AR-LBD','NR-AR','NR-AhR','NR-Aromatase','NR-ER-LBD',
                  'NR-ER','NR-PPAR-gamma','SR-ATAD5','SR-HSE','SR-MMP','SR-p53',
                  'Skin_Reaction','hERG','LD50_Zhu','SR-ARE']
# toxicity_tasks = [f"pass_{item}" for item in toxicity_tasks]
for item in toxicity_tasks:

    plt.figure(figsize=(6.4,2.4))
    ax = plt.subplot(111)
    plt.hist(admet_task_df[item],
             alpha=0.7,
             density = True,
             color = 'grey',
             bins=20)
    plt.axvline(x=threshold_value_dict[item],
                color='blue',
                linewidth=2.0)
    ax.set_xlabel(f"Predicted_{item}")
    ax.set_ylabel('Percentage of Molecules')
    ax.spines[['right','top']].set_visible(False)
    plt.tight_layout()
    plt.savefig(f'/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/prediction_distribution/individual_pharmacokinetics_plot/toxicity_{item}_distribution')
    plt.close()


## Plot Agent
plt.figure()
plt.hist(admet_prediction_df['mean_rate'], label = 'Agent Scoring Distribution')
plt.xlabel('Agent Score')
# plt.ylabel('Density')
plt.savefig('/n/data1/hms/dbmi/farhat/alex/Mtb_Inhibition_DL_Pipeline/evaluation/prediction_distribution/agent_score_distribution.png')

