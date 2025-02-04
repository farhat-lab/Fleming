
import pandas as pd

def return_result(response):
	if "I'm sorry, I can't assist with that" in response:
		return -1
	if '**Unfavorable**' in response or 'Result: Unfavorable' in response or 'Result**: Unfavorable' in response or 'Result Unfavorable' in response:
		prediction = 1
	elif '**Favorable**' in response or 'Result: Favorable' in response or 'Result**: Favorable' in response or 'Result Favorable' in response:
		prediction = 0
	else:
		prediction = response.split('Result')[-1].split('Likelihood')[0].split('\n')[0].split('Reasoning')[0].replace('*','').replace(':','').replace('.','').strip().replace('-','')
		if prediction == 'Unfavorable':
			prediction = 1
		elif prediction == 'Favorable':
			prediction = 0
		else:
			# if len(prediction) > len('Unfavorable'):
			# 	import pdb; pdb.set_trace()
			import pdb; pdb.set_trace()
	return prediction


def return_likelihood(response):
	prediction = response.replace('*','').split('Likelihood')[-1].split('\n')[0].split('Reasoning')[0].replace('*','').replace(':','').replace('(', '').strip().split(' ')[0]
	return float(prediction)

inject_ADMET = True

#dataset = 'DILI'
#dataset = 'hERG'
dataset = 'Carcinogens_Lagunin'
if not inject_ADMET:
    df = pd.read_csv(f'./{dataset}/new_results_final.tsv', sep = '\t', header = None)
else:
    df = pd.read_csv(f'./{dataset}/ADMET_results.tsv', sep = '\t', header = None)

smiles = []
gpt4 = []
gpt4_label = []
gpt4_likelihood = []
pqa = []
pqa_label = []
pqa_likelihood = []
label = []
admet_predictions = []

for index, row in df.iterrows():
    drug = row[0]
    description = row[1]
    prompt = row[2]
    response = row[3]
    label = row[4]
    admet_prediction = float(row[5])
    likelihood = row[6]
    final_likelihood = row[7]
    if inject_ADMET:
        admet_prediction = row[8]
        model = row[9]
    else:
        model = row[8]

    prediction = return_result(response)

    
    if model == 'PQA':
        pqa.append(prediction)
        pqa_label.append(label)
        pqa_likelihood.append(final_likelihood)
        if inject_ADMET:
            admet_predictions.append(admet_prediction)
    elif model == 'GPT4':
        gpt4.append(prediction)
        gpt4_label.append(label)
        gpt4_likelihood.append(final_likelihood)
    


#Calculate AUROC 

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import pdb; pdb.set_trace()
print(f'PQA AUROC: {roc_auc_score(pqa_label, pqa)}')
print(f'PQA Likelihood AUROC: {roc_auc_score(pqa_label, pqa_likelihood)}')
print(f'PQA Accuracy: {accuracy_score(pqa_label, pqa)}')
print(f'GPT4 AUROC: {roc_auc_score(gpt4_label, gpt4)}')
import pdb; pdb.set_trace()
print(f'GPT4 Likelihood AUROC: {roc_auc_score(gpt4_label, gpt4_likelihood)}')
print(f'GPT4 Accuracy: {accuracy_score(gpt4_label, gpt4)}')
print(f'ADMET AUROC: {roc_auc_score(pqa_label, admet_predictions)}')



