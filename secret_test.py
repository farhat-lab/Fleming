import os
from openai import AzureOpenAI
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import io
import base64
import pqapi
import pandas as pd
from collections import Counter
from admet_ai import ADMETModel
from pathlib import Path
# from ADMET_wrapper import ADMET_AI
import numpy as np


azure_openai_api_key = 'f65f6d81f08947a488e4599b08110058'
azure_openai_endpoint = 'https://azure-ai-dev.hms.edu'
pqapi_token = 'pqa-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1za2FybGluc2tpQGZ1dHVyZWhvdXNlLm9yZyIsImV4cCI6MTc2MjQ3NjYwNC4xOTU2NzJ9.FaS6uy0G_StC8SLEBtEEf44Ms50TzIB0IuWf2jBqyEw'
os.environ['PQA_API_TOKEN'] = pqapi_token



#test = pqapi.agent_query(f"What chemical substructures in a drug tend to cause drug-induced liver injury?").answer.answer
#test = pqapi.agent_query(f"What chemical substructures in a drug tend to cause herg channel blockage?").answer.answer
#test = pqapi.agent_query(f"What chemical substructures in a drug tend to cause cancer?").answer.answer
# #test = pqapi.agent_query("Can a cyclohexanol moiety in a drug cause drug-induced liver injury?").answer.answer
# #test = pqapi.agent_query("Does having chlorine in a drug cause drug-induced liver injurty?").answer.answer
# test = pqapi.agent_query("Do hexachlorocyclohexanes cause drug-induced liver injury?").answer.answer
# print(test)
# import pdb; pdb.set_trace()


def launch_gpt4(message, temp = 0.5, max_tokens = 250, image = None):
	if image is None:
		message = [{"role": "system", "content": "You are a medicinal chemist."}, {"role":"user", "content": message}]
	else:
		new_content = [
        {
          "type": "text",
          "text": message
        },
        {
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/jpeg;base64,{image}"
          },
        },
      ]
		message = [{"role": "system", "content": "You are a medicinal chemist."}, {"role":"user", "content": new_content}]
		
	client = AzureOpenAI(
		azure_endpoint = azure_openai_endpoint,
		api_key=azure_openai_api_key,
		api_version="2024-05-01-preview")

	response = client.chat.completions.create(
		model='gpt-4o',
		messages = message,
		temperature=temp,
		max_tokens=max_tokens,
	)

	return response.choices[0].message.content

def run_substructure_analysis(molecule_context, sub_structures):
	results = []

	for sub in tqdm(sub_structures):
		molecule = sub.split(':')[0]
		new_prompt = molecule_context + 'This is information about ' + sub + f'From this information along with knowing its context in the molecule from description does the addition of {molecule} in this molecule influence its liklihood of being a carcinogen? If so does it influence it favorably or unfavorably?'
		new_res = launch_gpt4(new_prompt, temp=0.1, max_tokens=1000)
		results.append(new_res)
	
	final_prompt = ''' Ill provide you with a molecule and its description. Then I will provide you reasoning on the substructures and how they influence the potential for this drug to be a carcinogen. From this reasoning say whether the molecule is likely to be or not to be a carcinogen.'''
	final_prompt = final_prompt + molecule_context + '\t'.join(results) + 'Give the answer as Result: Favorable or Unfavorable. Provide reasoning.'
	final_result = launch_gpt4(final_prompt, temp=0.1, max_tokens=1000)
	import pdb; pdb.set_trace()
	return final_result


def encode_image(image):
# Convert the image to a binary format using an in-memory bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG')  # Ensure the format matches your image type, e.g., 'PNG' or 'JPEG'
    image_bytes = buffer.getvalue()

    # Encode the binary data to Base64
    image_base64 = base64.b64encode(image_bytes)

    # Convert Base64 bytes to a UTF-8 string
    image_base64_utf8 = image_base64.decode('utf-8')

    return image_base64_utf8


def generate_image(smiles):
    molecule = Chem.MolFromSmiles(smiles)

    # Generate an image of the molecule
    image = Draw.MolToImage(molecule, size=(300, 300))
    return encode_image(image)

def return_result(response):
	if "I'm sorry, I can't assist with that" in response:
		return -1
	if '**Unfavorable**' in response or 'Result: Unfavorable' in response or 'Result**: Unfavorable' in response or 'Result Unfavorable' in response:
		prediction = 1
	elif '**Favorable**' in response or 'Result: Favorable' in response or 'Result**: Favorable' in response or 'Result Favorable' in response:
		prediction = 0
	else:
		prediction = response.split('Result')[-1].split('Likelihood')[0].split('\n')[0].split('Reasoning')[0].replace('*','').replace(':','').replace('.','').strip()
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


#dataset = 'hERG'
#dataset = 'DILI'
dataset = 'Carcinogens_Lagunin'
inject_ADMET_AI = True
if inject_ADMET_AI:
	#ADMET_AI = ADMET_AI(dataset)
	ADMET_AI = ADMETModel(Path(f'./ADMET_predictions/{dataset}/'))

DILI_lit_info = "Drug-induced liver injury (DILI) is often associated with specific chemical substructures present in pharmaceutical compounds. Liu et al. identified several structural alerts linked to hepatotoxicity. These include anabolic and glucocorticoid steroid substructures (Alert 1), hydrazines (Alert 2), and arylacetic acids (Alert 3), the latter of which is associated with severe liver injuries and has led to the withdrawal of many drugs from the market (liu2015datadrivenidentificationof pages 5-6). Sulfonamides (Alert 4) are known to cause acute liver failure, while aniline moieties (Alert 5) are associated with mutagenicity (liu2015datadrivenidentificationof pages 5-6).\n\nAdditional alerts include proton pump inhibitors (Alert 6) and acyclic bivalent sulfur moieties (Alert 7), both linked to hepatotoxicity (liu2015datadrivenidentificationof pages 5-6). Acyclic di-aryl ketone moieties (Alert 8) have been found in drugs withdrawn due to severe liver injuries (liu2015datadrivenidentificationof pages 6-6). Other concerning substructures are halogen atoms bonded to sp3 carbons (Alert 9), para oxygen and nitrogen di-substituted benzene rings (Alert 11), and fused tricyclic moieties (Alert 12), which are known to induce liver toxicity through mechanisms such as mitochondrial dysfunction (liu2015datadrivenidentificationof pages 6-6).\n\nThese structural alerts are crucial for screening drug candidates to mitigate the risk of DILI during drug discovery and development (liu2015datadrivenidentificationof pages 1-2)."
hERG_lit_info = "Chemical substructures that tend to cause hERG channel blockage include several key features. A flexible molecule with a central tertiary amine, at least two aromatic moieties, and a polar group on one side of the molecule are critical for hERG binding activity. Additionally, hydrophobic and positive ionizable features contribute to hERG liability (ducuny2011acriticalassessment pages 2-2). \n\nThe presence of large groups and aryl groups is also significant, as these substructures interact with aryl residues Tyr652 and Phe656 in the hERG channel, blocking potassium ion passage. These substructures are characterized by high molecular weight, hydrophobicity, and large stereoscopic space, with specific examples including imide acidic, amidine, and sulfonamide groups (zhang2016insilicoprediction pages 17-18).\n\nFurthermore, the steric compatibility of blockers with the channel's constriction zone is crucial, with planar moieties like phenyl groups being significant for trappable blockers. The re-solvation cost and polarity of certain groups, such as Cl versus F and ortho- versus meta-pyridyl nitrogen, also affect hERG blockage (wan2020…invivorelevant pages 16-17). These structural characteristics collectively influence the interaction of drugs with the hERG channel, leading to potential blockage."
Carcinogen_lit_info = "Certain chemical substructures are associated with carcinogenicity, meaning they have the potential to cause cancer. Aromatic amine groups, such as ArNH2, ArNHR, and ArNR2, and aromatic nitro groups (ArNO2) are recognized as strong carcinogens. However, in the study by Tanabe et al., these groups were not found to be significantly prevalent in the dataset analyzed, suggesting that their presence alone may not be a definitive indicator of carcinogenicity (tanabe2010predictionofcarcinogenicity pages 7-8).\n\nAdditionally, aromatic nitroso (ArNO) and aromatic N-nitroso (ArNNOx) groups are also known carcinogens, but their limited presence in the dataset prevented them from being classified as significantly positive rich (tanabe2010predictionofcarcinogenicity pages 7-8). The study further identified that aliphatic N-nitroso and aliphatic halogen groups were more prevalent in carcinogenic chemicals, indicating a stronger association with cancer risk (tanabe2010predictionofcarcinogenicity pages 8-8).\n\nThe research highlights that while these substructures are linked to carcinogenicity, other factors beyond these chemical structures may contribute to the carcinogenic potential of chemicals. This complexity underscores the need for comprehensive models and datasets to accurately predict carcinogenicity (tanabe2010predictionofcarcinogenicity pages 8-8)."

#dataset = 'hERG'
#dataset = 'Carcinogens_Lagunin'
labels = []
old_res = []
new_res = []
new_likelihood = []
admet_ai_result = []

gpt4_labels = []
gpt4_old_res = []
gpt4_new_res = []
gpt4_new_likelihood = []


df = pd.read_csv(f'./{dataset}/results.tsv', sep = '\t', header = None)
result_file = open(f'./{dataset}/ADMET_results_fake.tsv', 'w')

for index, row in tqdm(df.iterrows(), total=int(df.shape[0])):
	drug = row[0]
	label = row[4]
	model = row[6]
	if model == 'PQA':
		if row[3] == 'Call to GPT4 failed for final analysis':
			prediction = 1
		else:
			prediction = return_result(row[3])
		result = generate_image(drug)
		new_prompt = row[2].replace('Keep into account that (1) Now take into account the structure that I gave you and whether the structure can shield any potential toxic functional groups via steric shielding or other mechanisms and (2) Also consider that just because a chemical is similar to another it does not necessarily mean they will have the same effect. It suggests it but it is not strong evidence for this.', '')
		#new_prompt += "Make sure to include all substructures in your reasoning."
		#new_prompt += f"To help with determining the potential for this drug to be toxic to the liver, consider the following information about liver toxicity: {lit_info}. This can help when a molecule is on the fence."
		#new_prompt += "Keep into account the fact that if subgroups have little to no effect but there are subgroups have an obvious, strong unfavorable effect, the molecule has the potential to be unfavorable unless steric interactions or other relevant structural mechanisms is present. In that case the molecule is more likely to be favorable Also single unfavorable subgroups, in the absence of strong evidence of bioactivation or reactive metabolite formation, should not dominate the risk assessment."
		#new_prompt += "Also keep in mind this drug would be given at a similar does as acetametophin, so a low dose in general. Make sure to keep this dosage in mind in your reasoning as drugs with subgroups with small or moderate effect would not be unfavorable in low doses. Acetametophin is not considered to cause liver-induced drug injurty. You must output your dosage reasoning. Low dosage can override the unfavorable effects of subgroups with small or moderate effect."
		add_on = "Also keep in mind this drug would be given at a similar does as acetametophin, so a low dose in general. Make sure to keep this dosage in mind in your reasoning as drugs with subgroups with small or moderate effect would not be unfavorable in low doses. Acetametophin is not considered to cause liver-induced drug injurty. You must output your dosage reasoning. Low dosage can override the unfavorable effects of subgroups with small or moderate effect. Does this change your response at all? Respond with favorable or unfavorable in this format Result:Favorable or Unfavorable"

		add_on_one = "Keep into account the fact that if subgroups have little to no effect but there are subgroups have an obvious, strong unfavorable effect, the molecule has the potential to be unfavorable unless steric interactions or other relevant structural mechanisms is present. In that case the molecule is more likely to be favorable Also single unfavorable subgroups, in the absence of strong evidence of bioactivation or reactive metabolite formation, should not dominate the risk assessment. Respond with favorable or unfavorable in this format Result:Favorable or Unfavorable"
		add_on_two = "Also keep in mind this drug would be given at a similar does as acetametophin, so a low dose in general. Make sure to keep this dosage in mind in your reasoning as drugs with subgroups with small or moderate effect would not be unfavorable in low doses. Acetametophin is not considered to cause liver-induced drug injurty. You must output your dosage reasoning. Low dosage can override the unfavorable effects of subgroups with small or moderate effect. Does this change your response at all? Respond with favorable or unfavorable in this format Result:Favorable or Unfavorable"

		info_to_inject = None
		prompt_to_use = None
		if dataset == 'DILI':
			info_to_inject = DILI_lit_info
			#prompt_to_use = f"{row[2]}. This is also some ground truth information that can help {info_to_inject}. Remember to return results as Result: Favorable or Unfavorable, where favorable means it won't cause injury to the liver and unfavorable means it will cause injury to the liver."
			# try:
			prompt_to_use = f"{row[2]}. This is also some ground truth information that can help {info_to_inject}. Also return the result as a decimal between 0 and 1 where 0 means it won't cause injury to the liver and 1 means it will cause injury to the liver. The closer to 0 the more confident you are it won't cause injury, the closer to 1 the more confident you are it will cause injury, the closer to 0.5 the more uncertain you are as to the effect of the drug on liver toxicity. Output this as Likelihood: <Decimal goes here>"
			
			# except:
			# 	print("FAILED")
			# 	res = row[3]
			# 	final_likelihood = 1
		elif dataset == 'hERG':
			info_to_inject = hERG_lit_info
			#prompt_to_use = f"{row[2]}. This is also some ground truth information that can help {info_to_inject}. Remember to return results as Result: Favorable or Unfavorable, where favorable means it won't bind to herG so won't cause herG blocking and unfavorable means it will bind to herG so will cause herG blocking."
			prompt_to_use = f"{row[2]}. This is also some ground truth information that can help {info_to_inject}. Also return the result as a decimal between 0 and 1 where 0 means it won't block the hERG channel and 1 it will block the hERG channel. The closer to 0 the more confident you are it won't block hERG, the closer to 1 the more confident you are it will block hERG, the closer to 0.5 the more uncertain you are as to the effect of the drug on hERG blockage. Output this as Likelihood: <Decimal goes here>"
			# res = launch_gpt4(prompt_to_use, temp = 0.000000001, max_tokens = 1000, image = result)
		elif dataset == 'Carcinogens_Lagunin':
			info_to_inject = Carcinogen_lit_info
			#prompt_to_use = f"{row[2]}. This is also some ground truth information that can help {info_to_inject}. Remember to return results as Result: Favorable or Unfavorable, where favorable means it is not a carcinogen and unfavorable means it is a carcinogen."
			prompt_to_use = f"{row[2]}. This is also some ground truth information that can help {info_to_inject}. Also return the result as a decimal between 0 and 1 where 0 means it is not a carcinogen and 1 it is a carcinogen. The closer to 0 the more confident you are it isn't a carcinogen, the closer to 1 the more confident you are it is a carcinogen, the closer to 0.5 the more uncertain you are as to whether the molecule is a carcinogen or not. Output this as Likelihood: <Decimal goes here>"
			if inject_ADMET_AI:
				ADMET_result = ADMET_AI.predict(drug)[dataset]
				prompt_to_use += "ADMET.AI is a ML model trained on carcinogenic molecules. Take into account the ADMET AI prediction for this molecule. If the ADMET AI prediction is favorable (closer to 0), it is less likely to be a carcinogen. If the ADMET AI prediction is unfavorable (closer to 1), it is more likely to be a carcinogen. If the ADMET AI prediction is uncertain (closer to 0.5), it is more uncertain as to whether the molecule is a carcinogen or not. You can decide whether to use the ADMET_AI prediction or not depending on how sure you are of your prediction."
				prompt_to_use += f"ADMET.AI had this prediction {ADMET_result}"
		else:
			raise Exception("No info!")
		
		likelihoods = []
		while len(likelihoods) < 2  or np.std(likelihoods) > 0.05:
			try:
				res = launch_gpt4(prompt_to_use, temp = 0.000000001, max_tokens = 1000, image = result)
				while 'sorry' in res or 'Sorry' in res:
					res = launch_gpt4(prompt_to_use, temp = 0.000000001, max_tokens = 1000, image = result)
					print("Repeating!")

				# import pdb; pdb.set_trace()
				likelihoods.append(return_likelihood(res))
				# import pdb; pdb.set_trace()
				# final_likelihood = return_likelihood(res)
				if len(likelihoods) > 2:
					print(f"Need more clarifications! Currently have this {likelihoods}")
				
				if len(likelihoods) > 10:
					break
			except:
				print("We got an error we retried it!")
				x = return_result(res)
				likelihoods = [x, x]

		print(f"{res}\t{label}\t{likelihoods}")
		final_likelihood = np.mean(likelihoods)
		final_result = return_result(res)
		

		if final_result == prediction and prediction == label:
			print("No change, correct stayed correct")
		elif final_result == prediction and prediction != label:
			print("No change, it stays wrong")
			#import pdb; pdb.set_trace()
		elif final_result != prediction and prediction != label:
			print("Change, corrected the output!")
		elif final_result != prediction and prediction == label:
			print("Change, made it wrong :/")
			#import pdb; pdb.set_trace()

		old_res.append(prediction)
		new_res.append(return_result(res))
		labels.append(label)
		new_likelihood.append(final_likelihood)	
		to_insert = res.replace('\n', '')
		prompt_to_include = prompt_to_use.replace('\n', '')

		if inject_ADMET_AI:
			admet_ai_result.append(ADMET_result)
			row = f"{drug}\t{row[1]}\t{prompt_to_include}\t{to_insert}\t{label}\t{row[5]}\t{likelihoods}\t{final_likelihood}\t{ADMET_result}\t{model}\n"
		else:
			row = f"{drug}\t{row[1]}\t{prompt_to_include}\t{to_insert}\t{label}\t{row[5]}\t{likelihoods}\t{final_likelihood}\t{model}\n"
	else:
		if row[3] == 'Call to GPT4 failed for final analysis':
			prediction = 1
		else:
			prediction = return_result(row[3])
		result = generate_image(drug)

		row[2].replace('Keep into account that (1) Now take into account the structure that I gave you and whether the structure can shield any potential toxic functional groups via steric shielding or other mechanisms and (2) Also consider that just because a chemical is similar to another it does not necessarily mean they will have the same effect. It suggests it but it is not strong evidence for this.', '')
		if dataset == 'DILI':
			prompt_to_use = f"{row[2]}. Also return the result as a decimal between 0 and 1 where 0 means it won't cause injury to the liver and 1 means it will cause injury to the liver. The closer to 0 the more confident you are it won't cause injury, the closer to 1 the more confident you are it will cause injury, the closer to 0.5 the more uncertain you are as to the effect of the drug on liver toxicity. Output this as Likelihood: <Decimal goes here>"
		elif dataset == 'hERG':
			prompt_to_use = f"{row[2]}. This is also some ground truth information that can help {info_to_inject}. Also return the result as a decimal between 0 and 1 where 0 means it won't block the hERG channel and 1 it will block the hERG channel. The closer to 0 the more confident you are it won't block hERG, the closer to 1 the more confident you are it will block hERG, the closer to 0.5 the more uncertain you are as to the effect of the drug on hERG blockage. Output this as Likelihood: <Decimal goes here>"
		elif dataset == 'Carcinogens_Lagunin':
			prompt_to_use = f"{row[2]}. Also return the result as a decimal between 0 and 1 where 0 means it is not a carcinogen and 1 it is a carcinogen. The closer to 0 the more confident you are it isn't a carcinogen, the closer to 1 the more confident you are it is a carcinogen, the closer to 0.5 the more uncertain you are as to whether the molecule is a carcinogen or not. Output this as Likelihood: <Decimal goes here>"
			if inject_ADMET_AI:
				ADMET_result = ADMET_AI.predict(drug)[dataset]
				prompt_to_use += "ADMET.AI is a ML model trained on carcinogenic molecules. Take into account the ADMET AI prediction for this molecule. If the ADMET AI prediction is favorable (closer to 0), it is less likely to be a carcinogen. If the ADMET AI prediction is unfavorable (closer to 1), it is more likely to be a carcinogen. If the ADMET AI prediction is uncertain (closer to 0.5), it is more uncertain as to whether the molecule is a carcinogen or not. You can decide whether to use the ADMET_AI prediction or not depending on how sure you are of your prediction."
				prompt_to_use += f"ADMET.AI had this prediction {ADMET_result}"

		likelihoods = []
		while len(likelihoods) < 2  or np.std(likelihoods) > 0.05:
			try:
				res = launch_gpt4(prompt_to_use, temp = 0.000000001, max_tokens = 1000, image = result)
				while 'sorry' in res or 'Sorry' in res:
					res = launch_gpt4(prompt_to_use, temp = 0.000000001, max_tokens = 1000, image = result)
					print("Repeating!")
				likelihoods.append(return_likelihood(res))
				# import pdb; pdb.set_trace()
				# final_likelihood = return_likelihood(res)
				if len(likelihoods) > 2:
					print(f"Need more clarifications! Currently have this {likelihoods}")
				
				if len(likelihoods) > 10:
					break
			except:
				print("We got an error we retried it!")
		
		final_likelihood = np.mean(likelihoods)
		final_result = return_result(res)

		gpt4_old_res.append(prediction)
		gpt4_new_res.append(return_result(res))
		gpt4_labels.append(label)
		gpt4_new_likelihood.append(final_likelihood)
		to_insert = res.replace('\n', '')
		prompt_to_include = prompt_to_use.replace('\n', '')
		if inject_ADMET_AI:
			row = f"{drug}\t{row[1]}\t{prompt_to_include}\t{to_insert}\t{label}\t{row[5]}\t{likelihoods}\t{final_likelihood}\t{ADMET_result}\tGPT4\n"
		else:			
			row = f"{drug}\t{row[1]}\t{prompt_to_include}\t{to_insert}\t{label}\t{row[5]}\t{likelihoods}\t{final_likelihood}\tGPT4\n"


	result_file.write(row)
	result_file.flush()

		# resultats = []
		# answers = []
		# while len(resultats) < 5:
		# 	print("Trying....")
		# 	res_answer = []

		# 	res = launch_gpt4(f"This was the prompt: {new_prompt}. This was your answer {row[3]}. With this consideration {add_on_one} does your answer change? Response with Favorable or Unfavorable as before in this format, 'Result: Favorable, Unfavorable'", temp = 0.000000001, max_tokens = 1000, image = result)
		# 	res_answer.append(res)
		# 	if "sorry" in res or "I'm unable to" in res or "I'm unable to provide a definitive answer" in res:
		# 		continue
		# 	first_add_on_result = return_result(res)
		# 	if first_add_on_result != prediction:
		# 		print("Swapped!")
		# 		resultats.append(first_add_on_result)
		# 		answers.append(res)
		# 		continue
			
		# 	res = launch_gpt4(f"This was the prompt: {new_prompt}. This was your answer {row[3]}. With this consideration {add_on_two} does your answer change? Response with Favorable or Unfavorable as before in this format, 'Result: Favorable, Unfavorable'", temp = 0.000000001, max_tokens = 1000, image = result)
		# 	if "sorry" in res or "I'm unable to" in res or "I'm unable to provide a definitive answer" in res:
		# 		continue
		# 	second_add_on_result = return_result(res)
		# 	if second_add_on_result != prediction:
		# 		print("Swapped!")
		# 		resultats.append(second_add_on_result)
		# 		answers.append(res)
		# 		continue
		# 	res_answer.append(res)

		# 	answers.append(res_answer)
		# 	resultats.append(1)

		# 	# #res = launch_gpt4(new_prompt, temp = 0.00000001, max_tokens = 1000, image = result)
		# 	# #final_resultat = return_result(res)
		# 	# new_resultat = launch_gpt4(f"{res}\t{add_on}", temp = 0.000000001, max_tokens = 1000, image = result)
		# 	# final_resultat = return_result(new_resultat)
		# 	# if "I'm sorry" not in res and "I'm sorry" not in new_resultat and "I'm unable to view the image" not in res and "I'm unable to view the image" not in new_resultat and "I'm unable to provide a definitive answer" not in res and "I'm unable to provide a definitive answer" not in new_resultat:
		# 	# #if final_resultat != -1 and "I'm unable to view the image" not in res and "I'm unable to provide a definitive answer" not in res:
		# 	# 	resultats.append(final_resultat)
		# 	# 	answers.append([res, new_resultat])
		# 	# 	#answers.append(res)
		# print(resultats)
		# import pdb; pdb.set_trace()
		# final_result = Counter(resultats).most_common(1)[0][0]
		# if len(set(resultats)) > 1:
		# 	# multi_prompt = 'Look at the following reasoning and determine which one is more likely to be correct. Specifically say if you think the molecule is favorable or unfavorable, reporting the result as Result: Favorable/Unfavorable. Here is the reasoning chains:\n'
		# 	# for answer in answers:
		# 	# 	multi_prompt += f"{answer[0]} Then {answer[1]}\n"
		# 	# final_result = launch_gpt4(multi_prompt, temp = 0.000000001, max_tokens = 1000, image = result)
		# 	import pdb; pdb.set_trace()
		# print(res)
		# print(label)


result_file.close()
#Calculate AUC and Accuracy
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

print(f"Old Accuracy: {accuracy_score(labels, old_res)}")
print(f"Old AUC: {roc_auc_score(labels, old_res)}")
print(f"New Accuracy: {accuracy_score(labels, new_res)}")
print(f"New AUC: {roc_auc_score(labels, new_res)}")
print(f"New Likelihood AUC: {roc_auc_score(labels, new_likelihood)}")

print(f"GPT4 Old Accuracy: {accuracy_score(gpt4_labels, gpt4_old_res)}")
print(f"GPT4 Old AUC: {roc_auc_score(gpt4_labels, gpt4_old_res)}")
print(f"GPT4 New Accuracy: {accuracy_score(gpt4_labels, gpt4_new_res)}")
print(f"GPT4 New AUC: {roc_auc_score(gpt4_labels, gpt4_new_res)}")
print(f"GPT4 New Likelihood AUC: {roc_auc_score(gpt4_labels, gpt4_new_likelihood)}")

if inject_ADMET_AI:
	print(f"ADMET AI Accuracy: {accuracy_score(labels, admet_ai_result)}")
	print(f"ADMET AI AUC: {roc_auc_score(labels, admet_ai_result)}")
#import pdb; pdb.set_trace()
# df = pd.read_csv(f'./{dataset}/results.tsv', sep = '\t', header = None)
# for index, row in tqdm(df.iterrows(), total=int(df.shape[0])):
# 	drug = row[0]
# 	label = row[4]
# 	model = row[6]
# 	result = generate_image(drug)
# 	if model == 'PQA':
# 		res = launch_gpt4(f"This was the prompt: {row[2]}. This was your response {row[3]}. Would you like to keep your answer or change it? Respond Yes for changing or No for staying the same.", temp = 0.000000001, max_tokens = 1000, image = result)
# 		print(res)
# 		if 'Yes' in res:
# 			import pdb; pdb.set_trace()