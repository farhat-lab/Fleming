try:
	import pqapi
except Exception as exc:
	pqapi = None
	print(f"Warning: pqapi is not available; PQA features disabled. ({exc})")
import pandas as pd
import re
import os
from tqdm import tqdm
import asyncio
import sys
from rdkit import Chem
from rdkit.Chem import Draw
import io
import base64
import yaml
import numpy as np
import fcntl

class PQA_Agent():
	"""
	Mixin class for PQA capabilities. 
	Expected to be mixed into a class that provides:
	- self.llm_infer
	- self.generate_image
	- self.role
	- self.task_description (used in run_substructure_analysis?)
	"""
	# def __init__(self):
	# 	pass

	async def launch_pqa(self, sub_structure, property_interest):
		if pqapi is None:
			raise RuntimeError("pqapi is not installed/configured; cannot run PQA queries.")
		print(f"Is the presence of a {sub_structure} substructure in a drug associated with improved tuberculosis inhibition compared to similar compounds without it?")
		response = await pqapi.agent_query(f"Is the presence of a {sub_structure} substructure in a drug associated with improved tuberculosis inhibition compared to similar compounds without it?")
		return response.answer.answer
	
	async def launch_pqa_queries(self, sub_structures, property_interest):
		tasks = [self.launch_pqa(sub_structure, property_interest) for sub_structure in sub_structures]
		results = await asyncio.gather(*tasks)
		return results
	
	def launch_pqa_queries_sync(self, sub_structures, property_interest):
		if pqapi is None:
			raise RuntimeError("pqapi is not installed/configured; cannot run PQA queries.")
		queries = []
		for sub_structure in sub_structures:
			queries.append(f"How does the presence of a {sub_structure} substructure influence {property_interest}, both favorably and unfavorably?")

		print(f"Launching pqa for {sub_structures}")
		success = False
		while not success:
			try:
				jobs = [pqapi.submit_agent_job(query=q) for q in queries]
				results = asyncio.run(pqapi.gather_pqa_results_via_polling([job['metadata']['query_id'] for job in jobs]))
				return [result['response']['answer']['answer'] for result in results]
			except Exception as e:
				print(f"FAILED PQA with this exception {e} trying again!")
			except Exception as e:
				print(f"FAILED PQA with this exception {e} trying again!")

	def return_molecule_substructures(self, smile_string):
		prompt_to_run = self.describe_molecule(smile_string)
		prompt_to_run = self.describe_molecule(smile_string)
		description = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": prompt_to_run}])
	
		prompt = f'''Provide a list of the substructures of this molecule in a list format separated by semicolons like: substructure1;substructure2;substructure3.
		Exclude any numbers from the subgroups. This is the molecule: {description}'''

		sub_structures = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": prompt}])

		sub_structures = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": prompt}])
		sub_structures = [i.rstrip().strip() for i in sub_structures.split(';')]
		sub_structures = [i.replace('group', '').rstrip().strip() for i in sub_structures]
		return description, sub_structures
	
	def collate_substructures(self, smile_string, description, potential_substructures, image):
		
		prompt = f'''Here is a molecule smile structure {smile_string} and its description: {description}.
		Here is a list of potential substructures that could be in the molecule: {potential_substructures}. 
		Using these and the provided image and output the substructures that are present in the molecule.
		Make sure you are not redundant in the outputted substructures. For example, benzene and benzene ring are the same thing.
		Make sure no substructures have numbers in them. For example a 8-methyl-1,2,3,4-tetrahydronaphthalene should be outputted as a methyl and tetrahydronaphthalen substructure.
		Or 5-methyl is just a methyl substructure.
		Output in a list format separated by commas like: substructure1,substructure2,substructure3.
		Just output the corrected list, nothing else.'''

		sub_structures = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": self.get_content(prompt, image)}])

		prompt = f'''Do any of these substructures have a number in them? {sub_structures}
		If so, please remove the number and output the substructure without the number.
		Output in a list format separated by commas like: substructure1,substructure2,substructure3.
		Just output the corrected list, nothing else.'''

		sub_structures = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": self.get_content(prompt, image)}])

		sub_structures = [i.rstrip().strip() for i in sub_structures.split(',')]
		sub_structures = [i for i in sub_structures if len(i) > 1]
		
		prompt = f'''These are a list of substructures in a molecule clean this list {sub_structures} by (1) getting rid of duplicate substructures, (2) getting rid of nonsense substructures, (3) collating sub-structures that are repetitive for example if there is substructure X and substructure Y and there is substructure X and Y, either keep each separate or combine bothGiven the list of chemical substructures {sub_structures}, clean and simplify the list by performing the following steps:
        (1) Remove duplicates.
        (2) Remove invalid or nonsensical substructures.
        (3) Remove overly broad terms.
        (4) Group structurally similar or overlapping substructures. For example, if both substructure X and substructure Y are listed separately, and there's also a combined substructure "X and Y", then either keep X and Y separately or combine them — whichever is clearer and avoids redundancy. Use the image to help with this determination, if two entries refer to the same structure in the molecule then one of them has to be removed.

        Return the cleaned list as a comma-separated string of distinct, meaningful substructures. No reasoning just the list. '''

		sub_structures = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": self.get_content(prompt, image)}])
		sub_structures = [i.strip() for i in sub_structures.split(',')]
		
		return sub_structures


	def get_sub_structure_info(self, sub_structures, property_interest):
		sub_structures_to_run = sub_structures
		sub_structure_to_pqa = {}
	def get_sub_structure_info(self, sub_structures, property_interest):
		sub_structures_to_run = sub_structures
		sub_structure_to_pqa = {}

		if len(sub_structures_to_run):
			#import pdb; pdb.set_trace()
			results = self.launch_pqa_queries_sync(sub_structures_to_run, property_interest)
			results = self.launch_pqa_queries_sync(sub_structures_to_run, property_interest)
		
			# results = await self.launch_pqa_queries(sub_structures_to_run, self.property_interest)
			for sub_structure, pqa_result in zip(sub_structures_to_run, results):
				pqa_result = pqa_result.replace("\n", " ")
				sub_structure_to_pqa[sub_structure] = pqa_result

		substructures_info = "\n".join([f"{sub_structure}: {sub_structure_to_pqa[sub_structure]}" for sub_structure in sub_structures])

		return substructures_info, sub_structure_to_pqa
	
	def return_result(self, response):
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
				print("THIS FAILED SO DEFAULTED TO 0")
				return 0
		return prediction
	
	def run_prompt(self, prompt_to_use, image_to_inject):
		likelihoods = []
		while len(likelihoods) < 2  or np.std(likelihoods) > 0.05:
			try:
				res = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": self.get_content(prompt_to_use, image_to_inject)}])
				# res = self.launch_gpt4(prompt_to_use, temp = 0.000000001, max_tokens = 1000, image = image_to_inject).choices[0].message.content
				while 'sorry' in res or 'Sorry' in res:
					# res = self.launch_gpt4(prompt_to_use, temp = 0.000000001, max_tokens = 1000, image = image_to_inject).choices[0].message.content
					res = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": self.get_content(prompt_to_use, image_to_inject)}])
					print("Repeating!")

				likelihoods.append(self.return_likelihood(res))

				if len(likelihoods) > 2:
					print(f"Need more clarifications! Currently have this {likelihoods}")
				
				if len(likelihoods) > 10:
					break
			except Exception as e:
				print(f"We got an error {e} are now retrying it!")
				#import pdb; pdb.set_trace()
				#x = self.return_result(res)
				#likelihoods = [x, x]

		final_result = self.return_result(res)
		final_likelihood = np.mean(likelihoods)

		return res, prompt_to_use, final_likelihood, likelihoods, final_result
	
	def get_question(self, question):
		to_send = f"""
		Given this question you want to ask: 
		
		Evaluate whether the addition of this substructure to this molecule influences its likelihood of Y considering both Favorable and Unfavorable effects.

		And the question you were given {question}

		What would you write in Y to ask this question. Return only the text you would write in Y.

		"""

		return self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": to_send}])
	
	def get_decimal_question(self, question):
		to_send = f"""
		Given this question you want to ask:

		Also return the result as a decimal between 0 and 1 where 0 means it is not X and 1 it is X. The closer to 0 the more confident you are it isn't X, the closer to 1 the more confident you are it is X, the closer to 0.5 the more uncertain you are as to whether the molecule is X or not. Output this as Likelihood: <Decimal goes here>"

		And the question you were given {question}

		What would you write in X to ask this question. Return only the text you would write in X.

		"""

		return self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": to_send}])
	
	def get_property_question(self, question):
		to_send = f"""
		Given this question you want to ask:

		How does the presence of this substructure influence X, both favorably and unfavorably?"

		And the question you were given {question}

		What would you write in X to ask this question. Return only the text you would write in X.

		"""

		return self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": to_send}])
	
	def get_general_question(self, question):
		to_send = f"""
		Given this question you want to answer: {question}

		Rewrite it as a broader question that could be used in a literature search, such that the answer would help you answer the original question.
		Return only the rewritten question.
		"""

		return self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": to_send}])

	def run_substructure_analysis(self, drug, description, sub_structure_mapping, image, info_to_inject, question, decimal_question):
		molecule_context = f"The molecule is the following: {drug} with the following description: {description}"
		results = []

		for molecule in tqdm(sub_structure_mapping):
			sub = f'{molecule}:{sub_structure_mapping[molecule]}'
			#new_prompt = molecule_context + 'This is information about ' + sub + f'From this information along with knowing its context in the molecule from description and the image does the addition of {molecule} in this molecule influence its liklihood of being {self.task_description}? If so does it influence it favorably or unfavorably? If not, you can say there is likely minimal to no influence.'
			new_prompt = molecule_context + 'This is information about ' + sub + f'Evaluate whether the addition of {molecule} to this molecule influences its likelihood of {question} considering both Favorable and Unfavorable effects. Then give a conclusion: Favorable, Unfavorable, or Minimal Influence.'
			new_prompt = molecule_context + 'This is information about ' + sub + f'Evaluate whether the addition of {molecule} to this molecule influences its likelihood of {question} considering both Favorable and Unfavorable effects. Then give a conclusion: Favorable, Unfavorable, or Minimal Influence.'

			try:
				new_res = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": self.get_content(new_prompt, image)}])
			except:
				print("The call to GPT4 failed for substructure analysis, trying now with no images")
				try:
					new_res = self.llm_infer([{"role": "system", "content": self.role}, {"role":"user", "content": new_prompt}])
				except:
					print("Call for GPT4 failed for substructure analysis")
					new_res = sub
			#import pdb; pdb.set_trace()
			results.append(new_res)
		
		final_prompt = f'''Ill provide you with a molecule, its description, and an image of the structure. Then I will provide you reasoning on the substructures and how they influence the potential of {question}. From this reasoning and the image say whether the molecule is likely to be or not to be {question}.'''
		final_prompt = f'''Ill provide you with a molecule, its description, and an image of the structure. Then I will provide you reasoning on the substructures and how they influence the potential of {question}. From this reasoning and the image say whether the molecule is likely to be or not to be {question}.'''
		final_prompt = final_prompt + molecule_context + '\t'.join(results) + 'Give the answer as Result: Favorable or Unfavorable. Provide reasoning.'
		final_prompt = final_prompt + f"This is also some ground truth information that can help {info_to_inject}. Remember that just one subunit being very favorable will not lead to favorable effect but rather it is the interplay of multiple favorable elements that leads to favorability. {decimal_question} Output this as Likelihood: <Decimal goes here>"
		final_prompt = final_prompt + f"This is also some ground truth information that can help {info_to_inject}. Remember that just one subunit being very favorable will not lead to favorable effect but rather it is the interplay of multiple favorable elements that leads to favorability. {decimal_question} Output this as Likelihood: <Decimal goes here>"
		return final_prompt
	
	def run_molecule(self, molecule, question):

		get_lit_info = self.get_general_lit_info(question)
		question_blurb = self.get_question(question)
		decimal_blurb = self.get_decimal_question(question)
		property_blurb = self.get_property_question(question)

		image = self.generate_image(molecule)
		
		#HERE YOU WANNA FIX LETS GO!!!! SAMPLE MULtiple substructures
		final_sub_structures = []
		for _ in range(4):
			description, sub_structures = self.return_molecule_substructures(molecule)

			while len(sub_structures) == 1:
				description, sub_structures = self.return_molecule_substructures(molecule)
				print("Restarting sub structures because only one was found")
			final_sub_structures.extend(sub_structures)
		
		sub_structures = self.collate_substructures(molecule, description, ','.join(list(set(final_sub_structures))), image)
		print(f"Substructures found: {sub_structures}")
		sub_structure_info, sub_structure_mapping = self.get_sub_structure_info(sub_structures, property_blurb)
		get_lit_info = self.get_general_lit_info(question)
		question_blurb = self.get_question(question)
		decimal_blurb = self.get_decimal_question(question)
		property_blurb = self.get_property_question(question)

		image = self.generate_image(molecule)
		
		#HERE YOU WANNA FIX LETS GO!!!! SAMPLE MULtiple substructures
		final_sub_structures = []
		for _ in range(4):
			description, sub_structures = self.return_molecule_substructures(molecule)

			while len(sub_structures) == 1:
				description, sub_structures = self.return_molecule_substructures(molecule)
				print("Restarting sub structures because only one was found")
			final_sub_structures.extend(sub_structures)
		
		sub_structures = self.collate_substructures(molecule, description, ','.join(list(set(final_sub_structures))), image)
		print(f"Substructures found: {sub_structures}")
		sub_structure_info, sub_structure_mapping = self.get_sub_structure_info(sub_structures, property_blurb)

		#Trying with PQA
		
		answer, final_prompt, likelihood, likelihoods, final_result = self.run_prompt(self.run_substructure_analysis(molecule, description, sub_structure_mapping, image, get_lit_info, question_blurb, decimal_blurb), image)
		return answer

	
	def encode_image(self, image):
		# Convert the image to a binary format using an in-memory bytes buffer
			buffer = io.BytesIO()
			image.save(buffer, format='JPEG')  # Ensure the format matches your image type, e.g., 'PNG' or 'JPEG'
			image_bytes = buffer.getvalue()

			# Encode the binary data to Base64
			image_base64 = base64.b64encode(image_bytes)

			# Convert Base64 bytes to a UTF-8 string
			image_base64_utf8 = image_base64.decode('utf-8')

			return image_base64_utf8

	def generate_image(self, smiles):
		molecule = Chem.MolFromSmiles(smiles)

		# Generate an image of the molecule
		image = Draw.MolToImage(molecule, size=(300, 300))
		return self.encode_image(image)
	
	def return_likelihood(self, response):
		prediction = response.replace('*','').split('Likelihood')[-1].split('\n')[0].split('Reasoning')[0].replace('*','').replace(':','').replace('(', '').strip().split(' ')[0]
		return float(prediction)
	
	def get_general_lit_info(self, question):
		question_to_ask = self.get_general_question(question)
		return 'Based on the context, there are several key structural features and mechanisms that contribute to liver toxicity in small molecules:\n\nStructural Features:\n\n1. High lipophilicity (logP ≥ 3) combined with high daily doses (≥100mg) significantly increases risk of severe drug-induced liver injury by facilitating drug uptake into hepatocytes and affecting interactions with mitochondrial membranes (chen2015druginducedliverinjury pages 7-10).\n\n2. Specific structural alerts associated with hepatotoxicity include:\n- Hydrazines and sulfonamide moieties\n- Arylacetic acid groups\n- Aniline structures\n- Halogen atoms bonded to sp3 carbon\n- Fused tricyclic structures (liu2015datadrivenidentificationof pages 5-6)\n\n3. Chlorobenzenes and naphthalene rings can be metabolically activated to form epoxides and quinones that are highly reactive (pizzo2015identificationofstructural pages 2-5).\n\nKey Mechanisms:\n\n1. Metabolic Activation and Reactive Metabolites:\n- Many compounds require bioactivation by liver enzymes (especially CYP450) to become toxic\n- Formation of reactive intermediates like epoxides and quinones\n- Creation of protein/DNA adducts through reactive metabolites (rusyn2021keycharacteristicsof pages 3-4, singh2016druginducedlivertoxicity pages 4-5)\n\n2. Oxidative Stress:\n- Generation of reactive oxygen species (ROS)\n- Depletion of cellular antioxidants like glutathione\n- Damage to cellular macromolecules and membranes (rusyn2021keycharacteristicsof pages 4-6)\n\n3. Mitochondrial Dysfunction:\n- Disruption of oxidative phosphorylation\n- Alteration of membrane permeability\n- Impact on ATP production\n- Triggering of cell death pathways (rusyn2021keycharacteristicsof pages 6-8)\n\n4. Transport and Metabolic Disruption:\n- Interference with bile acid transporters\n- Disruption of lipid metabolism\n- Impairment of phospholipid degradation\n- Inhibition of mitochondrial fatty acid oxidation (rusyn2021keycharacteristicsof pages 8-9)\n\n5. Cell Death Mechanisms:\n- Direct hepatotoxicity leading to apoptosis or necrosis\n- Immune-mediated responses involving hapten formation\n- Cholestatic injury through disruption of bile transport\n- Release of danger-associated molecular patterns triggering inflammation (mihajlovic2022mitochondriaasthe pages 6-7)\n\nThese mechanisms often interact and overlap, making it complex to predict liver toxicity. For example, acetaminophen demonstrates multiple mechanisms including metabolic activation to NAPQI, glutathione depletion, protein binding, and mitochondrial damage (iorga2017druginducedliverinjury pages 10-12). Modern drug development aims to avoid these structural features and mechanisms, though challenges remain in developing completely safe drugs (weaver2020managingthechallenge pages 20-24).'
		res = pqapi.agent_query(question_to_ask)
		return res.dict()['session']['answer']

	def __call__(self, molecule, question):
		return self.run_molecule(molecule, question)

if __name__ == '__main__':
	config_path = "/home/unix/yektefai/Fleming/FlemingDev/configs/fleming.yaml"
	pqa_agent = PQA_Agent(config_path)
	pqa_agent('CC1=C(C(=O)NC2=CN=CC=C2)SN=N1', 'Is this small molecule likely to be toxic to the liver?')

