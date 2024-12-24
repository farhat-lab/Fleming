import pqapi
import pandas as pd
import re
import os
from openai import AzureOpenAI
from molecular_describer import prompt_generator
from tdc.single_pred import Tox
from tqdm import tqdm
import asyncio
import sys
from admet_ai import ADMETModel
from rdkit import Chem
from rdkit.Chem import Draw
import io
import base64

# data = Tox(name = 'DILI')
# split = data.get_split()
pqapi_token = 'pqa-eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1za2FybGluc2tpQGZ1dHVyZWhvdXNlLm9yZyIsImV4cCI6MTc2MjQ3NjYwNC4xOTU2NzJ9.FaS6uy0G_StC8SLEBtEEf44Ms50TzIB0IuWf2jBqyEw'
azure_openai_api_key = 'f65f6d81f08947a488e4599b08110058'
azure_openai_endpoint = 'https://azure-ai-dev.hms.edu'
os.environ['PQA_API_TOKEN'] = pqapi_token

class PQA_Benchmark():
	def __init__(self, dataset_name, property_interest, task_description):
		self.dataset_name = dataset_name
		self.property_interest = property_interest
		self.data = Tox(name = dataset_name)
		self.split = self.data.get_split()
		self.test = self.split['test']
		self.task_description = task_description

		if not os.path.exists(f'./{dataset_name}'):
			os.mkdir(f'./{dataset_name}')
		
		if not os.path.exists(f'./{dataset_name}/sub_structure_mapping.tsv'):
			self.sub_structure_mapping = open(f'./{dataset_name}/sub_structure_mapping.tsv', 'w')
			self.sub_structure_to_pqa = {}
		else:
			self.sub_structure_mapping = open(f'./{dataset_name}/sub_structure_mapping.tsv', 'r')
			self.sub_structure_to_pqa = {}
			self.sub_structure_mapping.seek(0)
			for line in self.sub_structure_mapping:
				sub_structure, pqa_result = line.split('\t')
				self.sub_structure_to_pqa[sub_structure] = pqa_result
			
			self.sub_structure_mapping.close()
			self.sub_structure_mapping = open(f'./{dataset_name}/sub_structure_mapping.tsv', 'a')
		
		if not os.path.exists(f'./{dataset_name}/results.tsv'):
			self.results = open(f'./{dataset_name}/results.tsv', 'w')
			self.structures_analyzed = []
		else:
			self.results = open(f'./{dataset_name}/results.tsv', 'r')
			self.structures_analyzed = []
			self.results.seek(0)
			for line in self.results:
				data = line.split('\t')
				self.structures_analyzed.append(data[0])
			
			self.results.close()
			self.results = open(f'./{dataset_name}/results.tsv', 'a')
		
		self.admet_model = ADMETModel()
			
	async def launch_pqa(self, sub_structure, property_interest):
		#response = pqapi.agent_query("How does Pyridazine structure influence toxicity of a molecule as a drug?")
		print(f"How does having a {sub_structure} structure in a drug influence its {property_interest}?")
		response = await pqapi.agent_query(f"How does having a {sub_structure} structure in a drug influence its {property_interest}?")
		return response.answer.answer
	
	async def launch_pqa_queries(self, sub_structures, property_interest):
		tasks = [self.launch_pqa(sub_structure, property_interest) for sub_structure in sub_structures]
		results = await asyncio.gather(*tasks)
		return results
	
	def launch_pqa_queries_sync(self, sub_structures, property_interest):
		queries = []
		for sub_structure in sub_structures:
			#queries.append(f"How does having a {sub_structure} structure in a drug influence its {property_interest}?")
			queries.append(f"Can having a {sub_structure} structure in a drug cause {property_interest}?")
			#queries.append(f"Does having {sub_structure} in a drug cause {property_interest}?")

		print(f"Launching pqa for {sub_structures}")
		success = False
		while not success:
			try:
				jobs = [pqapi.submit_agent_job(query=q) for q in queries]
				results = asyncio.run(pqapi.gather_pqa_results_via_polling([job['metadata']['query_id'] for job in jobs]))
				return [result['response']['answer']['answer'] for result in results]
			except:
				print("FAILED PQA trying again!")

	def launch_gpt4(self, message, temp = 0.5, max_tokens = 250, image = None):
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
		
		while True:
		
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

			if "I'm sorry, I can't assist with that" in response.choices[0].message.content or "I'm unable to view the image" in response.choices[0].message.content or "I'm unable to provide a definitive answer" in response.choices[0].message.content: 
				print("Failed to generate response, trying again")
				continue
			else:
				return response

	def return_molecule_substructures(self, smile_string):
		prompt_to_run, iupac_success = prompt_generator(smile_string)
		description = self.launch_gpt4(prompt_to_run, temp=0.1).choices[0].message.content

		# prompt = ''' I'll provide you a description of a molecule, please give the substructures that is important to its chemical properties. 
		#                  Please only respond the substructures separating by commas like: substructure1, substructure2, ... . Please only keep the key words and removing words like group or moiety etc, 
		#                  Remove the positional notation information and make sure all are in in the lower case form.
		#                  Here is the molecule description ''' + description

		prompt = f'''Provide a list of the substructures of this molecule in a list format separated by semicolons like: substructure1;substructure2;substructure3.
		Exclude any numbers from the subgroups. This is the molecule: {description}'''
		
		sub_structures = self.launch_gpt4(prompt, temp=0.1).choices[0].message.content
		sub_structures = [i.rstrip().strip() for i in sub_structures.split(';')]
		sub_structures = [i.replace('group', '').rstrip().strip() for i in sub_structures]
		return description, sub_structures
	
	def get_sub_structure_info(self, sub_structures):
		sub_structures_to_run = []
		for sub_structure in sub_structures:
			if sub_structure not in self.sub_structure_to_pqa:
				sub_structures_to_run.append(sub_structure)

		if len(sub_structures_to_run):
			#import pdb; pdb.set_trace()
			results = self.launch_pqa_queries_sync(sub_structures_to_run, self.property_interest)
		
			# results = await self.launch_pqa_queries(sub_structures_to_run, self.property_interest)
			for sub_structure, pqa_result in zip(sub_structures_to_run, results):
				pqa_result = pqa_result.replace("\n", " ")
				self.sub_structure_to_pqa[sub_structure] = pqa_result
				self.sub_structure_mapping.write(f'{sub_structure}\t{pqa_result}\n')
				self.sub_structure_mapping.flush()

		substructures_info = "\n".join([f"{sub_structure}: {self.sub_structure_to_pqa[sub_structure]}" for sub_structure in sub_structures])
		substructure_mapping = {}
		for sub_structure in sub_structures:
			substructure_mapping[sub_structure] = self.sub_structure_to_pqa[sub_structure]

		return substructures_info, substructure_mapping
	
	def synthesize_get_output(self, drug, description, substructures_info = None, image = None):
		if substructures_info is not None:
			final_prompt = f"""
			
			I’ll provide you with a molecule and its description. Then I will provide you information from literature on each substructure and how they influence the toxicity of the molecule. There might be irrelavent information regarding the substructure included in the description, so feel free to ignore any information that you think is irrelevant. A lot of this information is context dependent use the molecular description to place these substructures into context and use that context along with the retrieved text to make a conclusion.
			The molecule is the following: {drug} with the following description: {description}
			Here are the substructures along with information from literature: {substructures_info}
			Based on this information would this molecule have favorable or unfavorable {self.property_interest}. At the end of your output please just say Result: Unfavorable or Favorable depending on what you decide.
			
			"""
		else:
			final_prompt = f"""
			
			I’ll provide you with a molecule, its description, and image. 
			The molecule is the following: {drug} with the following description: {description}
			Based on this information would this molecule have favorable or unfavorable {self.property_interest}. At the end of your output please just say Result: Unfavorable or Favorable depending on what you decide.
			
			"""
		try:
			return self.launch_gpt4(final_prompt, temp=0.1, max_tokens = 1000, image = image).choices[0].message.content, final_prompt
		except:
			print("The call to GPT4 failed for final analysis, trying now with no images")
			try:
				return self.launch_gpt4(final_prompt, temp=0.1, max_tokens = 1000).choices[0].message.content, final_prompt
			except:
				print("Call for GPT4 failed for final analysis")
				return "Call to GPT4 failed for final analysis", final_prompt
	
	def run_substructure_analysis(self, drug, description, sub_structure_mapping, image):
		molecule_context = f"The molecule is the following: {drug} with the following description: {description}"

		results = []

		for molecule in tqdm(sub_structure_mapping):
			sub = f'{molecule}:{sub_structure_mapping[molecule]}'
			new_prompt = molecule_context + 'This is information about ' + sub + f'From this information along with knowing its context in the molecule from description and the image does the addition of {molecule} in this molecule influence its liklihood of being {self.task_description}? If so does it influence it favorably or unfavorably? If not, you can say there is likely minimal to no influence..'
			try:
				new_res = self.launch_gpt4(new_prompt, temp=0.1, max_tokens=1000, image = image).choices[0].message.content
			except:
				print("The call to GPT4 failed for substructure analysis, trying now with no images")
				try:
					new_res = self.launch_gpt4(new_prompt, temp=0.1, max_tokens=1000).choices[0].message.content
				except:
					print("Call for GPT4 failed for substructure analysis")
					new_res = sub
			results.append(new_res)
		
		final_prompt = f'''Ill provide you with a molecule, its description, and an image of the structure. Then I will provide you reasoning on the substructures and how they influence the potential for this drug to be {self.task_description}. From this reasoning and the image say whether the molecule is likely to be or not to be {self.task_description}.'''
		final_prompt = final_prompt + molecule_context + '\t'.join(results) + 'Give the answer as Result: Favorable or Unfavorable. Provide reasoning.'
		try:
			final_result = self.launch_gpt4(final_prompt, temp=0.1, max_tokens=1000, image = image).choices[0].message.content
		except:
			print("The call to GPT4 failed for final analysis, trying now with no images")
			try:
				final_result = self.launch_gpt4(final_prompt, temp=0.1, max_tokens=1000).choices[0].message.content
			except:
				print("Call for GPT4 failed for final analysis")
				final_result = "Call to GPT4 failed for final analysis"
		return final_result, final_prompt
	
	def run_molecules_pqa(self, range_run = None):
		
		if range_run is not None:
			to_run = self.test.iloc[range_run[0]:range_run[1]]
		else:
			to_run = self.test
		
		for index, row in tqdm(to_run.iterrows(), total=to_run.shape[0]):
			if row['Drug'] in self.structures_analyzed:
				print("Found in analyzed structures")
				continue
			
			image = self.generate_image(row['Drug'])

			description, sub_structures = self.return_molecule_substructures(row['Drug'])
			#import pdb; pdb.set_trace()
			sub_structure_info, sub_structure_mapping = self.get_sub_structure_info(sub_structures)

			#Trying with PQA

			answer, final_prompt = self.run_substructure_analysis(row['Drug'], description, sub_structure_mapping, image)
			# answer, final_prompt = self.synthesize_get_output(row['Drug'], description, sub_structure_info)
			print(answer)
			answer = answer.replace('\n', '')
			final_prompt = final_prompt.replace('\n', '').replace('\t', '')
			self.results.write(f"{row['Drug']}\t{description}\t{final_prompt}\t{answer}\t{row['Y']}\t{self.admet_model.predict(row['Drug'])[self.dataset_name]}\tPQA\n")
			self.results.flush()

			#Trying just GPT4
			answer, final_prompt = self.synthesize_get_output(row['Drug'], description,None, image)
			print(answer)
			answer = answer.replace('\n', '')
			final_prompt = final_prompt.replace('\n', '').replace('\t', '')
			self.results.write(f"{row['Drug']}\t{description}\t{final_prompt}\t{answer}\t{row['Y']}\t{self.admet_model.predict(row['Drug'])[self.dataset_name]}\tGPT4\n")
			self.results.flush()
	
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

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print("Usage: python run_analysis.py <dataset_name> <property_interest>")
		sys.exit(1)
	
	dataset_name = sys.argv[1]
	property_interest = sys.argv[2]
	task_description = sys.argv[3]

	#benchmark = PQA_Benchmark('DILI', 'toxicity to the liver')
	#benchmark = PQA_Benchmark('Carcinogens_Lagunin', 'ability to cause cancer or be a carcinogen')
	benchmark = PQA_Benchmark(dataset_name, property_interest, task_description)
	benchmark.run_molecules_pqa()

	# test_molecule = split['test']['Drug'].values[0]
	# result = return_molecule_substructures(test_molecule)
	# print(result)
	# import pdb; pdb.set_trace()
	# pqa_result = launch_pqa(result[0], 'toxicity to the liver')
	# import pdb; pdb.set_trace()

