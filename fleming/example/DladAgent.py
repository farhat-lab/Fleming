import os
from typing import List

from DladActions import InhibitionPrediction, PharmacokineticsPrediction, MolecularOptimization, MoleculeGeneration, DescribeMolecule, SearchLiterature, RationaleExtraction

from agentlite.actions import BaseAction, FinishAct, ThinkAct
from agentlite.actions.InnerActions import INNER_ACT_KEY
from agentlite.agents import ABCAgent, BaseAgent
from agentlite.commons import AgentAct, TaskPackage
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.terminal_logger import AgentLogger
from agentlite.agents.agent_utils import AGENT_CALL_ARG_KEY

from agentlite.agents import ManagerAgent

# set PROMPT_DEBUG_FLAG to True to see the debug info
agent_logger = AgentLogger(PROMPT_DEBUG_FLAG=False)

class InhibitionAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        actions: List[BaseAction] = [InhibitionPrediction()],
        **kwargs
    ):
        
        name = "Inhibition_Prediction_Agent"
        role = "You'll be provided a SMILES structure, and please give the prediction to its M.tb inhibition using the InhibitionPrediction tool you have."
        
        super().__init__(
            name = name,
            role = role,
            llm = llm,
            actions = actions,
            logger = agent_logger
        )

        self.__build_examples__()

    def __build_examples__(self):
        ### Sample 1
        # task
        task = "Is CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 inhibitory against M.TB?"

        # 1. think action and obs
        thought = "CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 is the smiles structure of the molecule, I should first use M.tb_Inhibition_Prediction to predict the inhibitory effect of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 against M.tb"
        act_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_1 = ""

        # 2. api call action and obs
        act_params = {'smiles': 'CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1'}
        act_2 = AgentAct(name=InhibitionPrediction().action_name,
                         params=act_params)
        obs_2 = "(0.8, 0.2)"

        # 3. think action and obs
        answer = "The prediced inhibition of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 is 0.8, which is above the threshold 0.7, usually indicates a potential inhibitory effect of this molecule against M.tb."
        act_3 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer})
        obs_3 = "Task Completed."

        task_pack = TaskPackage(instruction=task)
        act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3)]
        self.add_example(task=task_pack, action_chain=act_obs)

class PharmacokineticsAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        actions: List[BaseAction] = [PharmacokineticsPrediction()],
        **kwargs
    ):
        
        name = "Pharmacokinetics_Prediction_Agent"
        role = "You'll be provided a SMILES structure, and please give the prediction to its Pharmacokinetics Properties using the PharmacokineticsPrediction tool you have."
        
        super().__init__(
            name = name,
            role = role,
            llm = llm,
            actions = actions,
            logger = agent_logger
        )

        self.__build_examples__()

    def __build_examples__(self):
        ### Sample 1
        # task
        task = "How's the pharmacokinetic properties of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 ?"

        thought = "To get the parmacokinetic properties of the molecule, I should continue and test for its pharmacokinetic properties using the PharmacokineticsPrediction tool."
        act_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_1 = "OK"

        # 5. api call action and obs
        act_params = {'smiles': 'CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1'}
        act_2 = AgentAct(name=PharmacokineticsPrediction().action_name,
                         params=act_params)
        obs_2 = "{'SMILES':'CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1', 'pass_AMES': 1, 'pass_BBB_Martins': 0, 'pass_Bioavailability_Ma': 1, 'pass_LD50_Zhu': 0}"

        # 3. think action and obs
        thought = "From the output dictionary, the SMILES matches the molecule we are testing. To read the output, I should read each key in the dictionary and its corresponding value, value 1 means the molecule pass that test, while value 0 means the molecule fails that test. \
                   I should also calculate the sum of the values to see how many tasks the model failed out of the 4 properties. \
                   I should be careful not missing any failed test and list the names here."
        act_3 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_3 = "OK"
                     
        answer = " In this example, \
                   I can see the model passes most tests, but fails the Blood Brain Barrier test and LD50 Test, so the molecules passes 2 out of the 4 tests, and failed on 2 out of the 4 tests "
        act_4 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer})
        obs_4 = "Task Completed."

        task_pack = TaskPackage(instruction=task)
        act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4)]
        self.add_example(task=task_pack, action_chain=act_obs)

class MolecularOptimizationAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        actions: List[BaseAction] = [MolecularOptimization()],
        **kwargs
    ):
        
        name = "Molecular_Optimization_Agent"
        role = "You'll be provided a SMILES structure, please use the MolecularOptimization Tool to optimize its pharmacokinetic properties."
        
        super().__init__(
            name = name,
            role = role,
            llm = llm,
            actions = actions,
            logger = agent_logger
        )

        self.__build_examples__()
    
    def __build_examples__(self):
        ### Sample 1
        # task
        task = "Please optimize this molecule CCCCC(CC)CCCN(C)CCC(=O)c1ccc(CC)cc1 regarding its pharmacokinetic properties while maintaining its inhibition property."

        # 1. think action and obs
        thought = "CCCCC(CC)CCCN(C)CCC(=O)c1ccc(CC)cc1 is the smiles structure of the molecule, I should put it into the molecular optimization tool to optimize it."
        act_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_1 = ""

        # 2. api call action and obs
        act_params = {'smiles': 'CCCCC(CC)CCCN(C)CCC(=O)c1ccc(CC)cc1'}
        act_2 = AgentAct(name=MolecularOptimization().action_name,
                         params=act_params)
        obs_2 = "{'SMILES': 'CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1', 'improved_admet_score_by': 2.0}"

        # 3. think action and obs
        thought = "The molecular optimization tool modified CCCCC(CC)CCCN(C)CCC(=O)c1ccc(CC)cc1 into CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1, which improved the admet score by 2.0."
        act_3 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_3 = "OK"

        # 4. finish action
        answer = "After Molecular Optimization,CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1 turns into CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1, which has a better pharmacokinetics property compare to the seed molecule."
        act_4 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer})
        obs_4 = "Task Completed."

        task_pack = TaskPackage(instruction=task)
        act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4)]
        self.add_example(task=task_pack, action_chain=act_obs)

        ### Sample 2
        # task
        task = "Please optimize this molecule CCCCC(CC)CCCN(C)CCC(=O)c1ccc(CC)cc1 regarding its pharmacokinetic properties while maintaining its inhibition property."

        # 1. think action and obs
        thought = "CCCCC(CC)CCCN(C)CCC(=O)c1ccc(CC)cc1 is the smiles structure of the molecule, I should put it into the molecular optimization tool to optimize it."
        act_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_1 = ""

        # 2. api call action and obs
        act_params = {'smiles': 'CCCCC(CC)CCCN(C)CCC(=O)c1ccc(CC)cc1'}
        act_2 = AgentAct(name=MolecularOptimization().action_name,
                         params=act_params)
        obs_2 = "{'SMILES': 'CCCCC(CC)CCCN(C)CCC(=O)c1ccc(CC)cc1', 'improved_admet_score_by': 0.0}"

        # 3. think action and obs
        thought = "The molecular optimization tool didn't make modification to the original molecule, which means no neighboring molecules with better pharmacokinetics properties are found."
        act_3 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_3 = "OK"

        # 4. finish action
        answer = "Couldn't find a molecular neighbor with better pharmacokinetics propertyies for CCCCC(CC)CCCN(C)CCC(=O)c1ccc(CC)cc1."
        act_4 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer})
        obs_4 = "Task Completed."

        task_pack = TaskPackage(instruction=task)
        act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4)]
        self.add_example(task=task_pack, action_chain=act_obs)

class MoleculeGenerationAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        actions: List[BaseAction] = [MoleculeGeneration()],
        **kwargs
    ):
        
        name = "Molecular_Generation_Agent"
        role = "You can generate M.tb inhibitory molecules from the MoleculeGeneration Tool."
        
        super().__init__(
            name = name,
            role = role,
            llm = llm,
            actions = actions,
            logger = agent_logger
        )

        self.__build_examples__()

    def __build_examples__(self):
        ### Sample 1
        # task
        task = "Please give me a molecule with potential M.tb inhibition."

        thought = "To get a molecule with M.tb inhibition from scratch, I should use the MoleculeGeneration Tool."
        act_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_1 = "OK"

        # api call action and obs
        act_2 = AgentAct(name=PharmacokineticsPrediction().action_name,
                         params={})
        obs_2 = "CCCN(c1ccc(CC)cc1)S(=O)(=O)c1nc(Cl)cs1"

        # Response    
        answer = "CCCN(c1ccc(CC)cc1)S(=O)(=O)c1nc(Cl)cs1"
        act_3 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer})
        obs_3 = "Task Completed."

        task_pack = TaskPackage(instruction=task)
        act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3)]
        self.add_example(task=task_pack, action_chain=act_obs)        

class MoleculeDescriber(BaseAgent): 
    def __init__(
        self,
        llm: BaseLLM,
        actions: List[BaseAction] = [DescribeMolecule()],
        **kwargs
    ):
        
        name = "Molecular_Describer_Agent"
        role = "You can generate molecular descriptions for a molecule using DescribeMolecule Tool."
        
        super().__init__(
            name = name,
            role = role,
            llm = llm,
            actions = actions,
            logger = agent_logger
        )

        self.__build_examples__()

    def __build_examples__(self):
        ### Sample 1
        # task
        task = "Please give me a molecular description for CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1"

        thought = "To get a description for CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 I should use the DescribeMolecule Tool."
        act_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_1 = "OK"

        # api call action and obs
        act_2 = AgentAct(name=DescribeMolecule().action_name,
                         params={'smiles': 'CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1'})
        obs_2 = "This molecule features a thienopyridine core with a phenyl substitution, and it includes a thioether linkage within the heteroaromatic system. Additionally, it contains a piperazine moiety connected to the core structure."

        # Response    
        answer = "This molecule features a thienopyridine core with a phenyl substitution, and it includes a thioether linkage within the heteroaromatic system. Additionally, it contains a piperazine moiety connected to the core structure."
        act_3 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer})
        obs_3 = "Task Completed."

        task_pack = TaskPackage(instruction=task)
        act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3)]
        self.add_example(task=task_pack, action_chain=act_obs)        

class LiteratureAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        actions: List[BaseAction] = [SearchLiterature()],
        **kwargs
    ):
        
        name = "Literature_Agent"
        role = "You can search literatures for the provided molecular description using SearchLiterature Tool."
        
        super().__init__(
            name = name,
            role = role,
            llm = llm,
            actions = actions,
            logger = agent_logger
        )

        self.__build_examples__()

    def __build_examples__(self):
        ### Sample 1
        # task
        task = '''Please evaluate the molecule with the following descriptors on whether it has potential to become a novel TB antibiotic. 
                  Molecule Description: This molecule is a diphenylmethane derivative with two aromatic benzene rings. One benzene ring has a phenolic hydroxyl group and a bromine substituent, while the two rings are connected through an alkyl chain with a tertiary amine group.'''

        thought = "To find literature support and evaluate this molecule, I should use the SearchLiterature Tool. \
            To summarize the output, I should keep the details and all the references to let the user know where the information is from"
        act_1 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_1 = "OK"

        # api call action and obs
        act_2 = AgentAct(name=SearchLiterature().action_name,
                         params={'description': task})
        obs_2 = '''The bioavailability of a compound with a dihydropyridine core, dimethyl and diethyl ester substitutions, a phenyl group, and an extended enone system with an isobutyl ester side chain can be influenced by several structural features. The dihydropyridine (DHP) core is crucial for binding interactions with biological targets, such as the multidrug-resistance transporter MDR1, and modifications to this core can significantly impact bioavailability (steiger2014synthesisandstudy pages 40-43). Ester substitutions at the C-3 and C-5 positions of the DHP core, such as dimethyl and diethyl esters, can affect the compound's conformation and, consequently, its solubility and metabolic stability, which are key factors in bioavailability (steiger2014synthesisandstudy pages 6-8).

                    The presence of a phenyl group and an extended enone system may enhance π-stacking interactions, which can improve binding affinity and potentially influence bioavailability positively (steiger2014synthesisandstudy pages 80-86). However, the inclusion of an isobutyl ester side chain could introduce steric hindrance, potentially affecting the compound's ability to interact with its target and thus its bioavailability (steiger2014synthesisandstudy pages 190-193). Overall, the structural modifications and substitutions in this compound are likely to have a complex impact on its bioavailability, balancing between enhanced binding interactions and potential steric challenges.

                    References

                    1. (steiger2014synthesisandstudy pages 190-193): SA Steiger. Synthesis and study of heterocyclic, dimeric and chiral ligands for the multidrug-resistance transporter. Unknown journal, 2014.

                    2. (steiger2014synthesisandstudy pages 40-43): SA Steiger. Synthesis and study of heterocyclic, dimeric and chiral ligands for the multidrug-resistance transporter. Unknown journal, 2014.

                    3. (steiger2014synthesisandstudy pages 6-8): SA Steiger. Synthesis and study of heterocyclic, dimeric and chiral ligands for the multidrug-resistance transporter. Unknown journal, 2014.

                    4. (steiger2014synthesisandstudy pages 80-86): SA Steiger. Synthesis and study of heterocyclic, dimeric and chiral ligands for the multidrug-resistance transporter. Unknown journal, 2014.'''

        thought = "I already have the output from the SearchLiterature tool, I should keep all the details and all the references to let the user know where the information is from without summarizing it."
        act_3 = AgentAct(name=ThinkAct.action_name, params={INNER_ACT_KEY: thought})
        obs_3 = "OK"

        # Response    
        answer = '''The bioavailability of a compound with a dihydropyridine core, dimethyl and diethyl ester substitutions, a phenyl group, and an extended enone system with an isobutyl ester side chain can be influenced by several structural features. The dihydropyridine (DHP) core is crucial for binding interactions with biological targets, such as the multidrug-resistance transporter MDR1, and modifications to this core can significantly impact bioavailability (steiger2014synthesisandstudy pages 40-43). Ester substitutions at the C-3 and C-5 positions of the DHP core, such as dimethyl and diethyl esters, can affect the compound's conformation and, consequently, its solubility and metabolic stability, which are key factors in bioavailability (steiger2014synthesisandstudy pages 6-8).
                    The presence of a phenyl group and an extended enone system may enhance π-stacking interactions, which can improve binding affinity and potentially influence bioavailability positively (steiger2014synthesisandstudy pages 80-86). However, the inclusion of an isobutyl ester side chain could introduce steric hindrance, potentially affecting the compound's ability to interact with its target and thus its bioavailability (steiger2014synthesisandstudy pages 190-193). Overall, the structural modifications and substitutions in this compound are likely to have a complex impact on its bioavailability, balancing between enhanced binding interactions and potential steric challenges'''
        
        act_4 = AgentAct(name=FinishAct.action_name, params={INNER_ACT_KEY: answer})
        obs_4 = "Task Completed."

        task_pack = TaskPackage(instruction=task)
        act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4)]
        self.add_example(task=task_pack, action_chain=act_obs)       

class RationaleExtractionAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseLLM,
        actions: List[BaseAction] = [RationaleExtraction()],
        **kwargs
    ):
        
        name = "Rationale_Extraction_Agent"
        role = "You can use the RationaleExtraction tool to extract the substructure of the molecule that the inhibition prediction model thinks leading to its inhibitory property."
        
        super().__init__(
            name = name,
            role = role,
            llm = llm,
            actions = actions,
            logger = agent_logger
        )

        self.__build_examples__()

    def __build_example__():
        return 0

class ManagerAgent(ManagerAgent):
    def __init__(
        self,
        llm: BaseLLM,
        **kwargs
    ):
        
        name = "Mtb_Antibiotics_Discovery_Manager_Agent"
        role = "You are controlling InhibitionAgent, PharmacokineticsAgent, MolecularOptimizationAgent, and MoleculeGeneration Agent to assist Antibiotic Discovery for Mtb."

        self.inhibition_agent = InhibitionAgent(llm=llm)
        self.pharmacokinetics_agent = PharmacokineticsAgent(llm=llm)
        self.mo_agent = MolecularOptimizationAgent(llm=llm)
        self.generation_agent = MoleculeGenerationAgent(llm=llm)
        self.molecule_describer = MoleculeDescriber(llm=llm)
        self.literature_searcher = LiteratureAgent(llm=llm)

        team = [self.inhibition_agent,
                self.pharmacokinetics_agent,
                self.mo_agent,
                self.generation_agent,
                self.molecule_describer,
                self.literature_searcher,]
                
        super().__init__(
            name = name,
            role = role,
            llm = llm,
            TeamAgents=team,
            logger = agent_logger
        )

        self.__build_examples__()
    
    def __build_examples__(self):

        ## Example 1
        exp_task = "What is the potential of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 as a Mtb antibiotics?"
        exp_task_pack = TaskPackage(instruction=exp_task)

        act_1 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""I first need to ask {self.inhibition_agent.name} to see if the molecule is inhibitory against Mtb using the user provided threshold, \
                                   if not provided, I should use 0.7 as the default threshold. \
                                   If is inhibitory, I should send the SMILES of the molecules to {self.pharmacokinetics_agent.name} to test for its pharmacokinetics property."""
            },
        )
        obs_1 = "OK"

        act_2 = AgentAct(
            name=self.inhibition_agent.name,
            params={AGENT_CALL_ARG_KEY: "Is CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 inhibitory against Mtb using 0.7 inhibition threshold?" }
        )
        obs_2 = "The prediced inhibition of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 is 0.2, which is below the threshold 0.7, usually indicates a non-inhibitory effect of this molecule against M.tb."
        
        act_3 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""I have already know that CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 is non-inhibitory. there's no need for me to undergo further tests."""
            },
        )
        obs_3 = "OK"

        act_4 = AgentAct(name=FinishAct.action_name, 
                         params={INNER_ACT_KEY: "CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 doesn't have potential inhibition again Mtb."})
        obs_4 = "Task Completed."

        exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4)]
        self.add_example(task=exp_task_pack, action_chain=exp_act_obs)
        
        ## Example 2
        exp_task = "What is the potential of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 as a Mtb antibiotics ?"
        exp_task_pack = TaskPackage(instruction=exp_task)

        act_1 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""I first need to ask {self.inhibition_agent.name} to see if the molecule is inhibitory against Mtb, \
                                   If yes, I should send the SMILES of the molecules to {self.pharmacokinetics_agent.name} to test for its pharmacokinetics property."""
            },
        )
        obs_1 = "OK"

        act_2 = AgentAct(
            name=self.inhibition_agent.name,
            params={AGENT_CALL_ARG_KEY: "Is CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 inhibitory against Mtb ?" }
        )
        obs_2 = "The prediced inhibition of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 is 0.8, which is above the threshold 0.7, usually indicates a potential inhibitory effect of this molecule against M.tb."
        
        act_3 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""I have already know that CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 is inhibitory. I now should ask {self.pharmacokinetics_agent.name} to test for its pharmacokinetics properties."""
            },
        )
        obs_3 = "OK"

        act_4 = AgentAct(
            name=self.pharmacokinetics_agent.name,
            params={AGENT_CALL_ARG_KEY: "How's the pharmacokinetic properties of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1"}
        )
        obs_4 = """In this example, \
                   I can see the model passes most tests, but fails the Blood Brain Barrier test and LD50 Test, so the molecules passes 2 out of the 4 tests, and failed on 2 out of the 4 tests, I should list all the failed pharmacokinetics tasks to the user for their information"""

        act_5 = AgentAct(name=FinishAct.action_name, 
                         params={INNER_ACT_KEY: "CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 has potential inhibition to Mtb, but failed on Blood Brain Barrier test and LD50 Pharmacokinectic Properties Test. To sum up, it's still a potential TB antibiotics."})
        obs_5 = "Task Completed."

        exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4), (act_5, obs_5)]
        self.add_example(task=exp_task_pack, action_chain=exp_act_obs)

        ## Example 3
        exp_task = "Please optimize this molecule CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 regarding its pharmacokinetic properties while maintaining its inhibition property."
        exp_task_pack = TaskPackage(instruction=exp_task)

        act_1 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""I have CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 as my seed molecule and need to ask {self.mo_agent.name} to optimize this molecule for me, \
                                   if {self.mo_agent.name} can output a better molecules, I should ask {self.mo_agent.name} to optimize the new output molecule iteratively \
                                   until {self.mo_agent.name} couldn't give a better molecule."""
            },
        )
        obs_1 = "OK"

        act_2 = AgentAct(
            name=self.mo_agent.name,
            params={AGENT_CALL_ARG_KEY: "Please optimize this molecule CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 regarding its pharmacokinetic properties while maintaining its inhibition property." }
        )
        obs_2 =  "After Molecular Optimization,CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 turns into CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1, which has a better pharmacokinetics property compare to the seed molecule."
        
        act_3 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""The output molecule CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1 has better pharmacokinetics property than the seed molecule CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1, I should call {self.mo_agent.name} again using CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1 as the seed molecule. """
            },
        )
        obs_3 = "OK"

        act_4 = AgentAct(
            name=self.mo_agent.name,
            params={AGENT_CALL_ARG_KEY: "Please optimize this molecule CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1 regarding its pharmacokinetic properties while maintaining its inhibition property." }
        )
        obs_4 =  "Couldn't find a molecular neighbor with better pharmacokinetics propertyies than CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1"

        act_5 = AgentAct(name=FinishAct.action_name, 
                         params={INNER_ACT_KEY: "After 2 iteration of molecular optimization, CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 can be modified to CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1, which has better pharmacokinetic properties."})
        obs_5 = "Task Completed."

        exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3), (act_4, obs_4), (act_5, obs_5)]
        self.add_example(task=exp_task_pack, action_chain=exp_act_obs)

        ## Example 4
        exp_task = "Please generate a molecule that has potential for M.tb inhibition."
        exp_task_pack = TaskPackage(instruction=exp_task)

        act_1 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""To generate a molecule from scratch that has potential M.tb inhibition, I should use {self.generation_agent.name} to generate a molecule."""
            },
        )
        obs_1 = "OK"

        act_2 = AgentAct(
            name=self.generation_agent.name,
            params={AGENT_CALL_ARG_KEY: "Please give me a molecule with potential M.tb inhibition." }
        )
        obs_2 =  "CC1CC(C)CN(CCCNc2ccc(CC[CH]OCC3CCCN3)cc2)C1"

        act_3 = AgentAct(name=FinishAct.action_name, 
                         params={INNER_ACT_KEY: "CC1CC(C)CN(CCCNc2ccc(CC[CH]OCC3CCCN3)cc2)C1 is the molecule generated that has potential Mtb inhibition."})
        obs_3 = "Task Completed."

        exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3)]
        self.add_example(task=exp_task_pack, action_chain=exp_act_obs)

        ## Example 5
        exp_task = "Please do a literature search based on only existing medicincal chemistry knowledge regarding M.tb inhibition for the molecule CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1."
        exp_task_pack = TaskPackage(instruction=exp_task)

        act_1 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""In order to obtain existing medicinal chemistry knowledge, I have to first use {self.molecule_describer.name} to generate descriptions of this molecule CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1, and then use {self.literature_searcher.name} to search for knowledge."""
            },
        )
        obs_1 = "OK"

        act_2 = AgentAct(
            name=self.molecule_describer.name,
            params={AGENT_CALL_ARG_KEY: "Please give me a molecular description for CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1." }
        )
        obs_2 =  "This molecule is a diphenylmethane derivative with two aromatic benzene rings. One benzene ring has a phenolic hydroxyl group and a bromine substituent, while the two rings are connected through an alkyl chain with a tertiary amine group."

        act_3 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""Now I get the description of the molecule, I should pass it to
                                {self.literature_searcher.name} to get knowledge about its potential to become an antibiotic for M.tb."""
            },
        )
        obs_3 = "OK"
  
        act_4 = AgentAct(
            name=self.literature_searcher.name,
            params={AGENT_CALL_ARG_KEY: "Please evaluate the molecule with the following descriptors on whether it has potential to become a novel TB antibiotic answer in detail about what might make it favorable, and what might cause it unfavorable. Molecule Description: This molecule is a diphenylmethane derivative with two aromatic benzene rings. One benzene ring has a phenolic hydroxyl group and a bromine substituent, while the two rings are connected through an alkyl chain with a tertiary amine group." }
        )
        obs_4 =  '''The molecule described as a diphenylmethane derivative with two aromatic benzene rings, one containing a phenolic hydroxyl group and a bromine substituent, and connected through an alkyl chain with a tertiary amine group, shows potential as a novel tuberculosis (TB) antibiotic. The structural elements of this molecule are consistent with those evaluated in studies aiming to modify the B-ring of Bedaquiline, a known TB drug, to improve its pharmacological properties. The presence of a phenolic hydroxyl group and a bromine substituent on the benzene ring could enhance the molecule's activity by affecting its lipophilicity and inhibitory concentrations, similar to the promising results observed with other Bedaquiline analogues (Choi2017 pages 5-6). These structural modifications can potentially yield favorable lipophilicity, which is crucial for maintaining antibacterial potency while reducing adverse effects like hERG inhibition (Choi2017 pages 1-1). Furthermore, the incorporation of a tertiary amine group may improve pharmacological properties, potentially enhancing efficacy against Mycobacterium tuberculosis (Choi2017 pages 5-6). However, the exact impact of these modifications on the molecule's activity and safety profile would require further empirical evaluation to confirm its potential as a TB antibiotic. References: 1. (Choi2017 pages 5-6): Choi, Peter J., et al. "Synthesis and Evaluation of Analogues of the Tuberculosis Drug Bedaquiline Containing Heterocyclic B-Ring Units." 2. (Choi2017 pages 1-1): Choi, Peter J., et al. "Synthesis and Evaluation of Analogues of the Tuberculosis Drug Bedaquiline Containing Heterocyclic B-Ring Units."'''

        act_5 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""Now I get the medicinal chemistry knowledge from the literature search agent, I should give all details and citations without summarizing it, while replace the description in the output back into smiles for the user to better understand."""
            },
        )
        obs_5 = 'OK'

        act_6 = AgentAct(name=FinishAct.action_name, 
                         params={INNER_ACT_KEY: ''' CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 shows potential as a novel tuberculosis (TB) antibiotic. The structural elements of this molecule are consistent with those evaluated in studies aiming to modify the B-ring of Bedaquiline, a known TB drug, to improve its pharmacological properties. The presence of a phenolic hydroxyl group and a bromine substituent on the benzene ring could enhance the molecule's activity by affecting its lipophilicity and inhibitory concentrations, similar to the promising results observed with other Bedaquiline analogues (Choi2017 pages 5-6). These structural modifications can potentially yield favorable lipophilicity, which is crucial for maintaining antibacterial potency while reducing adverse effects like hERG inhibition (Choi2017 pages 1-1). Furthermore, the incorporation of a tertiary amine group may improve pharmacological properties, potentially enhancing efficacy against Mycobacterium tuberculosis (Choi2017 pages 5-6). However, the exact impact of these modifications on the molecule's activity and safety profile would require further empirical evaluation to confirm its potential as a TB antibiotic. References: 1. (Choi2017 pages 5-6): Choi, Peter J., et al. "Synthesis and Evaluation of Analogues of the Tuberculosis Drug Bedaquiline Containing Heterocyclic B-Ring Units." 2. (Choi2017 pages 1-1): Choi, Peter J., et al. "Synthesis and Evaluation of Analogues of the Tuberculosis Drug Bedaquiline Containing Heterocyclic B-Ring Units."'''})
        obs_6 = "Task Completed."

        exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3),(act_4, obs_4), (act_5, obs_5), (act_6, obs_6)]
        self.add_example(task=exp_task_pack, action_chain=exp_act_obs)

        ## Example 6
        exp_task = "Please do a literature search based on only existing medicincal chemistry knowledge regarding toxicity for the molecule OC1CCN(CCCOc2ccc(CCc3nc(CCCCN4CCNCC4)no3)cc2)C1."
        exp_task_pack = TaskPackage(instruction=exp_task)

        act_1 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""In order to obtain existing medicinal chemistry knowledge, I have to first use {self.molecule_describer.name} to generate descriptions of this molecule OC1CCN(CCCOc2ccc(CCc3nc(CCCCN4CCNCC4)no3)cc2)C1, and then use {self.literature_searcher.name} to search for knowledge."""
            },
        )
        obs_1 = "OK"

        act_2 = AgentAct(
            name=self.molecule_describer.name,
            params={AGENT_CALL_ARG_KEY: "Please give me a molecular description for OC1CCN(CCCOc2ccc(CCc3nc(CCCCN4CCNCC4)no3)cc2)C1." }
        )
        obs_2 =  "This molecule is a diphenylmethane derivative with two aromatic benzene rings. One benzene ring has a phenolic hydroxyl group and a bromine substituent, while the two rings are connected through an alkyl chain with a tertiary amine group."

        act_3 = AgentAct(
            name=ThinkAct.action_name,
            params={
                INNER_ACT_KEY: f"""Now I get the description of the molecule, I should pass it to
                                {self.literature_searcher.name} to get knowledge about its toxicity properties."""
            },
        )
        obs_3 = "OK"
  
        act_4 = AgentAct(
            name=self.literature_searcher.name,
            # params={AGENT_CALL_ARG_KEY: '''You're an expert medicinal chemistry, you are accessing bioavailability of an input molecules. 
            #         Please evaluate the molecule with the following descriptors based on the substructures of the molecule on its bioavailability by gather all the information you have from the literatures. 
            #         Only include the information about the bioavailability and the substructure that leads to the favorable or unfavorable bioavailability property.
            #         Molecule Description: This molecule is a diphenylmethane derivative with two aromatic benzene rings. One benzene ring has a phenolic hydroxyl group and a bromine substituent, while the two rings are connected through an alkyl chain with a tertiary amine group.''' }
            params={AGENT_CALL_ARG_KEY: '''
                    Please evaluate the molecule with the following descriptions on its Toxicity.
                    Molecule Description: This molecule is a diphenylmethane derivative with two aromatic benzene rings. One benzene ring has a phenolic hydroxyl group and a bromine substituent, while the two rings are connected through an alkyl chain with a tertiary amine group.''' }
        
        )
        obs_4 =  '''The bioavailability of a compound with a dihydropyridine core, dimethyl and diethyl ester substitutions, a phenyl group, and an extended enone system with an isobutyl ester side chain can be influenced by several structural features. The dihydropyridine (DHP) core is crucial for binding interactions with biological targets, such as the multidrug-resistance transporter MDR1, and modifications to this core can significantly impact bioavailability (steiger2014synthesisandstudy pages 40-43). Ester substitutions at the C-3 and C-5 positions of the DHP core, such as dimethyl and diethyl esters, can affect the compound's conformation and, consequently, its solubility and metabolic stability, which are key factors in bioavailability (steiger2014synthesisandstudy pages 6-8).

                    The presence of a phenyl group and an extended enone system may enhance π-stacking interactions, which can improve binding affinity and potentially influence bioavailability positively (steiger2014synthesisandstudy pages 80-86). However, the inclusion of an isobutyl ester side chain could introduce steric hindrance, potentially affecting the compound's ability to interact with its target and thus its bioavailability (steiger2014synthesisandstudy pages 190-193). Overall, the structural modifications and substitutions in this compound are likely to have a complex impact on its bioavailability, balancing between enhanced binding interactions and potential steric challenges.'''

        act_5 = AgentAct(
            name=ThinkAct.action_name,
            params={
                # INNER_ACT_KEY: f"""Now I get the medicinal chemistry knowledge from the literature search agent, I should give all details and citations without summarizing it, while replace the description in the output back into smiles for the user to better understand. I should also evaluate which ADMET property, as detailed as possible, that might be the biggest problem for this molecule to become a potential drug"""
                INNER_ACT_KEY: f"""Now I get the medicinal chemistry knowledge from the literature search agent, I should respond with all the information."""
            },
        )
        obs_5 = "OK"

        act_6 = AgentAct(name=FinishAct.action_name, 
                         params={INNER_ACT_KEY: '''The bioavailability of a compound with a dihydropyridine core, dimethyl and diethyl ester substitutions, a phenyl group, and an extended enone system with an isobutyl ester side chain can be influenced by several structural features. The dihydropyridine (DHP) core is crucial for binding interactions with biological targets, such as the multidrug-resistance transporter MDR1, and modifications to this core can significantly impact bioavailability (steiger2014synthesisandstudy pages 40-43). Ester substitutions at the C-3 and C-5 positions of the DHP core, such as dimethyl and diethyl esters, can affect the compound's conformation and, consequently, its solubility and metabolic stability, which are key factors in bioavailability (steiger2014synthesisandstudy pages 6-8).
                    The presence of a phenyl group and an extended enone system may enhance π-stacking interactions, which can improve binding affinity and potentially influence bioavailability positively (steiger2014synthesisandstudy pages 80-86). However, the inclusion of an isobutyl ester side chain could introduce steric hindrance, potentially affecting the compound's ability to interact with its target and thus its bioavailability (steiger2014synthesisandstudy pages 190-193). Overall, the structural modifications and substitutions in this compound are likely to have a complex impact on its bioavailability, balancing between enhanced binding interactions and potential steric challenges.'''})
        obs_6 = "Task Completed."

        exp_act_obs = [(act_1, obs_1), (act_2, obs_2), (act_3, obs_3),(act_4, obs_4), (act_5, obs_5), (act_6, obs_6)]
        self.add_example(task=exp_task_pack, action_chain=exp_act_obs)


def run_agent(task:str,
              agent:str = 'manager',):
    
    llm_config_dict = {"llm_name": "gpt-3.5-turbo", 
                       "temperature": 0.0,
                       "max_tokens":2048}
    
    actions = [InhibitionPrediction()]

    llm_config = LLMConfig(llm_config_dict)

    llm = get_llm_backend(llm_config)

    ## Inhibition Prediction Agent
    # labor_agent = InhibitionAgent(llm=llm,
    #                              actions=actions)
    # test_task = "Is CC(c1ccncc1)(F)F inhibitory against M.tb?" # Inhibition

    ## Pharmacokinetics Agent
    # labor_agent = PharmacokineticsAgent(llm=llm,
    #                                     actions=[PharmacokineticsPrediction()])
    
    # test_task = "How's the pharmacokinetics properties of CC(c1ccncc1)(F)F" # PK
    
    ## Molecular Optimization Agent
    # labor_agent = MolecularOptimizationAgent(llm=llm,
    #                                          actions=[MolecularOptimization()])
    # test_task = "Please optimize this molecule CCCc1ccc(C(=O)OCCCC2(C)CC2)cc1 regarding its pharmacokinetic properties while maintaining its inhibition property."

    ## Molecule Generation
    # labor_agent = MoleculeGenerationAgent(llm=llm,
    #                                       actions=[MoleculeGeneration()])
    # test_task = 'Please give me a molecule with potential M.tb inhibition.'

    ## Describe Molecule Agent
    # labor_agent = MoleculeDescriber(llm=llm,
    #                                 actions=[DescribeMolecule()])
    # test_task = 'Please give me a molecular description for CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1'
    # test_task_pack = TaskPackage(instruction=test_task,
    #                              task_creator="User")
    # response = labor_agent(test_task_pack)

    ## PaperQA Agent
    # labor_agent = LiteratureAgent(llm=llm,
    #                                 actions=[SearchLiterature()])
    # test_task = 'Please evaluate the molecule with the following descriptors on whether it has potential to become a novel TB antibiotic. Molecule Description: This molecule is a diphenylmethane derivative with two aromatic benzene rings. One benzene ring has a phenolic hydroxyl group and a bromine substituent, while the two rings are connected through an alkyl chain with a tertiary amine group.'
    # test_task_pack = TaskPackage(instruction=test_task,
    #                              task_creator="User")
    # response = labor_agent(test_task_pack)

    # print("response:", response)

    ## Manager
    labor_agent = ManagerAgent(llm=llm)
    # test_task = "Please evaluate CCCc1ccc(CNCCc2csc(CCCN3CCN(C)CC3)c2)cc1 using existing medicinal chemistry knowledge on whether it has potential to become a M.tb antibiotics."
    test_task = task
    # test_task = "What is the potential of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1 as a Mtb antibiotics? Using 0.7 as an inhibition threshold."
    # test_task = "Please generate a molecule that has potential for M.tb inhibition, and optimize it regarding its pharmacokinetic properties while maintaining its inhibition property."
    test_task_pack = TaskPackage(instruction=test_task,
                                 task_creator="User")
    response = labor_agent(test_task_pack)

    print("response:", response)

    return response


if __name__ == "__main__":
#     # run_agent(task='''You're an expert medicinal chemistry, you are accessing ADMET properties of an input molecules. Please do a literature search based on only existing medicincal chemistry knowledge regarding admet property for the molecule c1cc(cc(c1)Br)C(=O)NC2C3CC4CC(C3)CC2C4, 
#     #                 tell me what admet properties it may fail and the substructure that leads to that. Do not include any information regarding testing methods. Have your output be organized, so it addressed every perspect of ADMET. Search from multiple sources.''')
    # run_agent(task='''You're an expert medicinal chemistry, you are accessing bioavailability of an input molecules. Please do a literature search based on only existing medicincal chemistry knowledge regarding its bioavailability for the molecule CCOC(=O)C1=C(C)NC(C)=C(C(=O)OCC)C1c1ccccc1/C=C/C(=O)OC(C)(C)C 
    #                 tell me what substructures that lead you think it has good or bad bioavailability, if not enough information, just say not enough information.''')

    run_agent(task='''You're an expert medicinal chemistry, you are accessing bioavailability of an input molecules. Please do a literature search based on only existing medicincal chemistry knowledge regarding bioavailability for the molecule c1cc(ccc1CN2CCN(CC2)c3c4c(nc[nH]4)nc(n3)N)Cl.''')



