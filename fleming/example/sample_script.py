from agentlite.commons import TaskPackage
from agentlite.agents import BaseAgent
from DladActions import InhibitionPrediction
from agentlite.llm.LLMConfig import LLMConfig
from agentlite.logging.terminal_logger import AgentLogger
from agentlite.llm.agent_llms import BaseLLM, get_llm_backend

agent_logger = AgentLogger(PROMPT_DEBUG_FLAG=True)

llm_config_dict = {"llm_name": "gpt-4", "temperature": 0.5}
llm_config = LLMConfig(llm_config_dict)
llm = get_llm_backend(llm_config)
# define an individual agent
search_agent_info = {
    "name": "mtb_inhibition_prediction_agent",
    "role": "You can use the InhibitionPrediction tool to predict M.tb inhibition for a molecule"
}
search_agent = BaseAgent(name=search_agent_info["name"], 
                         role=search_agent_info["role"], 
                         llm=llm, 
                         actions=[InhibitionPrediction()], 
                         logger=agent_logger
                         )

test_task = "What's the inhibition of CCc1ccc(CCN(C)CCc2ccc(O)c(Br)c2)cc1?"
test_task_pack = TaskPackage(instruction=test_task)
response = search_agent(test_task_pack)
print("response:", response)