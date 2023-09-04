from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.llms import OpenAI
from decouple import config


llm = OpenAI(temperature=0.6)

tools = load_tools(['serpapi', 'llm-math'], llm=llm)

agent = initialize_agent(
	tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

print(agent.agent.llm_chain.prompt.template)

agent.run("Configuration de l'environnement DBT")
