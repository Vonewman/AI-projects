from llama_index.agent import OpenAIAgent
from llama_index.llms import OpenAI

from llama_index.tools import BaseTool, FunctionTool

import nest_asyncio

def multiply(a: int, b: int) -> int:
	"""Multiply two integers and returns the result integer"""
	return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)


def add(a: int, b: int) -> int:
	"""Add two integers and returns the result integer"""
	return a + b

add_tool = FunctionTool.from_defaults(fn=add)

llm = OpenAI(model="gpt-3.5-turbo-0613")
agent = OpenAIAgent.from_tools([multiply_tool, add_tool], 
    llm=llm, verbose=True)

response = agent.chat("What is (121 * 3) + 42?")
print(str(response))