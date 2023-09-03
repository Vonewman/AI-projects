import json
from typing import Sequence, List

from llama_index.llms import OpenAI, ChatMessage
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

class MyAIAgent:
    def __init__(
        self,
        tools: Sequence[BaseTool] = [],
        llm: OpenAI = OpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
        chat_history: List[ChatMessage] = [],
    ) -> None:
        self._llm = llm
        self._tools = {tool.metadata.name: tool for tool in tools}
        self._chat_history = chat_history

    def reset(self) -> None:
        self._chat_history = []

    def chat(self, message: str) -> str:
        chat_history = self._chat_history
        chat_history.append(ChatMessage(role="user", content=message))
        functions = [
            tool.metadata.to_openai_function() for _, tool in self._tools.items()
        ]

        ai_message = self._llm.chat(chat_history, functions=functions).message
        chat_history.append(ai_message)

        function_call = ai_message.additional_kwargs.get("function_call", None)
        if function_call is not None:
            function_message = self._call_function(function_call)
            chat_history.append(function_message)
            ai_message = self._llm.chat(chat_history).message
            chat_history.append(ai_message)

        return ai_message.content


    def _call_function(self, function_call: dict) -> ChatMessage:
        tool = self._tools[function_call["name"]]
        output = tool(**json.loads(function_call["arguments"]))
        return ChatMessage(
            name=function_call["name"],
            content=str(output),
            role="function",
            additional_kwargs={"name": function_call["name"]},
        )


agent = MyAIAgent(tools=[multiply_tool, add_tool])
agent.chat("Hi")
print(agent.chat("What is 2123 * 215123"))
