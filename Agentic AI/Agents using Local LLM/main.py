import os
import json
import httpx
from phi.tools.duckduckgo import DuckDuckGo
from phi.agent.python import PythonAgent
from phi.file.local.csv import CsvFile
from phi.agent import Agent, RunResponse
from phi.model.openai.like import OpenAILike
MODEL = "hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF"

from openai import OpenAI

model=OpenAILike(base_url="http://localhost:1234/v1", api_key="lm-studio", id=MODEL)
# agent = Agent(
#     model=model,
#     markdown=True
# )

# Get the response in a variable
# run: RunResponse = agent.run("Share a 2 sentence horror story.")
# print(run.content)

# Print the response in the terminal
# agent.print_response("What is the purpose of life")

web_search_agent = Agent(
    model=model,
    instructions=["Always include sources"],
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
)

# web_search_agent.print_response("Tell me about OpenAI Sora?", stream=True)
# web_search_agent.print_response("Latest Updated from Antrophic MCP (Model Context Protocol) explain me more about it", stream=True)
web_search_agent.print_response("Who is the founder of Super Safe Intelligence company", stream=True)