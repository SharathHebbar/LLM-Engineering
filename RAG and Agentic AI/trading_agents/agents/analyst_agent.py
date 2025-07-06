import os
from crewai import Agent, LLM
# from litellm import LiteLlm
from dotenv import load_dotenv
from tools.stock_research_tool import get_stock_price

load_dotenv()

# llm = LLM(
#     base_url="http://localhost:1234/v1",
#     model="lm_studio/qwen3-0.6b",
#     api_key="lm-studio"
# ) 


# llm = LLM(
#     model="groq/llama-3.3-70b-versatile",
#     temperature=0
# )

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0
)

analyst_agent = Agent(
    role="Financial  Market Analyst",
    goal=("Perform in-depth evaluations of publicly traded stocks using real-time data,"
        "identifying trends, performance insights, and key financial signals to support decision-making."),
    backstory=("You are a veteran financial analyst with deep expertise in interpreting stock market data,"
               "technical trends, and fundamentals. You specialize in producing well-structured reports that evaluate stock"
               "performance using live market indicators."

    ),
    llm=llm,
    tool=[get_stock_price],
    verbose=True
    
)