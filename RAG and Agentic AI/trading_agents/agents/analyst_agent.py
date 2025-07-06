import os
from crewai import Agent, LLM
# from litellm import LiteLlm
from dotenv import load_dotenv
from helper.llm_init import llm
from tools.stock_research_tool import get_stock_price

load_dotenv()


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