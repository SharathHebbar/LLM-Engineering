import os
from crewai import Agent, LLM
from helper.llm_init import llm
from tools.stock_research_tool import get_stock_price


planner_agent = Agent(
    role="Planner Agent",
    goal=("Identify the doctors and patients schedule and plan patients visit."
        "identifying trends, performance insights, and key financial signals to support decision-making."),
    backstory=("You are a veteran financial analyst with deep expertise in interpreting stock market data,"
               "technical trends, and fundamentals. You specialize in producing well-structured reports that evaluate stock"
               "performance using live market indicators."

    ),
    llm=llm,
    tool=[get_stock_price],
    verbose=True
    
)