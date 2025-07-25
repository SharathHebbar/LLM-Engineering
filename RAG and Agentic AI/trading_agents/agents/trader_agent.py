from crewai import  Agent, LLM
from helper.llm_init import llm
from dotenv import load_dotenv

load_dotenv()

trader_agent = Agent(
    role="Strategic Stock Trader",
    goal=(
        "Decide whether to Buy, Sell, or Hold a given stock based on live market data, "
        "price movements, and financial analysis provided by available data."
        ),
    backstory=(
        "You are a strategic trader with years of experience in timing market entry and exit points."
        "You rely on real-time stock data, daily price movements, and volume trends to make trading decisions that optimize returns and reduce risk."
    ),
    llm=llm,
    tool=[],
    verbose=True
    
)