from crewai import  Agent, LLM

from dotenv import load_dotenv

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