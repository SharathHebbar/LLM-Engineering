from crewai import Task
from agents.analyst_agent import analyst_agent

get_stock_analysis = Task(
    description=(
        "Analyse the recent performance of the stock {stock}. use the live stock information tool to retrieve "
        "current price, percentage change, trading volume, and other market data. Provide a summary of how the stock"
        "is performing today and highlight any key observations from the data."
    ),
    expected_output=(
        "A clear, bullet-pointed summary of:\n"
        "- Current Stock price\n"
        "- Daily price change and percentage\n"
        "- Volume and volatility\n"
        "- Any immediate trends or observations"
    ),
    agent=analyst_agent
)