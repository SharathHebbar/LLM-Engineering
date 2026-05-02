import os
import pandas as pd

from pandas_agent import PandasAgent
from caching import CacheManager

if __name__ == "__main__":
    cache_manager = CacheManager()
    df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['New York', 'Los Angeles', 'Chicago']
    })
    agent = PandasAgent(df, cache_manager=cache_manager)

    question = "What is the average age of people in the dataframe?"
    answer = agent.run(question)
    print("Answer:", answer)