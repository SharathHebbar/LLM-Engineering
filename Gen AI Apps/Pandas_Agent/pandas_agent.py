import os

import pandas as pd

from langchain_groq import ChatGroq, data
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

from dotenv import load_dotenv

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')



class PandasAgent:

    def __init__(self, df: pd.DataFrame, cache_manager=None, model_name='openai/gpt-oss-20b'):
        self.df = df
        self.cache_manager = cache_manager
        self.llm = ChatGroq(model=model_name)

        self.pandas_agent_prompt = f"""
You are a data analyst working with a pandas dataframe.

You MUST follow this format:

Thought: describe what you want to do
Action: python_repl_ast
Action Input: valid python code

When giving final answer:
Final Answer: your explanation

The dataframe has columns: {', '.join(self.df.columns)}
"""
        self.agent = create_pandas_dataframe_agent(
            self.llm, 
            self.df, 
            verbose=True,
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            prefix=self.pandas_agent_prompt,
            agent_type="tool-calling"
        )


    def run(self, question: str):
        try:
            if self.cache_manager:
                exact_cache = self.cache_manager.get_exact_cache(question)
                if exact_cache:
                    print("Exact cache hit!")
                    return exact_cache
                
                semantic_cache = self.cache_manager.get_semantic_cache(question)
                if semantic_cache:
                    print("Semantic cache hit!")
                    return semantic_cache
            
            answer = self.agent.invoke({"input": question})

            if self.cache_manager:
                self.cache_manager.store_cache(question, answer['output'])

            return answer['output']
        except Exception as e:
            return f"Error processing the question: {str(e)}"