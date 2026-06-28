import streamlit as st

import os

from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase


load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = init_chat_model("openai/gpt-oss-20b", model_provider="groq", api_key=GROQ_API_KEY)

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

for tool in tools:
    print(f"Tool name: {tool.name}")
    print(f"Tool description: {tool.description}\n")


system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_agent(
    model, tools, system_prompt=system_prompt
)


question = st.chat_input("Ask a question about the database:")

if question:
    data = agent.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question,
                }
            ]
        })

    for msg in data["messages"]:
        
        if msg.__class__.__name__ == "HumanMessage":
            with st.chat_message("user"):
                st.write(msg.content)

        elif msg.__class__.__name__ == "AIMessage":
            with st.chat_message("assistant"):
                if msg.content:
                    st.write(msg.content)

                # Optional: show reasoning
                reasoning = msg.additional_kwargs.get("reasoning_content")
                if reasoning:
                    with st.expander("🧠 Reasoning"):
                        st.write(reasoning)

                # Optional: show tool calls
                tool_calls = msg.additional_kwargs.get("tool_calls")
                if tool_calls:
                    with st.expander("🔧 Tool Calls"):
                        st.json(tool_calls)

        elif msg.__class__.__name__ == "ToolMessage":
            with st.chat_message("tool"):
                st.code(msg.content)
