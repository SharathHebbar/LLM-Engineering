import streamlit as st
import pandas as pd

from caching import CacheManager
from pandas_agent import PandasAgent

st.title("Pandas Agent")


data = pd.read_excel("dataset/P_Data_Extract_From_World_Development_Indicators.xlsx", sheet_name="Data")
metadata = pd.read_excel("dataset/P_Data_Extract_From_World_Development_Indicators.xlsx", sheet_name="Series - Metadata")

cache_manager = CacheManager()
agent = PandasAgent(data, cache_manager=cache_manager)

st.sidebar.header("Data Preview")
if st.sidebar.checkbox("Show Data"):
    st.sidebar.subheader("Data")
    st.sidebar.dataframe(data.head())


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask a question about the data")
if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)
    


    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = agent.run(user_input)

        st.markdown(answer)
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })