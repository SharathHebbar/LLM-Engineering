import streamlit as st
import pandas as pd

from brain import brain_functions



st.title("Conversation Bot")
st.markdown(":violet-badge[:material/star: Agents] :violet-badge[:material/star: GenAI]")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_flag = True
table_df = None
table_name = None
with st.sidebar:
    st.header("Upload your dataset here")

    uploaded_file = st.file_uploader(
        label="Upload your .csv files",
        type=".csv"
    )
    if uploaded_file:
        print(uploaded_file.name)
        chat_flag = False
        dataframe = pd.read_csv(uploaded_file)
        table_df = dataframe
        table_name = uploaded_file.name
        table_name = table_name.split(".")[0]
        st.write(dataframe)
        


for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message['role']):
            if isinstance(message["content"], pd.DataFrame):
                st.dataframe(message["content"])
            else:
                st.markdown(message["content"])


if prompt := st.chat_input("Initiate Chat.", max_chars=1000, disabled=chat_flag):
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    with st.chat_message("user"):
        st.markdown(prompt)
    

    with st.chat_message("assistant"):
        output = brain_functions(table_df, table_name, prompt)
        if output is None:
            st.markdown("Error Fetching details. Please check your query")
            st.session_state.messages.append({
                "role": "assistant",
                "content": output
            })
        else:
            if isinstance(output, pd.DataFrame):
                st.dataframe(output)
            else:
                st.write(output)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": output
            })
