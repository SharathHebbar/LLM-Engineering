import streamlit as st
import time
st.header("Test")

max_cap = 15

progress_text = "Transcription in progress..."
my_bar = st.progress(0, text=progress_text)

for i in range(max_cap):

    my_bar.progress(i + 1, text = progress_text)
    time.sleep(2)
    st.write(i)