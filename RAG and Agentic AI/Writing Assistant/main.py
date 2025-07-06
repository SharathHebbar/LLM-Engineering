import os

import streamlit as st

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from dotenv import load_dotenv


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "qwen-2.5-32b"

if not GROQ_API_KEY:
    st.error("API Key not found")
    st.stop()


st.title("Text Improvement App")

st.sidebar.header("Select an App")

st.sidebar.markdown("Choose the functionality you would like to use: ")


app_choice = st.sidebar.selectbox(
    "Choose an App",
    options = [
        "Rewrite Sentence",
        "Image and Video Prompt Generator",
        "Text Simplifier"
    ]
)

rewrite_template = """
Below is a draft text that may need improvement.
Your goal is to:
- Edit the draft for clarity and readability.
- Adjust the tone as specified.
- Adapt the text to the requested dialect.

**Tone Examples: **
- **Formal:** "Greetings! Elon Musk has announced a new innovation at Tesla, revolutionizing the electric vehicle industry. After extensive research and development, this breakthrough aims to enhance sustainability and efficiency. We look forward to seeing its impact on the market."  
- **Informal:** "Hey everyone! Huge news—Elon Musk just dropped a game-changing update at Tesla! After loads of work behind the scenes, this new tech is set to make EVs even better. Can’t wait to see how it shakes things up!"  

**Dialect Differences:**  
- **American English:** French fries, apartment, garbage, cookie, parking lot  
- **British English:** Chips, flat, rubbish, biscuit, car park  
- **Australian English:** Hot chips, unit, rubbish, biscuit, car park  
- **Canadian English:** French fries, apartment, garbage, cookie, parking lot  
- **Indian English:** Finger chips, flat, dustbin, biscuit, parking space  

Start with a warm introduction if needed.  

**Draft Text, Tone, and Dialect:**  
- **Draft:** {draft}  
- **Tone:** {tone}  
- **Dialect:** {dialect}  

**Your {dialect} Response:**  
"""

prompt_generator_template = """
Below is a sentence written in poor English:
"{poor_sentence}"

Your task is to generate a detailed and descriptive prompt optimized for text-to-image or text-to-video generation.
The prompt should be vivid and visually-oriented to help generate high-quality media content.
"""

image_video_template = """
Below is a sentence:  
"{sentence}"

Your task is to generate a detailed and descriptive prompt optimized for text-to-image or text-to-video generation.  
The prompt should be vivid and visually-oriented to help generate high-quality media content.
"""

text_simplifier_template = """
Below is a piece of complex text:
"{complex_text}"

Your task is to rewrite this text in simpler and clearer language while preserving its original meaning.
"""

def load_LLM(GROQ_API_KEY, MODEL_NAME):
    """Loads LLM Model for Processing"""

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name = MODEL_NAME,
        streaming=True
    )
    return llm

st.header(f"{app_choice}")
st.markdown("Provide the required inputs.")
with st.container():
    if app_choice == "Rewrite Sentence":
        draft = st.text_area("Draft Text", height=200, placeholder="Enter your text here .......")
        col1, col2 = st.columns(2)
        with col1:
            tone = st.selectbox("Select your desired tone", options=["Formal", "Non Formal"])
        with col2:
            dialect = st.selectbox("Select dialect", options=[
                "American English",
                "British English",
                "Australian English",
                "Canadian English",
                "Indian English"
            ])
    elif app_choice == "Image and Video Prompt Generator":
        sentence = st.text_area("Sentence", height=200, placeholder="Enter a sentence describing your desired media.........")
    elif app_choice == "Text Simplifier":
        complex_text = st.text_area("Complex Text", height=200, placeholder="Enter the complex text here......")
    result = ""
    if st.button("Process"):
        with st.spinner("Processing your text...."):
            llm = load_LLM(GROQ_API_KEY, MODEL_NAME)
            if app_choice == "Rewrite Sentence":
                prompt_obj = PromptTemplate(
                    input_variables=["tone", "dialect", "draft"],
                    template=rewrite_template
                )
                chain = LLMChain(llm=llm, prompt=prompt_obj)
                result = chain.run(draft=draft, tone=tone, dialect=dialect)
            elif app_choice == "Image and Video Prompt Generator":
                prompt_obj = PromptTemplate(
                    input_variables=["sentence"],
                    template=image_video_template
                )
                chain = LLMChain(llm=llm, prompt=prompt_obj)
                result = chain.run(sentence=sentence)
            elif app_choice == "Text Simplifier":
                prompt_obj = PromptTemplate(
                    input_variables=["complex_text"],
                    template=text_simplifier_template
                )
                chain = LLMChain(llm=llm, prompt=prompt_obj)
                result = chain.run(complex_text=complex_text)
    
    st.markdown("#### Output: ")
    st.markdown(result)
