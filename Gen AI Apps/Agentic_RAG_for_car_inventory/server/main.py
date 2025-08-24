import streamlit as st
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict, Tuple
from langchain_core.documents import Document
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

model="gemini-2.0-flash"
emb_model = "models/gemini-embedding-001"


llm = ChatGoogleGenerativeAI(model=model)
embeddings = GoogleGenerativeAIEmbeddings(model=emb_model)


from typing import List
import requests

loader = TextLoader(file_path="./car_inventory.txt")
docs = loader.load()


splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=700,
    chunk_overlap=0
)

split_docs = splitter.split_documents(docs)

vectorstore = FAISS.from_documents(split_docs, embeddings)

st.title("AI Car-Buying Assistant")


def return_prompt(user_input, context):

    prompt_template = """
    You are an expert AI car-buying assistant.

    Below is the prior conversation history between you and the user:
    {chat_history}

    Below is a list of vehicle listings:  
    {context}

    User question: {question}

    Guidelines:
    1. ONLY use information present in the context. Do not fabricate or guess.
    2. If user asks about prices, calculate final price = msrp - discount (in USD).
    3. If no vehicle matches, say: "Sorry, I couldn’t find any vehicles matching your criteria."
    4. If vehicles match, show up to 2 suggestions with: make, model, year, final price, discount, dealer.
    5. Be concise, friendly, and factual.

    Now based on the context and question, provide the best response:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=prompt_template
    )

    return prompt


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    chat_history: List[Tuple[str, str]]

def retrieve(state: State):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True
    )
    response = qa_chain.invoke(state['question'])
    return {"context": response['result']}

def generate(state: State):
    prompt = return_prompt(user_input, state['context'])
    history_str = '\n'.join([f"User: {q}\nAssistant: {a}" for q, a in state.get("chat_history", [])])
    messages = prompt.invoke(
        {
            "question": state["question"],
            "context": state['context'],
            "chat_history": history_str
        }
    )
    response = llm.invoke(messages)

    chat_history = state.get("chat_history", [])
    chat_history.append((state["question"], response.content))
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


if "messages" not in st.session_state:
    st.session_state.messages = []


if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "system",
        "content": ""
    })

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message['role']):
            st.markdown(message["content"])

if user_input := st.chat_input("Your answer.", max_chars=1000):
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = graph.invoke({"question": user_input})
        st.session_state.messages.append({
            "role": "user",
            "content": response['answer']
        })
        st.markdown(response['answer'])
        st.subheader("Context")
        st.markdown(response['context'])