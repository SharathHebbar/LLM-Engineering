# AI Car-Buying Assistant

## Installation

```bash
pip install -r requirements.txt
```

### Project Overview

This project is a **Car-Buying Assistant Chatbot** built using Retrieval-Augmented Generation (RAG) to help users discover cars based on their preferences such as price, make, condition, body style, and more. The system leverages **LangChain** and **LangGraph** to build a modular, controllable RAG pipeline. A **Streamlit UI** provides a clean and interactive front-end for users to chat with the assistant.

---

### Tech Stack

| Component        | Technology                                 |
|------------------|---------------------------------------------|
| **Language Model**  | Gemini (`gemini-2.0-flash`)               |
| **Embedding Model** | Gemini Embeddings (`gemini-embedding-001`) |
| **AI Framework**     | LangChain                               |
| **RAG Controller**   | LangGraph                               |
| **Vector Store**     | FAISS                                   |
| **User Interface**   | Streamlit                               |
| **Data Generator**   | Faker (for synthetic dataset)           |
| **Text Splitter**    | LangChain `CharacterTextSplitter`       |

---

### Methodology

#### 1. **Data Preparation**
- A synthetic dataset of car listings was generated using the **Faker** library.
- Each listing includes realistic attributes like `make`, `model`, `year`, `price`, `condition`, `fuel_type`, `dealer`, and more.
- Web scraping was considered but discarded due to inconsistency and data cleanliness issues.

#### 2. **Data Formatting**
- Original data was in **JSON format**, which caused misinterpretations when passed directly to the LLM.
- To address this, each JSON object was converted into a **flat, readable key-value format string**, improving token comprehension by the LLM.

#### 3. **Document Chunking & Vectorization**
- LangChain's `CharacterTextSplitter` was used to divide the dataset into smaller, contextually valid chunks.
- Each chunk was embedded using **Gemini embeddings** and indexed using **FAISS** for fast similarity search.

#### 4. **RAG Pipeline with LangGraph**
- A **LangGraph `StateGraph`** was created to orchestrate the RAG pipeline with two main components:
  - **Retriever Node**: Fetches top 2–5 relevant documents from the vector store.
  - **Generator Node**: Uses the user query + context to generate a natural language answer.
- This architecture allows **modular control, fine-tuned flow design, and traceable state**.

#### 5. **Chat Interface**
- A **Streamlit** app serves as the chatbot interface, allowing users to:
  - Ask open-ended questions like "Show me electric SUVs under $30k".
  - Filter cars by price, condition, body style, or brand.
  - View concise, LLM-generated responses based on actual data.

