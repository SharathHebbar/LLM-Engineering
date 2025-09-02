import uvicorn
from fastapi import FastAPI, Request
import os
from dotenv import load_dotenv

from src.graphs.graph_builder import GraphBuilder
from src.llms.groq_llm import GroqLLM

load_dotenv()


app = FastAPI()

# LANGSMITH_PROJECT

os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")

@app.post("/blogs")
async def create_blogs(request: Request):
    data = await request.json()
    topic = data.get("topic", "")

    llm = GroqLLM().get_llm()

    graph_builder = GraphBuilder(llm)
    if topic:
        graph = graph_builder.setup_graph(usecase="topic")
        state = graph.invoke({"topic": topic})
 
    return {"data": state}


if __name__  == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)