from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.trace import Tracer
from typing import List
import asyncio 
import hashlib


tracer: Tracer = trace.get_tracer(__name__)

app = FastAPI()


async def retrieve_documents(query: str)-> List[str]:

    """Simulate document retrieval (Ex Vector Search or Knowledge base lookup).
    This function represents the retrieval stage in a RAG Pipeline.
    In a real system, this might query a vector database or search index
    """

    await asyncio.sleep(0.05) # Latency Simulation

    return [
        "FastAPI enables high-performance async APIs.",
        "OpenTelemetry provides vendor-neutral observability.",
        "LLM Observability requires tracing prompts and tokens."
    ]

def build_prompt(query: str, documents: List[str]) -> str:
    """
    Construct the final prompt from retrieved documents and the user query.
    Prompt Construction is kept separate so it can be observed or modified 
    independently if needed
    """

    context = "\n".join(documents)
    return f"""Context: {context} Question: {query}"""


class LLMResponse:
    """
    Minimal Abstraction for an LLM Response
    This keeps the example self-contained while still allowing us to attach token usage and other metadata for observability
    """

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int):
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
    


async def call_llm(prompt: str) -> LLMResponse:
    """Simulate an LLM API Call. In a real implementation, this would call APIs"""

    await asyncio.sleep(0.2)

    response_text = "FastAPI and OpenTelemetry enable end-to-end LLM Observability."
    prompt_tokens = len(prompt.split())
    completion_tokens = len(response_text.split())

    return LLMResponse(response_text, prompt_tokens, completion_tokens)


def summarize_repsonse(response: LLMResponse) -> str:
    """Example post-processing step.
    Post-processing is separated into its own phase so any additional latency or errors are not incorrectly attributed to the LLM itself
    """
    return response.text

# Observable FastAPI Endpoint
@app.post("/query")
async def rag_query(request: Request, query: str):
    """Handle a single RAG style request with explicit OpenTelemetry spans.
    This Endpoint demonstrates how to create one trace per request, with child spans
    for retrieval, LLM invocation, and post-processing.
    """

    with tracer.start_as_current_span("http.request") as http_span:
        http_span.set_attribute("http.method", "POST")
        http_span.set_attribute("http.route", "/query")

        with tracer.start_as_current_span("rag.retrieval") as retrieval_span:
            retrieval_span.set_attribute("rag.top_k", 4)
            retrieval_span.set_attribute("rag.similarity_threshold", 0.8)
            documents = await retrieve_documents(query)

            retrieval_span.set_attribute("rag.documents_returned", len(documents))

        with tracer.start_as_current_span("llm.call") as llm_span:
            llm_span.set_attribute("llm.provider", "example")
            llm_span.set_attribute("llm.model", "example-llm")
            llm_span.set_attribute("llm.temperature", 0.7)
            llm_span.set_attribute("llm.prompt_template_id", "rag_v1")

            prompt = build_prompt(query, documents)

            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            llm_span.set_attribute("llm.prompt_hash", prompt_hash)
            llm_span.set_attribute("llm.prompt_length", len(prompt))

            response = await call_llm(prompt)

            response_hash = hashlib.sha256(response.text.encode()).hexdigest()
            llm_span.set_attribute("llm.response_hash", response_hash)
            
            llm_span.set_attribute("llm.usage.prompt_tokens", response.prompt_tokens)
            llm_span.set_attribute("llm.usage.completion_tokens", response.completion_tokens)
            llm_span.set_attribute("llm.usage.total_tokens", response.total_tokens)

            estimated_cost = response.total_tokens * 0.000002

            llm_span.set_attribute("llm.cost_estimated_usd", estimated_cost)


        with tracer.start_as_current_span("llm.postprocess") as post_span:
            summary = summarize_repsonse(response)
            post_span.set_attribute("llm.summary_length", len(summary))

    

    return {"summary": summary}

