from crewai import  Agent, LLM


llm = LLM(
    base_url="http://localhost:1234/v1",
    model="lm_studio/medgemma-4b-it",
    api_key="lm-studio"
)


# llm = LLM(
#     base_url="http://localhost:1234/v1",
#     model="lm_studio/qwen3-0.6b",
#     api_key="lm-studio"
# ) 

# llm = LLM(
#     model="groq/llama-3.3-70b-versatile",
#     temperature=0
# )


# llm = LLM(
#     model="gemini/gemini-2.0-flash",
#     temperature=0
# )
