from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import json

model = "gemini-2.0-flash"
# model = LiteLlm(
#     base_url="http://localhost:1234/v1",
#     model="lm_studio/qwen3-0.6b",
#     api_key="lm-studio"
# )

host_agent = Agent(
    name="host_agent",
    model=model,
    description="Coordinates travel planning by calling flight, stay and activity agents.",
    instruction=(
        "You are the host agent responsible for orchestrating trip planning tasks.",
        "You call external agents to gather flights, stays, and activities, then return a final result."
    )
)


session_service = InMemorySessionService()

runner = Runner(
    agent=host_agent,
    app_name="host_app",
    session_service=session_service
)

USER_ID = "user_host"
SESSION_ID = "session_host"


async def execute(request):
    session_service.create_session(
        app_name="host_app",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    prompt = (
        f"Plan a trip tp  {request['destination']} from {request['start_date']} to {request['end_date']}, "
        f"within a total budget of {request['budget']}. Call the flights, stays, and activities agents for results."        
    )

    messages = types.Content(role="user", parts=[types.Part(text=prompt)])

    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=messages):
        if event.is_final_response():
            response_text = event.content.parts[0].text
            return {"summary": response_text}
            

