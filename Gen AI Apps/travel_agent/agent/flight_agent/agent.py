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

flight_agent = Agent(
    name="flight_agent",
    model=model,
    description="Suggests flight details for the user at a destination.",
    instruction="Given a destination, dates, and budget, suggest 2-3 flight details. \
        For each flight, provide a name, a short description, price estimate, and duration in hours.\
        Respond in plain English. Keep it concise and well-formatted."
)


session_service = InMemorySessionService()

runner = Runner(
    agent=flight_agent,
    app_name="flight_app",
    session_service=session_service
)

USER_ID = "user_flight"
SESSION_ID = "session_flight"


async def execute(request):
    session_service.create_session(
        app_name="flight_app",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    prompt = (
        f"User is flying to {request['destination']} from {request['start_date']} to {request['end_date']}, "
        f"with a budget of {request['budget']}. Suggest 2-3 flight, each with name, description, price estimate and duration. "
        f"Respond in JSON format using the key 'flight' with a list of activity objects."        
    )

    messages = types.Content(role="user", parts=[types.Part(text=prompt)])

    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=messages):
        if event.is_final_response():
            response_text = event.content.parts[0].text
            try:
                parsed = json.loads(response_text)
                if "flight" in parsed and isinstance(parsed['flight'], list):
                    return {"flight": parsed['flight']}
                else:
                    print("'flight' key missing or not a list in response JSON.")
                    return {"flight": response_text}
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Response Content: {response_text}")
                return {"flight": response_text}
            

            