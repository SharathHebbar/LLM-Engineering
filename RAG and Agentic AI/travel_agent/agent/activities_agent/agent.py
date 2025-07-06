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

activities_agent = Agent(
    name="activities_agent",
    model=model,
    description="Suggests interesting activities for the user at a destination.",
    instruction="Given a destination, dates, and budget, suggest 2-3 engaging toursit or cultural activities. \
        For each activity, provide a name, a short description, price estimate, and duration in hours.\
        Respond in plain English. Keep it concise and well-formatted."
)


session_service = InMemorySessionService()

runner = Runner(
    agent=activities_agent,
    app_name="activities_app",
    session_service=session_service
)

USER_ID = "user_activities"
SESSION_ID = "session_activities"


async def execute(request):
    session_service.create_session(
        app_name="activities_app",
        user_id=USER_ID,
        session_id=SESSION_ID
    )

    prompt = (
        f"User is flying to {request['destination']} from {request['start_date']} to {request['end_date']}, "
        f"with a budget of {request['budget']}. Suggest 2-3 activities, each with name, description, price estimate and duration. "
        f"Respond in JSON format using the key 'activities' with a list of activity objects."        
    )

    messages = types.Content(role="user", parts=[types.Part(text=prompt)])

    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=messages):
        if event.is_final_response():
            response_text = event.content.parts[0].text
            try:
                parsed = json.loads(response_text)
                if "activities" in parsed and isinstance(parsed['activities'], list):
                    return {"activities": parsed['activities']}
                else:
                    print("'activities' key missing or not a list in response JSON.")
                    return {"activities": response_text}
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed: {e}")
                print(f"Response Content: {response_text}")
                return {"activities": response_text}
            

            