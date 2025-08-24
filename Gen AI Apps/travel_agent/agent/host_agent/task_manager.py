from common.a2aclient import call_agent

FLIGHT_URL = "http://127.0.0.1:8001/run"
STAY_URL = "http://127.0.0.1:8002/run"
ACTIVITIES_URL = "http://127.0.0.1:8003/run"

async def run(payload):
    print(f"Incoming Payload: {payload}")
    flights = await call_agent(FLIGHT_URL, payload)
    stay = await call_agent(STAY_URL, payload)
    activities = await call_agent(ACTIVITIES_URL, payload)

    print(f"Flights: {flights}")
    print(f"Stay: {stay}")
    print(f"Activities: {activities}")

    flights = flights if isinstance(flights, dict) else {}
    stay = stay if isinstance(stay, dict) else {}
    activities = activities if isinstance(activities, dict) else {}


    return {
        "flights": flights.get("flights", "No flights returned."),
        "stay": stay.get("stay", "Np Stay Options returned."),
        "activities": activities.get("activities", "No activities returned.")
    }