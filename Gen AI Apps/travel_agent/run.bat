uvicorn agent.host_agent.__main__:app --port 8000 &
uvicorn agent.flight_agent.__main__:app --port 8001 &
uvicorn agent.stay_agent.__main__:app --port 8002 &      
uvicorn agent.activities_agent.__main__:app --port 8003 &
streamlit run travel_ui.py