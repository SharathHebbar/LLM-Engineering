import os
from dotenv import load_dotenv
import math
import geopy
import folium
import requests

from langchain_groq import ChatGroq
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

import streamlit as st
from streamlit_folium import st_folium

from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field

from geopy.geocoders import Nominatim

load_dotenv()


# Global Vars

geolocator = Nominatim(user_agent="my_local_ai_app")
llm = ChatGroq(model="openai/gpt-oss-120b")

USER_LAT, USER_LON = 12.9716, 77.5946


st.set_page_config(layout="wide", page_title="AI Local Guide")
st.title("Mapper")

if "final_state" not in st.session_state:
    st.session_state.final_state = None

if "current_query" not in st.session_state:
    st.session_state.current_query = ""

if "map_obj" not in st.session_state:
    st.session_state.map_obj = None

# UI Sidebar

with st.sidebar:
    st.title("Control Panel")
    # user_query = st.text_input("Enter your query:", "Find me some Italian restaurants near MG Road, Bengaluru.")
    location = st.text_input("Enter your location:", "MG Road, Bengaluru")
    radius = st.slider("Select search radius (in km):", 1, 10, 5)
    max_results = st.slider("Select maximum number of results:", 1, 20, 10)
    min_rating = st.slider("Select minimum rating:", 1.0, 5.0, 4.0, step=0.1)



# Pydantic models for structured output

class IntentState(BaseModel):
    location: str = Field(..., description="The location to search for user query.")
    intent: str = Field(..., description="The intent of the user query.")
    is_map: bool = Field(..., description="Whether the user query is related to map or not.")

class KeywordState(BaseModel):
    name: str = Field(..., description="The name of the search keyword.")
    search_term: str = Field(..., description="The type of the search keyword.")
    lat: float = Field(..., description="Latitude of the search keyword.")
    lon: float = Field(..., description="Longitude of the search keyword.")
    rating: float = Field(..., description="Rating of the search keyword.")
    address: str = Field(..., description="The address of the search keyword.")

class AgentState(BaseModel):
    user_query: str = Field(..., description="The user query.")
    final_result: str = Field(default="", description="The final result of the agent.")
    map_results: list[KeywordState] = Field(default_factory=list, description="The map results of the agent.")
    is_map: bool = Field(default=False, description="Whether the user query is related to map or not.")
    is_map_results: bool = Field(default=False, description="Whether the results are map-related.")
    search_term: str = Field(default="", description="The type of the search keyword.")
    user_lat: float = Field(default=0.0, description="The Latitude of user")
    user_lon: float = Field(default=0.0, description="The Longitude of user")


# Get Latitude and Longitude for a location
def get_coords(location_name: str) -> Optional[tuple]:
    """Get latitude and longitude for a given location name."""
    location = geolocator.geocode(location_name)
    if location:
        return (location.latitude, location.longitude)
    return None

# Get Distance
def get_distance_to_restaurant(user_lat, user_lon, dest_lat, dest_lon):
    """Calculates road distance and travel time using Google Routes API."""
    try:
        lon1, lat1, lon2, lat2 = map(math.radians, [user_lon, user_lat, dest_lon, dest_lat])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.sin(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371  # Radius of Earth in kilometers
        aerial_distance = c * r
        estimated_road_distance = round(aerial_distance * 1.25, 1)
        estimated_time = max(2, int(estimated_road_distance * 3))
        return f"~{estimated_road_distance} km ({estimated_time} mins)*"
    except Exception:
        return "Distance unavailable"
    
# Fetch Data from Map0
def fetch_data_from_map(state):
    """Queries the modern Google Places API (New) Text Search endpoint."""
    global radius, max_results, min_rating
    if radius:
        radius = min(radius * 1000, 50000)  # Convert km to meters and cap at 50km
    else:
        radius = 5000  # Default to 5km if no radius provided
    
    if max_results:
        max_results = int(max_results)
        if max_results < 1 or max_results > 20:
            max_results = 10  # Default to 10 if out of bounds
    else:
        max_results = 5  # Default to 10 if out of bounds

    if min_rating:
        try:
            min_rating = float(min_rating)
            if min_rating < 1.0 or min_rating > 5.0:
                min_rating = 3.5  # Default to 3.5 if out of bounds
        except ValueError:
            min_rating = 3.5  # Default to 3.5 if invalid

    search_keyword = state.search_term

    url = "https://places.googleapis.com/v1/places:searchText"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": os.getenv('GOOGLE_MAP_API_KEY'),
        # Field Masking is mandatory in the new API to limit what info you request/bill
        "X-Goog-FieldMask": "places.displayName,places.location,places.rating,places.formattedAddress"
    }
    
        
    # Constructing the structural payload based on Google's modern geofencing requirements
    data = {
        "textQuery": search_keyword,
        "maxResultCount": int(max_results),  # Max capacity value is 20
        "minRating": float(min_rating),
        "locationBias": {
            "circle": {
                "center": {
                    "latitude": state.user_lat,
                    "longitude": state.user_lon
                },
                "radius": float(radius)
            }
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=12)
        if response.status_code == 200:
            results = response.json().get('places', [])
            restaurants = []
            
            for place in results[:15]:
                location = place.get('location', {})
                display_name = place.get('displayName', {}).get('text', 'Unnamed Spot')
                
                restaurants.append({
                    "name": display_name,
                    "search_term": search_keyword,
                    "lat": location.get('latitude'),
                    "lon": location.get('longitude'),
                    "rating": place.get('rating', 'N/A'),
                    "address": place.get('formattedAddress', 'No address available')
                })
            return restaurants
        else:
            print(f"Google Places API (New) Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error querying Google Places API: {e}")
    return []

# Get User Query Intent
def get_user_query_intent(state: AgentState) -> dict:
    prompt = f"""
You are a Intent Agent which checks if the user query is Greetings or question:
User Query: {state.user_query}
If the user query is greetings return True else False
"""
    is_greetings = llm.invoke(prompt)
    if is_greetings.content == "True":
        state.user_lat = 0.0
        state.user_lon = 0.0
        state.search_term = ""
        state.is_map = False
        return state
    llm_with_structured_output = llm.with_structured_output(IntentState)
    result = llm_with_structured_output.invoke(state.user_query)
    # location = result.location
    global location
    print(location)
    print(result)
    USER_LAT, USER_LON = get_coords(location)
    state.user_lat = USER_LAT
    state.user_lon = USER_LON
    state.search_term = result.intent
    state.is_map = result.is_map
    return state


def should_continue(state: AgentState) -> bool:
    is_map = state.is_map
    if is_map:
        return "tools"
    return "conversation"

def llm_response(state: AgentState) -> str:
    is_map = state.is_map
    is_map_results = state.is_map_results
    if is_map and is_map_results:
        prompt = f"""Based on the user's query and the map results, generate a concise and informative response with table.\n\n {state.map_results}"""
        res = llm.invoke(prompt)
        state.final_result = res.content
        return state
    res = llm.invoke(state.user_query)
    state.final_result = res.content
    return state

def get_map_details(state: AgentState) -> List[KeywordState]:
    result = fetch_data_from_map(state)
    structured_results = [KeywordState(**keywords) for keywords in result]
    state.map_results = structured_results
    state.is_map_results = True
    return state


# Workflow

workflow = StateGraph(AgentState)
workflow.add_node("get_user_query_intent", get_user_query_intent)
workflow.add_node("get_map_details", get_map_details)
workflow.add_node("llm_response", llm_response)

workflow.add_edge(START, "get_user_query_intent")
workflow.add_conditional_edges("get_user_query_intent", should_continue, {
    "tools": "get_map_details",
    "conversation": "llm_response"
})
workflow.add_edge("get_map_details", "llm_response")
workflow.add_edge("llm_response", END)

map_search_agent = workflow.compile()

user_query = st.chat_input("Ask me anything about local places, I'll find it on the map for you!")
if user_query:
    st.session_state.current_query = user_query
    initial_state = AgentState(user_query=user_query)
    final_state = map_search_agent.invoke(initial_state)
    st.session_state.final_state = map_search_agent.invoke(initial_state)
    if (
        st.session_state.final_state["is_map"]
        and st.session_state.final_state["is_map_results"]
    ):

        final_state = st.session_state.final_state
        st.subheader("Geospatial Map")
        if final_state['is_map'] and final_state['is_map_results']:
            initial_lat, initial_lon = final_state['user_lat'], final_state['user_lon']
            m = folium.Map(location=[initial_lat, initial_lon], zoom_start=14, tiles="OpenStreetMap")
            folium.Marker(
                [initial_lat, initial_lon], 
                popup="Your Current Center Location", 
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)

            for place in final_state['map_results']:
                folium.Marker(
                    [place.lat, place.lon],
                    popup=f"<b>{place.name}</b><br>Search Term: {place.search_term}",
                    icon=folium.Icon(color="red", icon="cutlery")
                ).add_to(m)
                
            # st_folium(m, width="100%", height=550, key="interactive_leaflet_map")
            st.session_state.map_obj = m
        else:
            st.session_state.map_obj = None
        
        if st.session_state.final_state:

            final_state = st.session_state.final_state

            st.subheader("Response")

            st.markdown(final_state["final_result"])

            if st.session_state.map_obj is not None:

                st.subheader("Geospatial Map")

                st_folium(
                    st.session_state.map_obj,
                    width=900,
                    height=550,
                    key="map",
                )


    st.markdown(final_state['final_result'])