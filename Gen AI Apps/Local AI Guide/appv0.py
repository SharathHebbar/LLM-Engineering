import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Ensure the API key is set in the environment

# Set up page layout to wide for the side-by-side view
st.set_page_config(layout="wide")

# Initialize Groq Client
# It automatically picks up the GROQ_API_KEY environment variable
try:
    client = Groq()
except Exception as e:
    st.error("Groq API Key missing. Please set the GROQ_API_KEY environment variable.")

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! Tell me what kind of food you're craving near central Bengaluru."}]
if "map_data" not in st.session_state:
    st.session_state.map_data = []

# Mock Coordinates for user location (e.g., Central Bengaluru)
USER_LAT, USER_LON = 12.98474, 77.54906

def fetch_osm_restaurants(lat, lon, radius=1500):
    """Fetches real-time restaurant data from OpenStreetMap via Overpass API"""
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    node["amenity"="restaurant"](around:{radius},{lat},{lon});
    out body 12;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=10)
        if response.status_code == 200:
            elements = response.json().get('elements', [])
            restaurants = []
            for el in elements:
                tags = el.get('tags', {})
                restaurants.append({
                    "name": tags.get('name', 'Unnamed Restaurant'),
                    "cuisine": tags.get('cuisine', 'Local/General'),
                    "lat": el.get('lat'),
                    "lon": el.get('lon')
                })
            return restaurants
    except Exception as e:
        st.error(f"Error fetching spatial data: {e}")
    return []

# --- UI LAYOUT ---
st.title("🗺️ Open-Source Local Food Bot (Powered by Groq)")

col1, col2 = st.columns([1, 1])

# LEFT COLUMN: Chat Engine
with col1:
    st.subheader("Chat")
    
    # Render historical chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Capture Input
    if user_input := st.chat_input("Ask for restaurants..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # 1. Query Local Open Data
        raw_places = fetch_osm_restaurants(USER_LAT, USER_LON)
        st.session_state.map_data = raw_places  # Update map state to refresh UI pins
        
        # 2. Format Context for Groq
        places_context = "\n".join([f"- {p['name']} (Cuisine: {p['cuisine']})" for p in raw_places])
        
        system_prompt = f"""You are a helpful local food guide. Present these nearby options cleanly to the user. 
        Mention that they have been marked on their map. Keep it concise, friendly, and structured.
        
        Available Options from OpenStreetMap:
        {places_context}"""
        
        # 3. Stream Inference via Groq
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Utilizing Llama 3 on Groq infrastructure for ultra-low latency streaming
                completion = client.chat.completions.create(
                    model="openai/gpt-oss-120b",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    stream=True
                )
                
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                error_fallback = f"Failed to generate text from Groq: {e}\n\nHere are the raw results:\n{places_context}"
                message_placeholder.markdown(error_fallback)
                st.session_state.messages.append({"role": "assistant", "content": error_fallback})

# RIGHT COLUMN: Map Engine
with col2:
    st.subheader("Interactive Map")
    
    # Initialize baseline map centered around user
    m = folium.Map(location=[USER_LAT, USER_LON], zoom_start=14, tiles="OpenStreetMap")
    
    # Blue pin for user's location
    folium.Marker(
        [USER_LAT, USER_LON], 
        popup="Your Location", 
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    
    # Red pins for discovered restaurants
    for place in st.session_state.map_data:
        folium.Marker(
            [place['lat'], place['lon']],
            popup=f"<b>{place['name']}</b><br>Cuisine: {place['cuisine']}",
            icon=folium.Icon(color="red", icon="cutlery")
        ).add_to(m)
        
    # Render the map object inside Streamlit
    st_folium(m, width="100%", height=500, key="main_map")