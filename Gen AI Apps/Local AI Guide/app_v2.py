import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
from groq import Groq
import json
import os

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# 1. Page Configuration (Wide Layout for Side-by-Side View)
st.set_page_config(layout="wide", page_title="AI Local Food Guide")

# 2. Initialize Groq Client
try:
    client = Groq()
except Exception as e:
    st.error("Groq API Key missing. Please set the GROQ_API_KEY environment variable.")

# 3. Setup App Session State for Chat History and Geospatial Data
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your AI food guide. I can filter by cuisine or search radius. Try saying: *'Find me pizza places within 3km'*"}
    ]
if "map_data" not in st.session_state:
    st.session_state.map_data = []

# Mock Baseline Coordinates (e.g., Central Bengaluru)
USER_LAT, USER_LON = 12.9716, 77.5946

# 4. OpenStreetMap Overpass API Function
def fetch_osm_restaurants(lat, lon, radius=1500, cuisine=None):
    """Queries OpenStreetMap data dynamically based on coordinates, radius, and cuisine."""
    overpass_url = "https://overpass-api.de/api/interpreter"
    
    # Inject cuisine tag filter if parsed by the LLM
    cuisine_filter = f'["cuisine"="{cuisine.lower()}"]' if cuisine else ""
    
    overpass_query = f"""
    [out:json];
    node["amenity"="restaurant"]{cuisine_filter}(around:{radius},{lat},{lon});
    out body 15;
    """
    try:
        response = requests.get(overpass_url, params={'data': overpass_query}, timeout=12)
        if response.status_code == 200:
            elements = response.json().get('elements', [])
            restaurants = []
            for el in elements:
                tags = el.get('tags', {})
                restaurants.append({
                    "name": tags.get('name', 'Unnamed Spot'),
                    "cuisine": tags.get('cuisine', 'General'),
                    "lat": el.get('lat'),
                    "lon": el.get('lon')
                })
            return restaurants
    except Exception as e:
        st.error(f"Error querying OpenStreetMap: {e}")
    return []

# 5. Define Groq Tool/Function Specifications
tools = [
    {
        "type": "function",
        "function": {
            "name": "fetch_osm_restaurants",
            "description": "Fetches nearby restaurants from OpenStreetMap using radius and optional cuisine parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "integer",
                        "description": "Search radius around the user in meters. Convert text like '3km' to 3000. Default is 1500 if unmentioned."
                    },
                    "cuisine": {
                        "type": "string",
                        "description": "Specific lowercase cuisine requested (e.g., 'italian', 'indian', 'pizza', 'burger'). Leave blank if generic food is requested."
                    }
                },
                "required": []
            }
        }
    }
]

# 6. Streamlit User Interface Layout
st.title("🗺️ Local Open-Source Chatbot (Groq Tools + OpenStreetMap)")
st.write("An entirely open-source approach to local discovery without paid Google Maps dependencies.")

col1, col2 = st.columns([1, 1])

# --- LEFT PANEL: Chat Window ---
with col1:
    st.subheader("Conversation Thread")
    
    # Render persistent message bubbles
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Process User Chat Inputs
    if user_input := st.chat_input("Ask me about food options nearby..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # Phase A: Send prompt to Groq to inspect if Tool Call is necessary
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a local food assistant. Use your tools whenever a user asks to find, locate, or recommend restaurants."},
                    {"role": "user", "content": user_input}
                ],
                tools=tools,
                tool_choice="auto"
            )
            
            response_message = response.choices[0].message
            tool_calls = response_message.tool_calls
            
            # Phase B: Handle Tool execution if requested by Groq
            if tool_calls:
                for tool_call in tool_calls:
                    if tool_call.function.name == "fetch_osm_restaurants":
                        args = json.loads(tool_call.function.arguments)
                        radius = args.get("radius", 1500)
                        cuisine = args.get("cuisine", None)
                        
                        search_msg = f"Fetching {cuisine if cuisine else 'general'} spots within {radius} meters..."
                        message_placeholder.markdown(f"🔄 *{search_msg}*")
                        
                        # Hit OpenStreetMap live data API
                        places = fetch_osm_restaurants(USER_LAT, USER_LON, radius=radius, cuisine=cuisine)
                        st.session_state.map_data = places  
                        
                        # Build Markdown links explicitly with map redirection URLs
                        places_markdown_list = []
                        for p in places:
                            maps_url = f"https://www.google.com/maps/search/?api=1&query={p['lat']},{p['lon']}"
                            places_markdown_list.append(f"### [{p['name']}]({maps_url})\n*Cuisine: {p['cuisine'].title()}*")
                        
                        places_context = "\n\n".join(places_markdown_list)
                        
                        # Enforce the strict output format required using the system prompt
                        system_summary_prompt = f"""You are a helpful food assistant. Present the discovered locations using the EXACT list provided below. 
                        Do not modify or remove the markdown hyperlinked headers. 

                        Format layout constraint:
                        ### [Restaurant Name](Map URL link)
                        Short description of the option (mention its cuisine type and that it is pinned on the visual map panel).

                        Available Entries:
                        {places_context if places_context else "No entries found."}"""
                        
                        # Phase C: Streaming final natural response summary back to the chat bubble
                        final_stream = client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[
                                {"role": "user", "content": user_input},
                                {"role": "assistant", "content": None, "tool_calls": tool_calls},
                                {"role": "tool", "tool_call_id": tool_call.id, "name": "fetch_osm_restaurants", "content": system_summary_prompt}
                            ],
                            stream=True
                        )
                        
                        full_response = ""
                        for chunk in final_stream:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                message_placeholder.markdown(full_response + "▌")
                                
                        message_placeholder.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                        
                        # Rerun app instantly to force the map visualization component to refresh
                        st.rerun()
            else:
                plain_text = response_message.content
                message_placeholder.markdown(plain_text)
                st.session_state.messages.append({"role": "assistant", "content": plain_text})

# --- RIGHT PANEL: Map View Component ---
with col2:
    st.subheader("Geospatial Map")
    
    m = folium.Map(location=[USER_LAT, USER_LON], zoom_start=14, tiles="OpenStreetMap")
    
    # User Location Pin
    folium.Marker(
        [USER_LAT, USER_LON], 
        popup="Your Current Center Location", 
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    
    # Draw discovered restaurants pins
    for place in st.session_state.map_data:
        folium.Marker(
            [place['lat'], place['lon']],
            popup=f"<b>{place['name']}</b><br>Cuisine: {place['cuisine']}",
            icon=folium.Icon(color="red", icon="cutlery")
        ).add_to(m)
        
    st_folium(m, width="100%", height=550, key="interactive_leaflet_map")