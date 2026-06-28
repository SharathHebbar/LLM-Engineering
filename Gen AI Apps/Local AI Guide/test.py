import math
import streamlit as st
import folium
import requests
from streamlit_folium import st_folium


st.set_page_config(layout="wide", page_title="AI Local Food Guide")

st.title("AI Local Food Guide")


map_data = [{'name': 'Toscano',
  'cuisine': 'italian',
  'lat': 12.9716773,
  'lon': 77.5962738,
  'rating': 4.4,
  'address': '24, Vittal Mallya Rd, KG Halli, Shanthala Nagar, Ashok Nagar, Bengaluru, Karnataka 560001, India'},
 {'name': 'Pinocchio Ristorante',
  'cuisine': 'italian',
  'lat': 12.973023,
  'lon': 77.60626119999999,
  'rating': 4.8,
  'address': '2nd Floor, Forum Rex Walk, Shanthala Nagar, Ashok Nagar, Bengaluru, Karnataka 560001, India'},
 {'name': 'Pasta Street - Cunningham Road',
  'cuisine': 'italian',
  'lat': 12.9883413,
  'lon': 77.5937533,
  'rating': 4.4,
  'address': 'Chicago Avenue, 37/2, Cunningham Rd, opposite Fortis Hospital, Vasanth Nagar, Bengaluru, Karnataka 560001, India'},
 {'name': 'Toscano',
  'cuisine': 'italian',
  'lat': 12.9750347,
  'lon': 77.6029059,
  'rating': 4.2,
  'address': 'Museum Rd, Shanthala Nagar, Ashok Nagar, Bengaluru, Karnataka 560001, India'},
 {'name': 'Sunnys Restaurant',
  'cuisine': 'italian',
  'lat': 12.971848699999999,
  'lon': 77.5985968,
  'rating': 4.1,
  'address': '50, Lavelle Road, Shanthala Nagar, Ashok Nagar, Bengaluru, Karnataka 560001, India'}]

USER_LAT, USER_LON = 12.9716, 77.5946

def get_distance_to_restaurant(user_lat, user_lon, dest_lat, dest_lon):
    """Calculates road distance and travel time using Google Routes API."""
    url = "https://routes.googleapis.com/v1/computeRoutes"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": "AIzaSyCqTjhK3yI2GqzAuvsCmon1HR_UCapCq8w",
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration"
    }
    
    data = {
        "origin": {
            "location": {
                "latLng": {"latitude": float(user_lat), "longitude": float(user_lon)}
            }
        },
        "destination": {
            "location": {
                "latLng": {"latitude": float(dest_lat), "longitude": float(dest_lon)}
            }
        },
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE"
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            route = response.json().get('routes', [{}])[0]
            distance_meters = route.get('distanceMeters', 0)
            duration_str = route.get('duration', '0s')
            
            # Convert meters to kilometers and format time string
            distance_km = round(distance_meters / 1000, 2)
            duration_mins = int(duration_str.replace('s', '')) // 60
            return f"{distance_km} km ({duration_mins} mins)"
        else:
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
    except Exception as e:
        return f"Distance calculation exception: {e}"


# Initialize baseline map centered around user's location
m = folium.Map(location=[USER_LAT, USER_LON], zoom_start=13, tiles="OpenStreetMap")

# Draw your active User Location Pin
folium.Marker(
    [USER_LAT, USER_LON], 
    popup="Your Position", 
    icon=folium.Icon(color="blue", icon="user")
).add_to(m)

# Loop through discovered markers one by one
i = 0
for place in map_data:
    # 1. Fetch live route parameters dynamically
    route_info = get_distance_to_restaurant(USER_LAT, USER_LON, place['lat'], place['lon'])
    print(route_info, i)
    i += 1
    # 2. Add individual restaurant pins with distance metadata embedded in the click popup
    folium.Marker(
        [place['lat'], place['lon']],
        popup=f"<b>{place['name']}</b><br>Rating: {place['rating']}<br>Distance: {route_info}",
        icon=folium.Icon(color="red", icon="cutlery")
    ).add_to(m)
    
    # 3. Draw a visual routing vector line from your location to the restaurant
    folium.PolyLine(
        locations=[[USER_LAT, USER_LON], [place['lat'], place['lon']]],
        color="gray",
        weight=2.5,
        opacity=0.7,
        dash_array="5, 5",
        tooltip=f"Distance to {place['name']}: {route_info}"
    ).add_to(m)
    
# Render the dynamic vector map to your Streamlit front-end layer
st_folium(m, width="100%", height=550, key="interactive_google_backed_map") 