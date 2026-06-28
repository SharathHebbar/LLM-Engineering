import os
import 
from langchain_core.tools import tool


# 4. Modern Google Places API (New) Text Search function
@tool
def fetch_google_restaurants_new(lat, lon, radius=1500, cuisine=None, max_results=5, min_rating=3.5):
    """Queries the modern Google Places API (New) Text Search endpoint."""
    url = "https://places.googleapis.com/v1/places:searchText"
    
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": os.getenv('GOOGLE_MAP_API_KEY'),
        # Field Masking is mandatory in the new API to limit what info you request/bill
        "X-Goog-FieldMask": "places.displayName,places.location,places.rating,places.formattedAddress"
    }
    
    text_query = f"{cuisine} restaurant" if cuisine else "restaurant"
    
    # Constructing the structural payload based on Google's modern geofencing requirements
    data = {
        "textQuery": text_query,
        "maxResultCount": int(max_results),  # Max capacity value is 20
        "minRating": float(min_rating),
        "locationBias": {
            "circle": {
                "center": {
                    "latitude": lat,
                    "longitude": lon
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
                    "cuisine": cuisine if cuisine else "Restaurant",
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