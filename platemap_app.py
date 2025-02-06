import os
import streamlit as st
import wikipedia
import requests
from bs4 import BeautifulSoup
import pandas as pd
import cv2
from google.cloud import vision
import openai
from keplergl import KeplerGl
from geopy.geocoders import Nominatim
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

# Load API keys and credentials
openai.api_key = os.getenv("OPENAI_API_KEY")  # OpenAI API Key
usda_api_key = os.getenv("USDA_API_KEY")  # USDA API Key
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_PATH")  # Google Vision JSON path

# Initialize tools
geolocator = Nominatim(user_agent="platemap")
wikipedia.set_lang("en")
geocode_cache = {}

# Helper Functions
def resize_image(image_path, max_width=800):
    """Resize images to reduce processing time."""
    image = cv2.imread(image_path)
    if image.shape[1] > max_width:
        scaling_factor = max_width / image.shape[1]
        image = cv2.resize(image, (int(image.shape[1] * scaling_factor), int(image.shape[0] * scaling_factor)))
        cv2.imwrite(image_path, image)

@lru_cache(maxsize=50)
def cached_geocode(location):
    """Cache geocoded results for faster processing."""
    if location not in geocode_cache:
        geocode_cache[location] = geolocator.geocode(location)
    return geocode_cache[location]

@lru_cache(maxsize=50)
def get_food_origin_coordinates_cached(food_name):
    """Fetch origin data from Wikipedia and FoodAtlas."""
    origins = []  # List to store origin data
    
    # Wikipedia origin
    try:
        wiki_summary = wikipedia.summary(food_name, sentences=2)
        wiki_location = cached_geocode(food_name)
        if wiki_location:
            origins.append((f"Wikipedia: {wiki_summary}", wiki_location.latitude, wiki_location.longitude))
    except:
        pass
    
    # FoodAtlas origin
    try:
        url = f"https://www.tasteatlas.com/{food_name.lower().replace(' ', '-')}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            description = soup.find("meta", {"name": "description"})
            location_text = description["content"] if description else f"No specific origins found for {food_name}"
            location = cached_geocode(location_text.split(",")[0])
            if location:
                origins.append((location_text, location.latitude, location.longitude))
    except:
        pass

    if not origins:
        origins.append(("Unknown origin", 0, 0))  # Default if no origins found
    return origins

@lru_cache(maxsize=50)
def get_nutritional_data_cached(food_name):
    """Fetch nutritional data from USDA."""
    url = f"https://api.nal.usda.gov/fdc/v1/foods/search?query={food_name}&api_key={usda_api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data.get("foods"):
            food = data["foods"][0]
            nutrients = {
                "Calories": food.get("foodNutrients", [{}])[0].get("value", "N/A"),
                "Protein": food.get("foodNutrients", [{}])[1].get("value", "N/A"),
                "Fat": food.get("foodNutrients", [{}])[2].get("value", "N/A"),
                "Carbs": food.get("foodNutrients", [{}])[3].get("value", "N/A"),
            }
            return nutrients
    return f"No nutritional data for {food_name}."

@lru_cache(maxsize=50)
def generate_quirky_fact_cached(food_name):
    """Generate quirky facts using OpenAI."""
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Tell me a quirky fact about {food_name}.",
            max_tokens=30  # Reduced for efficiency
        )
        return response.choices[0].text.strip()
    except:
        return "Could not generate a quirky fact."

# Streamlit UI
st.title("🍽️ PlateMap: Explore Your Food's Story!")
uploaded_file = st.file_uploader("Upload an image of your plate", type=["jpg", "png"])
user_location = st.text_input("Where are you eating this? (City or State)", placeholder="E.g., Boston, Massachusetts")

if uploaded_file and user_location:
    with st.spinner("Processing..."):
        # Save and resize the image
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.read())
        resize_image("uploaded_image.jpg")
        
        # Detect food items
        detected_foods, annotated_image = detect_food_from_image("uploaded_image.jpg")
        st.image(annotated_image, caption="Detected Food Items", use_column_width=True)

        for food in detected_foods:
            # Fetch data in parallel
            with ThreadPoolExecutor() as executor:
                origin_data, nutrition_data, quirky_fact = executor.map(
                    lambda func: func(food['name']),
                    [get_food_origin_coordinates_cached, get_nutritional_data_cached, generate_quirky_fact_cached]
                )

            st.subheader(f"Dish: {food['name']} (Confidence: {food['confidence']:.2f})")
            st.markdown(f"**Origin:** {origin_data[0][0]}")
            st.json(nutrition_data)
            st.markdown(f"**Quirky Fact:** {quirky_fact}")

        # Display map
        st.markdown("## 🌍 Map")
        kepler_map = visualize_with_kepler(detected_foods[0]['name'], user_location)
        with open(kepler_map, "rb") as f:
            st.download_button(label="Download Map", data=f, file_name="kepler_map.html", mime="text/html")
