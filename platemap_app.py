import os
import json
import streamlit as st
import wikipedia
import requests
from bs4 import BeautifulSoup
import pandas as pd
import cv2
from google.cloud import vision
from google.oauth2 import service_account
from functools import lru_cache
from keplergl import KeplerGl
from geopy.geocoders import Nominatim

# Load API keys securely from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
usda_api_key = os.getenv("USDA_API_KEY")
google_credentials_json = os.getenv("GOOGLE_CREDENTIALS_JSON")

# Validate API keys and credentials
if not openai_api_key or not usda_api_key or not google_credentials_json:
    st.error("One or more required API keys are missing. Please check GitHub Secrets.")
    st.stop()

# Parse Google Vision API credentials
try:
    google_credentials = json.loads(google_credentials_json)
    credentials = service_account.Credentials.from_service_account_info(google_credentials)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
except Exception as e:
    st.error(f"Error initializing Google Vision API: {e}")
    st.stop()

# Initialize tools
geolocator = Nominatim(user_agent="platemap")
wikipedia.set_lang("en")

# Detect and annotate food in an image
def detect_food_from_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            content = image_file.read()
            image = vision.Image(content=content)

        response = vision_client.object_localization(image=image)
        detected_foods = []
        image_cv = cv2.imread(image_path)

        for obj in response.localized_object_annotations:
            if obj.name.lower() in ["food", "dish", "plate"]:
                vertices = [
                    (int(v.x * image_cv.shape[1]), int(v.y * image_cv.shape[0]))
                    for v in obj.bounding_poly.normalized_vertices
                ]
                detected_foods.append({"name": obj.name, "confidence": obj.score, "vertices": vertices})
                cv2.rectangle(image_cv, vertices[0], vertices[2], (0, 255, 0), 2)
                cv2.putText(image_cv, obj.name, (vertices[0][0], vertices[0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = "annotated_image.jpg"
        cv2.imwrite(output_path, image_cv)
        return detected_foods, output_path
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return [], None

# Fetch food origin information
@lru_cache(maxsize=50)
def get_food_origin_coordinates(food_name):
    origins = []

    try:
        # Wikipedia origin
        wiki_summary = wikipedia.summary(food_name, sentences=2)
        wiki_location = geolocator.geocode(food_name)
        if wiki_location:
            origins.append((f"Wikipedia: {wiki_summary}", wiki_location.latitude, wiki_location.longitude))
    except:
        pass

    try:
        # FoodAtlas origin
        url = f"https://www.tasteatlas.com/{food_name.lower().replace(' ', '-')}"
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            description = soup.find("meta", {"name": "description"})
            location_text = description["content"] if description else f"No specific origins found for {food_name}"
            location = geolocator.geocode(location_text.split(",")[0])
            if location:
                origins.append((location_text, location.latitude, location.longitude))
    except:
        pass

    if not origins:
        origins.append(("Unknown origin", 0, 0))

    return origins

# Fetch nutritional data from USDA
@lru_cache(maxsize=50)
def get_nutritional_data(food_name):
    try:
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
    except:
        pass

    return f"No nutritional data for {food_name}."

# Streamlit UI
st.title("🍽️ PlateMap: Explore Your Food's Story!")
uploaded_file = st.file_uploader("Upload an image of your plate", type=["jpg", "png"])
user_location = st.text_input("Where are you eating this? (City or State)", placeholder="E.g., Boston, Massachusetts")

if uploaded_file and user_location:
    with st.spinner("Processing..."):
        # Save uploaded file
        temp_image_path = "uploaded_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.read())

        # Detect food items
        detected_foods, annotated_image = detect_food_from_image(temp_image_path)
        if annotated_image:
            st.image(annotated_image, caption="Detected Food Items", use_column_width=True)

        for food in detected_foods:
            st.subheader(f"Dish: {food['name']} (Confidence: {food['confidence']:.2f})")
            st.markdown(f"**Wikipedia:** {get_food_origin_coordinates(food['name'])[0][0]}")
            st.markdown("**Nutritional Data:**")
            st.json(get_nutritional_data(food["name"]))

        os.remove(temp_image_path)
