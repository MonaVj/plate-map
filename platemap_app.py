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

try:
    with open("test_file.txt", "w") as f:
        f.write("Test successful!")
    st.success("File write test passed!")
except Exception as e:
    st.error(f"File write test failed: {e}")


# Set up Google API credentials
google_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_PATH")
if google_credentials:
    with open("google_credentials.json", "w") as f:
        f.write(google_credentials)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google_credentials.json"
else:
    st.error("Google Vision credentials not found. Check your environment variables.")

# Load API keys securely
openai.api_key = os.getenv("OPENAI_API_KEY")
usda_api_key = os.getenv("USDA_API_KEY")

# Initialize tools
geolocator = Nominatim(user_agent="platemap")
wikipedia.set_lang("en")

# Detect and annotate food in image
def detect_food_from_image(image_path):
    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as image_file:
        content = image_file.read()
        image = vision.Image(content=content)

    response = client.object_localization(image=image)
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

# Fetch food origin from Wikipedia and FoodAtlas
@lru_cache(maxsize=50)
def get_food_origin_coordinates(food_name):
    origins = []

    # Wikipedia origin
    try:
        wiki_summary = wikipedia.summary(food_name, sentences=2)
        wiki_location = geolocator.geocode(food_name)
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

# Generate quirky fact using OpenAI
@lru_cache(maxsize=50)
def generate_quirky_fact(food_name):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Tell me a quirky fact about {food_name}.",
            max_tokens=30
        )
        return response.choices[0].text.strip()
    except:
        return "Could not generate a quirky fact."

# Visualize data with KeplerGl
def visualize_with_kepler(food_name, user_location):
    user_coords = geolocator.geocode(user_location)
    food_origins = get_food_origin_coordinates(food_name)
    
    data = pd.DataFrame(
        [{"Name": "User Location", "Latitude": user_coords.latitude, "Longitude": user_coords.longitude, "Type": "Consumption Place"}] +
        [{"Name": origin[0], "Latitude": origin[1], "Longitude": origin[2], "Type": "Food Origin"} for origin in food_origins]
    )
    
    kepler_map = KeplerGl(height=600)
    kepler_map.add_data(data=data, name="Food Network")
    kepler_map.save_to_html(file_name="kepler_map.html")
    return "kepler_map.html"

# Streamlit UI
st.title("🍽️ PlateMap: Explore Your Food's Story!")
uploaded_file = st.file_uploader("Upload an image of your plate", type=["jpg", "png"])
user_location = st.text_input("Where are you eating this? (City or State)", placeholder="E.g., Boston, Massachusetts")

if uploaded_file and user_location:
    with st.spinner("Processing..."):
        with open("uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.read())
        
        detected_foods, annotated_image = detect_food_from_image("uploaded_image.jpg")
        st.image(annotated_image, caption="Detected Food Items", use_column_width=True)

        for food in detected_foods:
            st.subheader(f"Dish: {food['name']} (Confidence: {food['confidence']:.2f})")
            st.markdown(f"**Wikipedia:** {get_food_origin_coordinates(food['name'])[0][0]}")
            st.markdown("**Nutritional Data:**")
            st.json(get_nutritional_data(food["name"]))
            st.markdown(f"**Quirky Fact:** {generate_quirky_fact(food['name'])}")

        st.markdown("## 🌍 Map")
        kepler_map = visualize_with_kepler(detected_foods[0]['name'], user_location)
        with open(kepler_map, "rb") as f:
            st.download_button(label="Download Map", data=f, file_name="kepler_map.html", mime="text/html")
