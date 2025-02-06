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

# Load API keys and credentials from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_PATH")
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

# Fetch data from APIs
def get_food_origin_from_wikipedia(food_name):
    try:
        summary = wikipedia.summary(food_name, sentences=2)
        return summary
    except:
        return f"No information found for {food_name} on Wikipedia."

def get_food_origin_from_foodatlas(food_name):
    url = f"https://www.tasteatlas.com/{food_name.lower().replace(' ', '-')}"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        description = soup.find("meta", {"name": "description"})
        return description["content"] if description else f"No description for {food_name} on FoodAtlas."
    return f"No data for {food_name} on FoodAtlas."

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

def generate_quirky_fact(food_name):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Tell me a quirky fact about {food_name}.",
            max_tokens=50
        )
        return response.choices[0].text.strip()
    except:
        return "Could not generate a quirky fact."

def visualize_with_kepler(food_name, user_location):
    user_coords = geolocator.geocode(user_location)
    data = pd.DataFrame([
        {"Name": food_name, "Latitude": user_coords.latitude, "Longitude": user_coords.longitude, "Type": "User Location"},
        {"Name": "India", "Latitude": 20.5937, "Longitude": 78.9629, "Type": "Food Origin"}
    ])
    kepler_map = KeplerGl(height=600)
    kepler_map.add_data(data=data, name="Food Network")
    kepler_map.save_to_html(file_name="kepler_map.html")
    return "kepler_map.html"

# Streamlit UI
st.title("🍽️ PlateMap: Explore Your Food's Story!")
uploaded_file = st.file_uploader("Upload an image of your plate", type=["jpg", "png"])
if uploaded_file:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.read())
    detected_foods, annotated_image = detect_food_from_image("uploaded_image.jpg")
    st.image(annotated_image, caption="Detected Food Items", use_column_width=True)

    for food in detected_foods:
        st.subheader(f"Dish: {food['name']} (Confidence: {food['confidence']:.2f})")
        st.markdown(f"**Wikipedia:** {get_food_origin_from_wikipedia(food['name'])}")
        st.markdown(f"**FoodAtlas:** {get_food_origin_from_foodatlas(food['name'])}")
        st.markdown("**Nutritional Data:**")
        st.json(get_nutritional_data(food["name"]))
        st.markdown(f"**Quirky Fact:** {generate_quirky_fact(food['name'])}")

    st.markdown("## 🌍 Map")
    kepler_map = visualize_with_kepler(detected_foods[0]['name'], "Boston")
    with open(kepler_map, "rb") as f:
        st.download_button(label="Download Map", data=f, file_name="kepler_map.html", mime="text/html")
