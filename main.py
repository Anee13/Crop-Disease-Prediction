from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.image_processor import preprocess_image
import shutil
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model = load_model("model/crop_disease_fold1.keras")

# Your class list and remedy_info dict here...
classes = [
    "American Bollworm on Cotton", "Anthracnose on Cotton", "Army worm", "Becterial Blight in Rice",
    "Brownspot", "Common_Rust", "Cotton Aphid", "Flag Smut", "Gray_Leaf_Spot", "Healthy Maize",
    "Healthy Wheat", "Healthy cotton", "Leaf Curl", "Leaf smut", "Mosaic sugarcane", "RedRot sugarcane",
    "RedRust sugarcane", "Rice Blast", "Sugarcane Healthy", "Tungro", "Wheat Brown leaf Rust",
    "Wheat Stem fly", "Wheat aphid", "Wheat black rust", "Wheat leaf blight", "Wheat mite",
    "Wheat powdery mildew", "Wheat scab", "Wheat___Yellow_Rust", "Wilt", "Yellow Rust Sugarcane",
    "bacterial_blight in Cotton", "bollrot on Cotton", "bollworm on Cotton", "cotton mealy bug",
    "cotton whitefly", "maize ear rot", "maize fall armyworm", "maize stem borer",
    "pink bollworm in cotton", "red cotton bug", "thirps on cotton"
]

remedy_info = {
    "American Bollworm on Cotton": "Use biological control agents like Trichogramma. Apply neem-based pesticides.",
    "Anthracnose on Cotton": "Use resistant varieties and fungicides like Mancozeb.",
    "Army worm": "Spray with insecticides like Chlorpyrifos or Bacillus thuringiensis.",
    "Becterial Blight in Rice": "Use certified seeds and avoid water stagnation. Apply copper-based bactericides.",
    "Brownspot": "Apply potash and use fungicides like Carbendazim.",
    "Common_Rust": "Use resistant hybrids and fungicides like Propiconazole.",
    "Cotton Aphid": "Use neem oil spray or Imidacloprid.",
    "Flag Smut": "Seed treatment with Thiram. Use crop rotation.",
    "Gray_Leaf_Spot": "Use resistant maize varieties and proper fungicide application.",
    "Healthy Maize": "No action needed — the crop is healthy!",
    "Healthy Wheat": "No action needed — the crop is healthy!",
    "Healthy cotton": "No action needed — the crop is healthy!",
    "Leaf Curl": "Control whiteflies, and use resistant cotton varieties.",
    "Leaf smut": "Spray with Dithane M-45 or Mancozeb.",
    "Mosaic sugarcane": "Remove infected plants and control aphid transmission.",
    "RedRot sugarcane": "Plant disease-free setts, and use fungicide dips.",
    "RedRust sugarcane": "Improve drainage and use sulphur-based fungicides.",
    "Rice Blast": "Maintain field hygiene, and apply Tricyclazole or Carbendazim.",
    "Sugarcane Healthy": "No action needed — the crop is healthy!",
    "Tungro": "Use resistant varieties and control vector insects.",
    "Wheat Brown leaf Rust": "Use rust-resistant varieties and fungicides.",
    "Wheat Stem fly": "Use appropriate insecticides during early growth stages.",
    "Wheat aphid": "Apply insecticides like Dimethoate or Imidacloprid.",
    "Wheat black rust": "Use resistant varieties and avoid late sowing.",
    "Wheat leaf blight": "Apply foliar sprays of Mancozeb or Chlorothalonil.",
    "Wheat mite": "Use acaricides like Dicofol.",
    "Wheat powdery mildew": "Use sulfur dust or systemic fungicides.",
    "Wheat scab": "Avoid overhead irrigation; use fungicides at heading stage.",
    "Wheat___Yellow_Rust": "Use resistant varieties and Propiconazole sprays.",
    "Wilt": "Improve soil drainage and use fungicide dips before sowing.",
    "Yellow Rust Sugarcane": "Use clean planting material and remove infected plants.",
    "bacterial_blight in Cotton": "Use resistant varieties and copper-based sprays.",
    "bollrot on Cotton": "Destroy infected bolls and apply Carbendazim.",
    "bollworm on Cotton": "Spray Spinosad or Bacillus thuringiensis.",
    "cotton mealy bug": "Use neem oil or systemic insecticides like Imidacloprid.",
    "cotton whitefly": "Control with yellow sticky traps and sprays.",
    "maize ear rot": "Harvest timely and dry grains quickly.",
    "maize fall armyworm": "Use pheromone traps and apply Spinosad or Chlorantraniliprole.",
    "maize stem borer": "Destroy stubbles and apply Trichogramma egg cards.",
    "pink bollworm in cotton": "Use pheromone traps and Bt cotton hybrids.",
    "red cotton bug": "Spray with Malathion or Neem-based insecticides.",
    "thirps on cotton": "Use blue sticky traps and neem extracts or Spinetoram."
}

@app.post("/", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def index(request: Request, file: UploadFile = None):
    prediction = None
    remedy = None
    image_path = None

    if file:
        image_path = f"static/uploads/{file.filename}"
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        image = preprocess_image(image_path)
        result = model.predict(image)
        class_index = np.argmax(result)
        prediction = classes[class_index]
        remedy = remedy_info.get(prediction, "Remedy not available.")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": prediction,
        "remedy": remedy,
        "image_path": image_path
    })
