from flask import Flask, render_template, request
import numpy as np
import tensorflow
from utils.image_processor import preprocess_image

app = Flask(__name__)
model = tensorflow.keras.load_model("crop_disease_fold1.keras")

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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = 'static/uploads/' + file.filename
            file.save(image_path)

            image = preprocess_image(image_path)
            prediction = model.predict(image)
            class_index = np.argmax(prediction)
            disease_name = classes[class_index]
            remedy = remedy_info.get(disease_name, "Remedy not available. Please consult a local agronomist.")

            return render_template('index.html', prediction=disease_name, remedy=remedy, image_path=image_path)

    return render_template('index.html')