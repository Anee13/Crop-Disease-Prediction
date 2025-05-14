# 🌾 Crop Disease Prediction using CNN + MobileNetV2 (Hybrid Model)

This project is a deep learning–based web application that predicts 42 different crop diseases from plant leaf images. It uses a hybrid model architecture combining Convolutional Neural Networks (CNN) with MobileNetV2 and cross-validation techniques to achieve high accuracy and generalizability. The frontend is developed using HTML and CSS, and the backend is powered by Flask.

## ✅ Project Highlights

- 🔬 42-class crop disease classification
- 🤖 Hybrid deep learning model (CNN + MobileNetV2)
- 🔁 Integrated cross-validation for performance consistency
- 🌐 Web interface using Flask + HTML/CSS
- 📊 Achieved 92% validation accuracy

---

## 🧠 Model Overview

The model combines the strengths of:

- 📦 CNN for spatial feature extraction
- 📱 MobileNetV2 for lightweight yet efficient classification
- 🔁 K-Fold cross-validation to avoid overfitting and validate performance

---

## 📁 Dataset

- Source: 20k+ Multi-Class Crop Disease Images Dataset
- Classes: 42 disease categories across different crops
- Preprocessing: Image resizing, normalization, and data augmentation

---

## 💻 Tech Stack

| Layer         | Technology     |
| ------------- | -------------- |
| Frontend      | HTML, CSS      |
| Backend       | Flask (Python) |
| Deep Learning | Keras, TensorFlow |
| Model Type    | CNN + MobileNetV2 Hybrid |
| Deployment    | Localhost (can be Dockerized) |

---

## 🚀 How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/crop-disease-prediction
cd crop-disease-prediction
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the App

```bash
python app.py
```

The app will be available at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📸 Usage

1. Upload a clear image of a crop leaf.
2. Click "Predict".
3. The system displays:

   * Original and enhanced image
   * Predicted disease class
   * Suggested remedies

---

## 📈 Results

* Training Accuracy: \~95%
* Validation Accuracy: \~92%
* Cross-validation folds: 5
* Optimizer: Adam
* Loss Function: Categorical Crossentropy

---

## 🛠 Future Improvements

* Integrate Hugging Face image enhancement API
* Deploy using Docker + FastAPI for production
* Integrate IoT device for field-level prediction delivery
* Include voice/image-based input for farmers

---

## 🙋‍♂️ Author

Developed by: Aneesh K
Role: Data Science & AI Trainer | IT Engineer

---

