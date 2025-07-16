# 🚗 Car Detector & Model Classifier

This project detects cars in images and classifies their model using a ResNet-based deep learning model. Built with Python, Streamlit, and integrated with Kaggle datasets for training.

---
## 📸 Screenshots

### 🔍 Car Detection Interface

![Screenshot 1](Screenshots/Screenshot%202025-06-22%20095326.png)

### 🧠 Model Classification Output

![Screenshot 2](Screenshots/Screenshot%202025-06-22%20095334.png)

### 📊 Market Trend Visualization

![Screenshot 3](Screenshots/Screenshot%202025-06-22%20095343.png)

---

## ⚙️ Features

- 🔍 Car detection (via image upload)
- 🧠 Car model classification using a trained CNN (ResNet)
- 📊 Market trend visualization
- 📁 Dataset handling with Kaggle API

---

## 🗂️ Project Structure

- `app.py` – Main Streamlit app
- `car_model_classifier/` – Model & label logic
- `Cars Dataset/` – Raw car image dataset
- `train.py` – Training script for model
- `market_trends.py` – Market analysis (optional)
- `.kaggle/` – API key (private)
- `.streamlit/` – Streamlit configuration

---

## 🚀 Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/CAR-DETECTOR.git
cd CAR-DETECTOR
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
