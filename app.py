import streamlit as st
from PIL import Image
import cv2
import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from car_data_handler import get_car_info
from market_trends import integrate_market_trends_button  # Import the new module
from dotenv import load_dotenv
load_dotenv()
# Constants
MODEL_PATH = "car_model_classifier/resnet_car_model.pth"
LABELS_PATH = "car_model_classifier/labels.json"
TEMP_IMAGE_PATH = "temp.jpg"

# Set page config with custom styling
st.set_page_config(
    page_title="Car Model & Price Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .car-info-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-left: 5px solid #4ECDC4;
    }
    
    .info-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.8rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    
    .info-label {
        font-weight: bold;
        color: #555;
        font-size: 1.1rem;
    }
    
    .info-value {
        color: #333;
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .price-highlight {
        background: linear-gradient(90deg, #56CCF2, #2F80ED);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .status-analyzing {
        background: linear-gradient(90deg, #ffeaa7, #fdcb6e);
        color: #2d3436;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🚗 Car Model & Price Detector</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎯 Choose Input Method")
st.sidebar.markdown("---")
option = st.sidebar.radio(
    "Select how you want to provide the car image:",
    ["📤 Upload Image", "📷 Use Webcam"],
    index=0
)

# Check for Gemini API key
if not os.getenv('GEMINI_API_KEY'):
    st.sidebar.warning("⚠️ Gemini API key not found. Market trends feature will be limited.")
    st.sidebar.info("💡 Add GEMINI_API_KEY to your .env file to enable AI market analysis.")

# Load label map
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)
idx_to_class = {int(k): v for k, v in label_map.items()}

# Load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(label_map))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model

model = load_model()

# Image Preprocessing
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# Predict function
def predict(image_path):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = idx_to_class[predicted.item()]
    return predicted_class

# Function to display car info beautifully
def display_car_info(car_data):
    if not car_data:
        st.warning("🔍 No matching car data found in our database.")
        return
    
    # Debug: Show what data we received
    with st.expander("🔍 Debug: Show Raw Data", expanded=False):
        st.write("Data type:", type(car_data))
        st.write("Data keys:", list(car_data.keys()) if isinstance(car_data, dict) else "Not a dictionary")
        st.json(car_data)
    
    # Handle different data formats
    if isinstance(car_data, list) and len(car_data) > 0:
        car_data = car_data[0]  # Take first item if it's a list
    
    # Try different possible price key names
    price_keys = ['Average_Price_(₹)', 'Average Price (₹)', 'Average Price ($)', 'price', 'Price', 'average_price', 'Average_Price']
    price_value = 'N/A'
    
    for key in price_keys:
        if key in car_data and car_data[key] is not None:
            price_value = car_data[key]
            break
    
    # Format price if it exists
    if price_value != 'N/A':
        try:
            # Remove commas and convert to number for formatting
            if isinstance(price_value, str):
                price_clean = price_value.replace(',', '').replace('₹', '').strip()
                price_num = float(price_clean)
                price_value = f"{price_num:,.0f}"
            elif isinstance(price_value, (int, float)):
                price_value = f"{price_value:,.0f}"
        except:
            pass  # Keep original value if formatting fails
    
    # Create columns for better layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="car-info-card">
            <h3 style="color: #4ECDC4; margin-bottom: 1rem;">🚘 Vehicle Details</h3>
            <div class="info-row">
                <span class="info-label">🏷️ Model:</span>
                <span class="info-value">{car_data.get('Model', car_data.get('model', 'N/A'))}</span>
            </div>
            <div class="info-row">
                <span class="info-label">📅 Year:</span>
                <span class="info-value">{car_data.get('Year', car_data.get('year', 'N/A'))}</span>
            </div>
            <div class="info-row">
                <span class="info-label">⛽ Fuel Type:</span>
                <span class="info-value">{car_data.get('Fuel', car_data.get('fuel', 'N/A'))}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="car-info-card">
            <h3 style="color: #4ECDC4; margin-bottom: 1rem;">⚙️ Technical Specs</h3>
            <div class="info-row">
                <span class="info-label">🔧 Transmission:</span>
                <span class="info-value">{car_data.get('Transmission', car_data.get('transmission', 'N/A'))}</span>
            </div>
            <div class="info-row">
                <span class="info-label">🚗 Drive Type:</span>
                <span class="info-value">{car_data.get('Drive', car_data.get('drive', 'N/A'))}</span>
            </div>
            <div class="info-row">
                <span class="info-label">🛣️ KM Driven:</span>
                <span class="info-value">{car_data.get('KM_Driven', car_data.get('km_driven', 'N/A'))}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Price section - full width
    st.markdown(f"""
    <div class="car-info-card" style="text-align: center;">
        <h3 style="color: #4ECDC4; margin-bottom: 1rem;">💰 Pricing Information</h3>
        <div style="padding: 1rem;">
            <span class="price-highlight">
                Average Price: ₹{price_value}
            </span>
        </div>
        <p style="color: #666; margin-top: 1rem; font-style: italic;">
            *Price is based on market analysis and may vary depending on condition and location
        </p>
    </div>
    """, unsafe_allow_html=True)

# Handle file upload
# Initialize uploaded_file to avoid NameError
uploaded_file = None

# Track image upload/capture state
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = False

# Handle file upload
if option == "📤 Upload Image":
    st.subheader("📤 Upload Your Car Image")
    uploaded_file = st.file_uploader(
        "Choose a car image file", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the car for best results"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(TEMP_IMAGE_PATH)
        st.session_state.image_uploaded = True
        st.success("✅ Image successfully uploaded!")

# Handle webcam input
elif option == "📷 Use Webcam":
    st.subheader("📷 Capture from Webcam")
    picture = st.camera_input("Capture a car image")

    if picture:
        image = Image.open(picture)
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(TEMP_IMAGE_PATH)
        st.session_state.image_uploaded = True
        st.success("✅ Image captured successfully!")

# Show image preview and removal option
if st.session_state.image_uploaded and os.path.exists(TEMP_IMAGE_PATH):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(TEMP_IMAGE_PATH, caption="🖼️ Selected Car Image", use_container_width=True)

    col_remove, col_space = st.columns([1, 4])
    with col_remove:
        if st.button("🗑️ Remove Image"):
            os.remove(TEMP_IMAGE_PATH)
            st.session_state.image_uploaded = False
            st.success("✅ Image removed. You can upload or capture a new one.")
            st.rerun()



# Prediction and data display
if os.path.exists(TEMP_IMAGE_PATH) and st.button("🔍 Analyze Car", type="primary", use_container_width=True):
    # Show analyzing status
    st.markdown("""
    <div class="status-analyzing">
        🤖 AI is analyzing your car image... Please wait!
    </div>
    """, unsafe_allow_html=True)
    
    try:
        with st.spinner("🔄 Processing image and predicting car model..."):
            predicted_model = predict(TEMP_IMAGE_PATH)
        
        # Display prediction result
        st.markdown(f"""
        <div class="prediction-card">
            <h2>🎯 Prediction Result</h2>
            <h1 style="margin: 1rem 0; font-size: 2.5rem;">{predicted_model}</h1>
            <p>Our AI model has identified your car!</p>
        </div>
        """, unsafe_allow_html=True)

        # Get and display car information
        st.markdown("---")
        st.subheader("📊 Detailed Car Information")
        
        with st.spinner("🔍 Fetching detailed car information..."):
            car_info_df = get_car_info(predicted_model)
        
        display_car_info(car_info_df)
        
        # Additional features
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Analyze Another Image", use_container_width=True):
                if os.path.exists(TEMP_IMAGE_PATH):
                    os.remove(TEMP_IMAGE_PATH)
                st.rerun()
        
        with col2:
            # This is where we integrate the market trends
            integrate_market_trends_button(predicted_model, car_info_df)

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.info("💡 Try uploading a clearer image or contact support if the issue persists.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🚗 Car Model & Price Detector | Powered by AI 🤖</p>
</div>
""", unsafe_allow_html=True)

