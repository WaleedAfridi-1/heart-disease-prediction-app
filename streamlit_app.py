import streamlit as st
import numpy as np
import pickle
import time
from xgboost import XGBClassifier 

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS Styling with Animations
# ---------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
        }
        
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .card {
            background-color: #ffffff;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
            margin-bottom: 25px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(31, 38, 135, 0.25);
        }
        
        h1, h2, h3 {
            color: #d6336c;
            font-weight: 700;
        }
        
        h1 {
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(45deg, #d6336c, #f86d9a);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            animation: titleAnimation 3s infinite alternate;
        }
        
        @keyframes titleAnimation {
            0% { transform: scale(1); }
            100% { transform: scale(1.03); }
        }
        
        .stButton>button {
            background: linear-gradient(45deg, #d6336c, #f86d9a);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 12px 30px;
            font-size: 1.1rem;
            font-weight: 600;
            transition: all 0.3s ease;
            display: block;
            margin: 0 auto;
            width: 50%;
            box-shadow: 0 4px 15px rgba(214, 51, 108, 0.4);
            animation: buttonPulse 2s infinite;
        }
        
        @keyframes buttonPulse {
            0% { box-shadow: 0 4px 15px rgba(214, 51, 108, 0.4); }
            50% { box-shadow: 0 4px 20px rgba(214, 51, 108, 0.6); }
            100% { box-shadow: 0 4px 15px rgba(214, 51, 108, 0.4); }
        }
        
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(214, 51, 108, 0.6);
            background: linear-gradient(45deg, #c52a61, #e75a84);
            animation: none;
        }
        
        .result-positive {
            background: linear-gradient(45deg, #ff6b6b, #ff8e8e);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 20px;
            animation: pulse 2s infinite;
            box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3);
        }
        
        .result-negative {
            background: linear-gradient(45deg, #51cf66, #94d82d);
            color: white;
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 20px;
            animation: pulse 2s infinite;
            box-shadow: 0 8px 32px rgba(81, 207, 102, 0.3);
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.02); }
            100% { transform: scale(1); }
        }
        
        .input-label {
            font-weight: 600;
            color: #495057;
            margin-bottom: 8px;
            display: block;
        }
        
        .stNumberInput>div>div>input, .stSelectbox>div>div>select {
            border-radius: 10px;
            border: 1px solid #ced4da;
            padding: 10px;
        }
        
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #4c68d7 0%, #3b5bdb 100%);
            color: white;
        }
        
        .sidebar-header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .doctor-advice {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #2196f3;
            margin: 20px 0;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #f44336;
            margin: 20px 0;
        }
        
        .success-box {
            background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
            padding: 20px;
            border-radius: 15px;
            border-left: 5px solid #4caf50;
            margin: 20px 0;
        }
        
        .heart-animation {
            display: block;
            margin: 0 auto;
            width: 150px;
            height: 150px;
            background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%23d6336c' d='M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z'/%3E%3C/svg%3E") center/contain no-repeat;
            animation: heartbeat 1.5s infinite;
        }
        
        @keyframes heartbeat {
            0% { transform: scale(1); }
            15% { transform: scale(1.1); }
            30% { transform: scale(1); }
            45% { transform: scale(1.1); }
            60% { transform: scale(1); }
        }
        
        .loading-animation {
            display: block;
            margin: 0 auto;
            width: 80px;
            height: 80px;
            position: relative;
        }
        
        .loading-animation:before, .loading-animation:after {
            content: "";
            background-color: #d6336c;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            position: absolute;
            top: 0;
            left: 0;
            animation: heart-load 1.2s infinite ease-in-out;
        }
        
        .loading-animation:after {
            top: 0;
            left: 40px;
            animation-delay: 0.4s;
        }
        
        @keyframes heart-load {
            0%, 100% { transform: scale(0.8); opacity: 0.7; }
            50% { transform: scale(1.2); opacity: 1; }
        }
        
        .metric-card {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin: 10px 0;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #d6336c;
            margin: 5px 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6c757d;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .card {
                padding: 15px;
            }
            
            h1 {
                font-size: 1.8rem;
            }
            
            .stButton>button {
                width: 100%;
                padding: 10px 20px;
                font-size: 0.9rem;
            }
            
            .result-positive, .result-negative {
                padding: 15px;
                font-size: 1.3rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Load trained model
# ---------------------------
@st.cache_resource
def load_model():
    try:
        # Load the actual model and scaler files
        with open("xgboost_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)
        with open("scaler.pkl", "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.info("Please make sure 'xgboost_model.pkl' and 'scaler.pkl' are in the same directory as this app.")
        return None, None

model, scaler = load_model()

# ---------------------------
# Model Information
# ---------------------------
MODEL_INFO = {
    "name": "XGBClassifier (XGBoost)",
    "version": "3.0.4",
    "training_date": "2025-08-21",
    "dataset": "UCI Heart Disease Dataset",
    "dataset_size": "303 records, 13 features",
    "description": "This XGBoost model predicts the likelihood of heart disease based on patient medical attributes.",
    "accuracy": "90.16%",
    "precision": "0.91",
    "recall": "0.90",
    "f1_score": "0.90"
}                     

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-header">
            <h2>Heart Disease Prediction</h2>
            <p>Assess your heart health with AI</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Heart animation
    st.markdown('<div class="heart-animation"></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### How to Use")
    st.info("""
        1. Fill in the patient details
        2. Click the Predict button
        3. Review the results and recommendations
    """)
    
    st.markdown("---")
    st.markdown("### Important Note")
    st.warning("""
    This tool is for preliminary assessment only.
    Always consult a healthcare professional for proper diagnosis.
    """)

# ---------------------------
# Main Content
# ---------------------------
st.markdown("<div class='main'>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.title("❤️ Heart Disease Prediction")
    st.markdown("Enter patient details below to assess the risk of heart disease.")

with col2:
    # Heart animation
    st.markdown('<div class="heart-animation"></div>', unsafe_allow_html=True)

# ---------------------------
# User Inputs
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🧑‍⚕️ Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="input-label">Age</p>', unsafe_allow_html=True)
    age = st.slider("Age", 20, 100, 50, label_visibility="collapsed")
    
    st.markdown('<p class="input-label">Sex</p>', unsafe_allow_html=True)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x==1 else "Female", label_visibility="collapsed")
    
    st.markdown('<p class="input-label">Chest Pain Type</p>', unsafe_allow_html=True)
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                     format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x], 
                     label_visibility="collapsed")

with col2:
    st.markdown('<p class="input-label">Resting Blood Pressure (mm Hg)</p>', unsafe_allow_html=True)
    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120, label_visibility="collapsed")
    
    st.markdown('<p class="input-label">Cholesterol (mg/dl)</p>', unsafe_allow_html=True)
    chol = st.slider("Cholesterol (mg/dl)", 100, 600, 200, label_visibility="collapsed")
    
    st.markdown('<p class="input-label">Fasting Blood Sugar > 120 mg/dl</p>', unsafe_allow_html=True)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], 
                      format_func=lambda x: "Yes" if x==1 else "No", 
                      label_visibility="collapsed")

with col3:
    st.markdown('<p class="input-label">Max Heart Rate Achieved</p>', unsafe_allow_html=True)
    thalach = st.slider("Max Heart Rate Achieved", 70, 250, 150, label_visibility="collapsed")
    
    st.markdown('<p class="input-label">Exercise Induced Angina</p>', unsafe_allow_html=True)
    exang = st.selectbox("Exercise Induced Angina", [0, 1], 
                        format_func=lambda x: "Yes" if x==1 else "No", 
                        label_visibility="collapsed")
    
    st.markdown('<p class="input-label">Oldpeak (ST depression)</p>', unsafe_allow_html=True)
    oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 10.0, 1.0, step=0.1, label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)

# Additional inputs in a separate card
st.markdown("<div class='card'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<p class="input-label">Resting ECG</p>', unsafe_allow_html=True)
    restecg = st.selectbox("Resting ECG", [0, 1, 2], 
                          format_func=lambda x: ["Normal", "ST-T Abnormality", "LV Hypertrophy"][x], 
                          label_visibility="collapsed")

with col2:
    st.markdown('<p class="input-label">Slope of ST Segment</p>', unsafe_allow_html=True)
    slope = st.selectbox("Slope of ST Segment", [0, 1, 2], 
                        format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x], 
                        label_visibility="collapsed")

with col3:
    st.markdown('<p class="input-label">Number of Major Vessels</p>', unsafe_allow_html=True)
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3], label_visibility="collapsed")
    
    st.markdown('<p class="input-label">Thalassemia</p>', unsafe_allow_html=True)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3], 
                       format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x], 
                       label_visibility="collapsed")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Prediction
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("📊 Prediction")

if st.button("Predict Heart Disease Risk"):
    if model is None or scaler is None:
        st.error("Model or scaler not loaded. Please check that the model files are available.")
    else:
        # Loading animation placeholder
        loading_placeholder = st.empty()
        with loading_placeholder.container():
            st.markdown("""
                <div style='text-align: center; padding: 20px;'>
                    <div class="loading-animation"></div>
                    <p style='margin-top: 15px; font-weight: 600; color: #d6336c;'>Analyzing patient data...</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Prepare input data
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        
        # Scale input
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        
        # Clear loading animation
        loading_placeholder.empty()
        
        # Show result based on prediction
        if prediction[0] == 1:
            st.markdown('<div class="result-positive">⚠️ Heart Disease Detected</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.markdown("### 🚨 Immediate Doctor Consultation Recommended")
            st.markdown("""
            **Our analysis indicates possible heart disease risk factors.**
            
            **Please consult a cardiologist for:**
            - Comprehensive medical evaluation
            - Further diagnostic tests (ECG, stress test, echocardiogram)
            - Professional medical advice and treatment plan
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="doctor-advice">', unsafe_allow_html=True)
            st.markdown("### 📋 Before Visiting Doctor")
            st.markdown("""
            - Note down all your symptoms and their frequency
            - Prepare your medical history and current medications list
            - Avoid strenuous activities until consultation
            - Monitor your blood pressure regularly
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.markdown('<div class="result-negative">✅ No Heart Disease Detected</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.markdown("### 👍 Good Heart Health")
            st.markdown("""
            **Our analysis shows no signs of heart disease.**
            
            **To maintain good heart health:**
            - Continue regular health check-ups
            - Maintain balanced diet and healthy weight
            - Exercise regularly
            - Avoid smoking and limit alcohol
            - Manage stress effectively
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("### 💡 Preventive Care Tips")
            st.markdown("""
            - Get annual health check-ups
            - Monitor blood pressure and cholesterol levels
            - Maintain healthy BMI (18.5-24.9)
            - Practice 30 minutes of moderate exercise daily
            - Eat heart-healthy foods (fruits, vegetables, whole grains)
            """)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)



# ---------------------------
# Model Information Section
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.subheader("🤖 Model Information")

# Model details
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    **Model Name:** {MODEL_INFO['name']}  
    **Version:** {MODEL_INFO['version']}  
    **Training Date:** {MODEL_INFO['training_date']}  
    **Dataset:** {MODEL_INFO['dataset']}  
    **Dataset Size:** {MODEL_INFO['dataset_size']}  
    **Description:** {MODEL_INFO['description']}
    """)

with col2:
    # Model performance metrics in a grid
    st.markdown("**Performance Metrics**")
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{MODEL_INFO['accuracy']}</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{MODEL_INFO['precision']}</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{MODEL_INFO['recall']}</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{MODEL_INFO['f1_score']}</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)        




# ---------------------------
# Footer with Disclaimer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>
    <p><strong>Disclaimer:</strong> This prediction is based on machine learning analysis and should not replace professional medical advice. 
    Always consult qualified healthcare providers for diagnosis and treatment.</p>
    <p>© 2025 Heart Disease Prediction System | For educational purposes only</p>
</div>
""", unsafe_allow_html=True)
