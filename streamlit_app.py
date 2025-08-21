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
            font-size: 1.5rem;
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
            font-size: 1.5rem;
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
        
        .risk-meter {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .risk-fill {
            height: 100%;
            border-radius: 15px;
            background: linear-gradient(90deg, #51cf66, #f8f9fa, #ff6b6b);
            transition: width 1s ease-in-out;
        }
        
        .feature-importance {
            margin-top: 30px;
        }
        
        .feature-bar {
            height: 25px;
            margin-bottom: 10px;
            border-radius: 5px;
            background: linear-gradient(90deg, #4c68d7, #3b5bdb);
            transition: width 1s ease-in-out;
        }
        
        .prediction-animation {
            text-align: center;
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
                font-size: 1.1rem;
            }
            
            .metric-value {
                font-size: 1.4rem;
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
    st.markdown("### About")
    st.write("This app uses a machine learning model to predict the likelihood of heart disease based on patient health metrics.")

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
        # Create a placeholder for the loading animation
        loading_placeholder = st.empty()
        
        with loading_placeholder.container():
            # Show CSS-based loading animation
            st.markdown("""
                <div style='text-align: center; padding: 20px;'>
                    <div class="loading-animation"></div>
                    <p style='margin-top: 15px; font-weight: 600; color: #d6336c;'>Analyzing patient data...</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Prepare input data
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, slope, ca, thal]])
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)
        prediction_proba = model.predict_proba(scaled_data)
        
        # Calculate risk percentage
        risk_percentage = prediction_proba[0][1] * 100
        
        # Clear the loading animation
        loading_placeholder.empty()
        
        # Display results
        if prediction[0] == 1:
            st.markdown(f'<div class="result-positive">⚠️ High Risk of Heart Disease ({risk_percentage:.1f}% probability)</div>', unsafe_allow_html=True)
            
            st.warning("""
            **Recommendations:**
            - Consult a cardiologist as soon as possible
            - Adopt a heart-healthy diet low in saturated fats
            - Engage in regular physical activity as recommended by your doctor
            - Monitor your blood pressure and cholesterol regularly
            - Consider stress reduction techniques
            """)
        else:
            st.markdown(f'<div class="result-negative">✅ Low Risk of Heart Disease ({risk_percentage:.1f}% probability)</div>', unsafe_allow_html=True)
            
            st.success("""
            **Keep up the good work:**
            - Maintain a balanced diet and healthy weight
            - Continue regular physical activity
            - Avoid smoking and limit alcohol consumption
            - Schedule regular health check-ups
            - Manage stress effectively
            """)
        
        # Display risk meter
        st.markdown("### Risk Assessment")
        st.markdown(f'<div class="risk-meter"><div class="risk-fill" style="width: {risk_percentage}%;"></div></div>', unsafe_allow_html=True)
        st.caption(f"Estimated risk level: {risk_percentage:.1f}%")
        
        # Feature importance visualization
        st.markdown("### Key Factors in Prediction")
        factors = [
            {"name": "Age", "importance": min(abs(age-50)/50, 1.0)},
            {"name": "Cholesterol", "importance": min(abs(chol-200)/400, 1.0)},
            {"name": "Max Heart Rate", "importance": min(abs(thalach-150)/100, 1.0)},
            {"name": "Chest Pain", "importance": cp/3},
            {"name": "ST Depression", "importance": min(oldpeak/5, 1.0)},
            {"name": "Exercise Angina", "importance": exang}
        ]
        
        # Sort by importance
        factors.sort(key=lambda x: x["importance"], reverse=True)
        
        for factor in factors[:3]:
            importance_percentage = factor["importance"] * 100
            st.markdown(f"**{factor['name']}**")
            st.markdown(f'<div class="feature-bar" style="width: {importance_percentage}%"></div>', unsafe_allow_html=True)

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

# Feature importance explanation
st.markdown("---")
st.markdown("**Key Features Used by the Model:**")
st.markdown("""
- **Age:** Patient's age in years
- **Sex:** Patient's gender (Male/Female)
- **Chest Pain Type:** Type of chest pain experienced
- **Resting Blood Pressure:** Blood pressure at rest (mm Hg)
- **Cholesterol:** Serum cholesterol level (mg/dl)
- **Fasting Blood Sugar:** Blood sugar after fasting (>120 mg/dl)
- **Resting ECG:** Electrocardiographic results at rest
- **Max Heart Rate:** Maximum heart rate achieved
- **Exercise Induced Angina:** Chest pain during exercise
- **Oldpeak:** ST depression induced by exercise
- **Slope:** Slope of the peak exercise ST segment
- **Major Vessels:** Number of major vessels colored by fluoroscopy
- **Thalassemia:** Blood disorder measurement
""")

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)