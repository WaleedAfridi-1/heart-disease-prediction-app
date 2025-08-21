# heart-disease-prediction-app
An interactive Streamlit web app powered by XGBoost to predict the likelihood of heart disease using the UCI dataset. Features include a modern UI with animations, risk meter, feature importance visualization, probability score, and personalized health recommendations.


# ❤️ Heart Disease Prediction App

An interactive **Streamlit web application** powered by **XGBoost** that predicts the likelihood of **heart disease** based on patient medical attributes.  
👉 **Live Demo:** [Heart Disease Prediction App](https://cardio-heart-disease-prediction-app.streamlit.app/)

---

## 🚀 Features
- 🎨 **Beautiful UI** with modern CSS styling & animations  
- ⚡ **XGBoost model** trained on the UCI Heart Disease dataset  
- 🔮 **Live predictions** with probability score  
- 📊 **Risk meter visualization** with progress bar  
- 🧠 **Feature importance indicators** to explain predictions  
- 💡 **Detailed health recommendations** based on results  
- 📱 **Mobile responsive design**  

---

## 📊 Dataset
- **Source:** UCI Heart Disease Dataset  
- **Size:** 303 records, 13 features  
- **Target:** Presence of heart disease (`1 = Yes`, `0 = No`)  

---

## 🤖 Model Information
- **Algorithm:** XGBClassifier (XGBoost)  
- **Version:** 3.0.4  
- **Training Date:** 2025-08-21  
- **Accuracy:** 90.16%  
- **Precision:** 0.91  
- **Recall:** 0.90  
- **F1-Score:** 0.90  

### 🔑 Key Features Used
- Age, Sex, Chest Pain Type  
- Resting Blood Pressure, Cholesterol  
- Fasting Blood Sugar, Resting ECG  
- Max Heart Rate Achieved  
- Exercise Induced Angina, Oldpeak (ST Depression)  
- Slope, Major Vessels, Thalassemia  

---

## 🛠️ Tech Stack
- **Python 3.10+**  
- **Streamlit** (Frontend + Backend)  
- **Scikit-learn** (Scaling & Preprocessing)  
- **XGBoost** (Machine Learning Model)  
- **NumPy** (Data Handling)  
- **Pickle** (Model Serialization)  

---

## 📂 Project Structure
heart-disease-prediction-app/
- │
- ├── model/
- │   └── heart_disease_model.pkl    # Trained XGBoost model
- │
- ├── app.py                         # Main Streamlit application
- ├── requirements.txt               # Dependencies
- ├── README.md                      # Project documentation
- │
- └── assets/                        # Images, CSS, etc.


---

## ⚡ Installation & Usage

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/WaleedAfridi-1/heart-disease-prediction-app.git
cd heart-disease-prediction-app
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the App
```bash
streamlit run app.py
```

### 4️⃣ Access in Browser
```bash
http://localhost:8501
```
---

📈 Future Enhancements
----------------------
- Add Ridge / Lasso / Logistic Regression / Random Forest models for comparison  
- Integration with Wearable Devices Data (IoT)  
- Deployment on Docker & Cloud (AWS/GCP/Azure)  
- Add User Authentication & Profile Tracking  
- Multi-language support 🌍  

🙌 Acknowledgments
------------------
- Dataset: UCI Heart Disease Dataset  
- Framework: Streamlit  
- Model: XGBoost  

👨‍💻 Author
------------
**Waleed Afridi**  
📍 Karachi, Pakistan | 🏫 BSCS @ Federal Urdu University  
💼 Passionate about Data Science, AI & Web Development  


