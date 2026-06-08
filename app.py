"""
Real-Time Electricity Theft Detection System - Complete Version with Explainable AI
SRIP 2026 - VIT Chennai
Using FIXED Random Forest Model (No Theft3 Default Bias)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import warnings
import os
import subprocess
import sys

warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION (MUST BE FIRST)
# ============================================
st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide"
)

# ============================================
# AUTO-TRAIN FIXED MODEL IF NOT FOUND
# ============================================
def check_and_train_model():
    """Check if fixed model exists, if not ask user to upload CSV and train"""
    model_path = 'models/rf_fixed_model.pkl'
    
    if not os.path.exists(model_path):
        st.warning("⚠️ Model not found. First time setup required.")
        st.info("📁 Please upload the ETD2022 dataset (loaded_data.csv) to train the model.")
        st.info("⏳ Training takes 2-3 minutes. The fixed model has tolerance for normal variations.")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key="train_upload")
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = "temp_dataset.csv"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.info("⏳ Training fixed model... This takes 2-3 minutes. Please wait.")
            
            progress_bar = st.progress(0)
            progress_bar.progress(30)
            
            # Run training with the uploaded file
            result = subprocess.run(
                [sys.executable, "src/fix_model.py", temp_path], 
                capture_output=True, 
                text=True,
                timeout=300
            )
            
            progress_bar.progress(100)
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            if result.returncode == 0:
                st.success("✅ Fixed model trained successfully!")
                st.rerun()
            else:
                st.error("❌ Training failed!")
                st.code(result.stderr if result.stderr else result.stdout)
                st.stop()
        else:
            st.stop()

# Call this to check and train if needed
check_and_train_model()

# ============================================
# LOAD FIXED MODEL AND AUXILIARY FILES
# ============================================
@st.cache_resource
def load_model():
    """Load the fixed model and auxiliary files"""
    model = joblib.load('models/rf_fixed_model.pkl')
    scaler = joblib.load('models/scaler_fixed.pkl')
    label_encoder = joblib.load('models/label_encoder_fixed.pkl')
    feature_names = joblib.load('models/feature_names_fixed.pkl')
    normal_means = joblib.load('models/normal_means.pkl')
    normal_stds = joblib.load('models/normal_stds.pkl')
    original_features = joblib.load('models/original_feature_names.pkl')
    return model, scaler, label_encoder, feature_names, normal_means, normal_stds, original_features

def compute_full_features(values, normal_means, normal_stds, original_features):
    """Convert raw 10 values to full feature set (10 original + 10 deviations + 2 ratios)"""
    # Original features
    features = values.copy()
    
    # Deviation features (how far from normal mean)
    for i, col in enumerate(original_features):
        deviation = abs(values[i] - normal_means[col]) / (normal_stds[col] + 0.01)
        features.append(deviation)
    
    # Ratio features
    features.append(values[0] / (values[6] + 0.01))  # Elec/Gas ratio
    features.append(values[6] / (values[0] + 0.01))  # Gas/Elec ratio
    
    return features

# Load everything
try:
    model, scaler, label_encoder, feature_names, normal_means, normal_stds, original_features = load_model()
    st.sidebar.success("✅ Fixed Model Loaded Successfully! (With Normal Tolerance)")
    st.sidebar.write(f"📊 Features: {len(feature_names)}")
except Exception as e:
    st.sidebar.error(f"❌ Model error: {e}")
    st.stop()

# Custom CSS
st.markdown("""
    <style>
    .stButton > button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 10px;
    }
    .stButton > button:hover {
        background-color: #ff0000;
    }
    .theft-warning {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid red;
    }
    .normal-status {
        background-color: #ccffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid green;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("⚡ Real-Time Electricity Theft Detection System")
st.markdown("**SRIP 2026 - VIT Chennai** | *Smart Grid Security using Machine Learning*")
st.markdown("---")

# Sidebar
st.sidebar.header("📊 System Information")
st.sidebar.markdown(f"**Model:** Random Forest (Fixed - Normal Tolerance)")
st.sidebar.markdown(f"**Accuracy:** 94%+")
st.sidebar.markdown(f"**Features:** {len(feature_names)}")
st.sidebar.markdown("---")

st.sidebar.header("📖 Theft Types")
st.sidebar.markdown("""
- **Normal:** No theft detected
- **Theft1:** Constant reduction (0.1-0.8x)
- **Theft2:** Zero consumption during random periods
- **Theft3:** Hourly random reduction
- **Theft4:** Random fraction of mean consumption
- **Theft5:** Reports mean consumption constantly
""")

# Class descriptions
class_descriptions = {
    'Normal': "✅ Normal consumption pattern - No theft detected",
    'Theft1': "⚠️ Constant reduction (0.1-0.8x of actual consumption)",
    'Theft2': "⚠️ Zero consumption during random periods",
    'Theft3': "⚠️ Hourly random reduction (0.1-0.8x)",
    'Theft4': "⚠️ Random fraction of mean consumption",
    'Theft5': "⚠️ Reports mean consumption constantly"
}

# Show normal ranges in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Normal Value Ranges")
for col in original_features[:3]:
    mean_val = normal_means[col]
    std_val = normal_stds[col]
    short_name = col.split('[')[0][:20]
    st.sidebar.markdown(f"**{short_name}:** {mean_val:.1f} ± {std_val:.1f}")

# REAL test values from dataset
REAL_NORMAL = [22.04, 3.59, 0, 0, 4.59, 8.19, 136.59, 124.00, 3.34, 9.25]
REAL_THEFT1 = [11.02, 1.80, 0, 0, 2.30, 4.10, 68.30, 62.00, 1.67, 4.63]
REAL_THEFT2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
REAL_THEFT3 = [13.82, 2.25, 0, 0, 2.87, 5.13, 74.01, 66.13, 2.09, 5.79]
REAL_THEFT4 = [18.77, 1.64, 0, 0, 4.14, 10.46, 89.15, 68.65, 13.98, 6.52]
REAL_THEFT5 = [34.46, 2.99, 0, 0, 7.52, 19.00, 155.54, 118.29, 25.40, 11.85]
MODERATE_NORMAL = [28.50, 4.20, 2.50, 1.80, 6.30, 12.50, 110.00, 95.00, 5.50, 11.00]

# Initialize session state
if 'elec' not in st.session_state:
    st.session_state.elec = float(REAL_NORMAL[0])
if 'fans' not in st.session_state:
    st.session_state.fans = float(REAL_NORMAL[1])
if 'cooling' not in st.session_state:
    st.session_state.cooling = float(REAL_NORMAL[2])
if 'heating_elec' not in st.session_state:
    st.session_state.heating_elec = float(REAL_NORMAL[3])
if 'lights' not in st.session_state:
    st.session_state.lights = float(REAL_NORMAL[4])
if 'equip' not in st.session_state:
    st.session_state.equip = float(REAL_NORMAL[5])
if 'gas' not in st.session_state:
    st.session_state.gas = float(REAL_NORMAL[6])
if 'heating_gas' not in st.session_state:
    st.session_state.heating_gas = float(REAL_NORMAL[7])
if 'equip_gas' not in st.session_state:
    st.session_state.equip_gas = float(REAL_NORMAL[8])
if 'water' not in st.session_state:
    st.session_state.water = float(REAL_NORMAL[9])

# Store last prediction for session
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_confidence' not in st.session_state:
    st.session_state.last_confidence = None
if 'last_values' not in st.session_state:
    st.session_state.last_values = None

# Tabs - NOW WITH 6 TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 Single Prediction", "📁 Batch Upload", "📊 Live Monitoring", "📈 Analytics", "🔬 Explainable AI", "💡 Recommendations"])

# ============================================
# TAB 1: Single Prediction
# ============================================
with tab1:
    st.header("🔍 Real-Time Electricity Theft Detection")
    st.markdown("Enter the energy consumption readings to detect theft")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚡ Electricity Features")
        elec = st.number_input("Electricity: Facility [kW]", min_value=0.0, max_value=500.0, 
                               value=st.session_state.elec, key="elec_input", step=1.0)
        fans = st.number_input("Fans: Electricity [kW]", min_value=0.0, max_value=200.0, 
                               value=st.session_state.fans, key="fans_input", step=1.0)
        cooling = st.number_input("Cooling: Electricity [kW]", min_value=0.0, max_value=200.0, 
                                  value=st.session_state.cooling, key="cooling_input", step=1.0)
        heating_elec = st.number_input("Heating: Electricity [kW]", min_value=0.0, max_value=200.0, 
                                       value=st.session_state.heating_elec, key="heating_elec_input", step=1.0)
        interior_lights = st.number_input("Interior Lights: Electricity [kW]", min_value=0.0, max_value=100.0, 
                                          value=st.session_state.lights, key="lights_input", step=1.0)
        interior_equip = st.number_input("Interior Equipment: Electricity [kW]", min_value=0.0, max_value=200.0, 
                                         value=st.session_state.equip, key="equip_input", step=1.0)
    
    with col2:
        st.subheader("🔥 Gas Features")
        gas_facility = st.number_input("Gas: Facility [kW]", min_value=0.0, max_value=500.0, 
                                       value=st.session_state.gas, key="gas_input", step=1.0)
        heating_gas = st.number_input("Heating: Gas [kW]", min_value=0.0, max_value=400.0, 
                                      value=st.session_state.heating_gas, key="heating_gas_input", step=1.0)
        interior_equip_gas = st.number_input("Interior Equipment: Gas [kW]", min_value=0.0, max_value=200.0, 
                                             value=st.session_state.equip_gas, key="equip_gas_input", step=1.0)
        water_heater = st.number_input("Water Heater: Gas [kW]", min_value=0.0, max_value=200.0, 
                                       value=st.session_state.water, key="water_input", step=1.0)
    
    # Buttons
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5, col_btn6, col_btn7 = st.columns(7)
    
    with col_btn1:
        if st.button("🔍 Detect Theft", use_container_width=True):
            input_values = [elec, fans, cooling, heating_elec, interior_lights,
                           interior_equip, gas_facility, heating_gas, interior_equip_gas, water_heater]
            
            # Compute full features
            full_features = compute_full_features(input_values, normal_means, normal_stds, original_features)
            input_scaled = scaler.transform([full_features])
            
            with st.spinner("Analyzing consumption pattern..."):
                time.sleep(0.5)
                prediction = model.predict(input_scaled)
                probabilities = model.predict_proba(input_scaled)
                
                predicted_class = label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(probabilities) * 100
                
                # Store for other tabs
                st.session_state.last_prediction = predicted_class
                st.session_state.last_confidence = confidence
                st.session_state.last_values = input_values
                
                st.markdown("---")
                st.subheader("🔍 Detection Result")
                
                if predicted_class == "Normal":
                    st.markdown(f"""
                    <div class="normal-status">
                        <h3>✅ STATUS: NORMAL</h3>
                        <p>No electricity theft detected.</p>
                        <p>Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="theft-warning">
                        <h3>⚠️ ALERT: THEFT DETECTED!</h3>
                        <p><strong>Type:</strong> {predicted_class}</p>
                        <p><strong>Description:</strong> {class_descriptions.get(predicted_class, 'Suspicious consumption pattern')}</p>
                        <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Show probabilities
                st.subheader("📊 Detection Probabilities")
                prob_df = pd.DataFrame({
                    'Class': label_encoder.classes_,
                    'Probability (%)': probabilities[0] * 100
                }).sort_values('Probability (%)', ascending=False)
                
                fig, ax = plt.subplots(figsize=(8, 4))
                colors = ['red' if c != 'Normal' else 'green' for c in prob_df['Class']]
                ax.barh(prob_df['Class'], prob_df['Probability (%)'], color=colors)
                ax.set_xlabel('Probability (%)')
                ax.set_title('Class Probabilities')
                st.pyplot(fig)
    
    with col_btn2:
        if st.button("📋 Normal", use_container_width=True):
            st.session_state.elec = float(REAL_NORMAL[0])
            st.session_state.fans = float(REAL_NORMAL[1])
            st.session_state.cooling = float(REAL_NORMAL[2])
            st.session_state.heating_elec = float(REAL_NORMAL[3])
            st.session_state.lights = float(REAL_NORMAL[4])
            st.session_state.equip = float(REAL_NORMAL[5])
            st.session_state.gas = float(REAL_NORMAL[6])
            st.session_state.heating_gas = float(REAL_NORMAL[7])
            st.session_state.equip_gas = float(REAL_NORMAL[8])
            st.session_state.water = float(REAL_NORMAL[9])
            st.rerun()
    
    with col_btn3:
        if st.button("⚠️ Theft1", use_container_width=True):
            st.session_state.elec = float(REAL_THEFT1[0])
            st.session_state.fans = float(REAL_THEFT1[1])
            st.session_state.cooling = float(REAL_THEFT1[2])
            st.session_state.heating_elec = float(REAL_THEFT1[3])
            st.session_state.lights = float(REAL_THEFT1[4])
            st.session_state.equip = float(REAL_THEFT1[5])
            st.session_state.gas = float(REAL_THEFT1[6])
            st.session_state.heating_gas = float(REAL_THEFT1[7])
            st.session_state.equip_gas = float(REAL_THEFT1[8])
            st.session_state.water = float(REAL_THEFT1[9])
            st.rerun()
    
    with col_btn4:
        if st.button("🔄 Theft2", use_container_width=True):
            st.session_state.elec = float(REAL_THEFT2[0])
            st.session_state.fans = float(REAL_THEFT2[1])
            st.session_state.cooling = float(REAL_THEFT2[2])
            st.session_state.heating_elec = float(REAL_THEFT2[3])
            st.session_state.lights = float(REAL_THEFT2[4])
            st.session_state.equip = float(REAL_THEFT2[5])
            st.session_state.gas = float(REAL_THEFT2[6])
            st.session_state.heating_gas = float(REAL_THEFT2[7])
            st.session_state.equip_gas = float(REAL_THEFT2[8])
            st.session_state.water = float(REAL_THEFT2[9])
            st.rerun()
    
    with col_btn5:
        if st.button("📊 Theft4", use_container_width=True):
            st.session_state.elec = float(REAL_THEFT4[0])
            st.session_state.fans = float(REAL_THEFT4[1])
            st.session_state.cooling = float(REAL_THEFT4[2])
            st.session_state.heating_elec = float(REAL_THEFT4[3])
            st.session_state.lights = float(REAL_THEFT4[4])
            st.session_state.equip = float(REAL_THEFT4[5])
            st.session_state.gas = float(REAL_THEFT4[6])
            st.session_state.heating_gas = float(REAL_THEFT4[7])
            st.session_state.equip_gas = float(REAL_THEFT4[8])
            st.session_state.water = float(REAL_THEFT4[9])
            st.rerun()
    
    with col_btn6:
        if st.button("📈 Theft5", use_container_width=True):
            st.session_state.elec = float(REAL_THEFT5[0])
            st.session_state.fans = float(REAL_THEFT5[1])
            st.session_state.cooling = float(REAL_THEFT5[2])
            st.session_state.heating_elec = float(REAL_THEFT5[3])
            st.session_state.lights = float(REAL_THEFT5[4])
            st.session_state.equip = float(REAL_THEFT5[5])
            st.session_state.gas = float(REAL_THEFT5[6])
            st.session_state.heating_gas = float(REAL_THEFT5[7])
            st.session_state.equip_gas = float(REAL_THEFT5[8])
            st.session_state.water = float(REAL_THEFT5[9])
            st.rerun()
    
    with col_btn7:
        if st.button("🟡 Moderate Normal", use_container_width=True):
            st.session_state.elec = float(MODERATE_NORMAL[0])
            st.session_state.fans = float(MODERATE_NORMAL[1])
            st.session_state.cooling = float(MODERATE_NORMAL[2])
            st.session_state.heating_elec = float(MODERATE_NORMAL[3])
            st.session_state.lights = float(MODERATE_NORMAL[4])
            st.session_state.equip = float(MODERATE_NORMAL[5])
            st.session_state.gas = float(MODERATE_NORMAL[6])
            st.session_state.heating_gas = float(MODERATE_NORMAL[7])
            st.session_state.equip_gas = float(MODERATE_NORMAL[8])
            st.session_state.water = float(MODERATE_NORMAL[9])
            st.rerun()

# ============================================
# TAB 2: Batch Upload
# ============================================
with tab2:
    st.header("📁 Batch Detection - Upload CSV File")
    st.markdown("Upload a CSV file with energy consumption data for batch prediction")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(f"📊 File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head())
        
        if st.button("🔍 Run Batch Detection", use_container_width=True):
            with st.spinner("Processing batch..."):
                results = []
                for idx, row in df.iterrows():
                    values = row.iloc[:10].values.tolist()
                    full_features = compute_full_features(values, normal_means, normal_stds, original_features)
                    scaled = scaler.transform([full_features])
                    pred = model.predict(scaled)[0]
                    pred_class = label_encoder.inverse_transform([pred])[0]
                    probs = model.predict_proba(scaled)[0]
                    confidence = max(probs) * 100
                    results.append({'Prediction': pred_class, 'Confidence': f'{confidence:.1f}%'})
                
                result_df = pd.DataFrame(results)
                st.dataframe(result_df)
                
                csv = result_df.to_csv(index=False)
                st.download_button("📥 Download Results", csv, "detection_results.csv", "text/csv")

# ============================================
# TAB 3: Live Monitoring
# ============================================
with tab3:
    st.header("📊 Live Consumption Monitoring")
    st.markdown("Simulate real-time energy consumption monitoring")
    
    if 'monitoring_data' not in st.session_state:
        st.session_state.monitoring_data = []
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        monitoring_duration = st.slider("Monitoring Duration (seconds)", 10, 60, 30)
        start_monitoring = st.button("🎬 Start Live Monitoring", use_container_width=True)
        
        if start_monitoring:
            st.session_state.monitoring_data = []
            st.info(f"Monitoring for {monitoring_duration} seconds...")
    
    with col2:
        placeholder = st.empty()
        
        if start_monitoring:
            progress_bar = st.progress(0)
            for i in range(monitoring_duration):
                consumption = np.random.uniform(10, 200, 10).tolist()
                full_features = compute_full_features(consumption, normal_means, normal_stds, original_features)
                scaled = scaler.transform([full_features])
                pred = model.predict(scaled)[0]
                pred_class = label_encoder.inverse_transform([pred])[0]
                
                st.session_state.monitoring_data.append({
                    'Time': datetime.now().strftime("%H:%M:%S"),
                    'Consumption': np.mean(consumption),
                    'Status': pred_class
                })
                
                df_live = pd.DataFrame(st.session_state.monitoring_data)
                
                with placeholder.container():
                    if pred_class == "Normal":
                        st.success(f"🟢 Status: {pred_class}")
                    else:
                        st.error(f"🔴 ALERT: {pred_class} Detected!")
                    
                    if len(df_live) > 0:
                        st.line_chart(df_live.set_index('Time')['Consumption'])
                        st.dataframe(df_live.tail(10))
                
                progress_bar.progress((i + 1) / monitoring_duration)
                time.sleep(1)
            
            st.success("✅ Monitoring Complete!")

# ============================================
# TAB 4: Analytics
# ============================================
with tab4:
    st.header("📈 System Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", "94%+", "Fixed Model")
    with col2:
        st.metric("Model Type", "Random Forest", "Normal Tolerance")
    with col3:
        st.metric("Classes", "6", "Theft1-5 + Normal")
    with col4:
        st.metric("Features", str(len(feature_names)), "22 Total")
    
    st.subheader("🔑 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': original_features,
        'Importance': model.feature_importances_[:len(original_features)]
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance for Theft Detection')
    st.pyplot(fig)
    
    st.subheader("📖 Theft Type Descriptions")
    for theft_type, desc in class_descriptions.items():
        if theft_type in label_encoder.classes_:
            st.markdown(f"**{theft_type}:** {desc}")

# ============================================
# TAB 5: Explainable AI
# ============================================
with tab5:
    st.header("🔬 Explainable AI - Why Did the Model Predict Theft?")
    st.markdown("""
    **Understanding the decision:** Feature importance shows what consumption patterns most influence detection.
    The fixed model has NO default bias toward Theft3.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Select a Test Case")
        
        explain_option = st.radio(
            "Choose test case:",
            ["Normal", "Theft1", "Theft2", "Theft3", "Theft4", "Theft5", "Moderate Normal", "Custom Values"],
            key="explain_option"
        )
        
        if explain_option == "Normal":
            explain_values = REAL_NORMAL
            st.info("📋 Normal consumption pattern")
            
        elif explain_option == "Theft1":
            explain_values = REAL_THEFT1
            st.warning("⚠️ Theft1 pattern (50% reduction)")
            
        elif explain_option == "Theft2":
            explain_values = REAL_THEFT2
            st.error("🔄 Theft2 pattern (All zeros)")
            
        elif explain_option == "Theft3":
            explain_values = REAL_THEFT3
            st.info("📊 Theft3 pattern (Low random values)")
            
        elif explain_option == "Theft4":
            explain_values = REAL_THEFT4
            st.info("📈 Theft4 pattern (Random fraction of mean)")
            
        elif explain_option == "Theft5":
            explain_values = REAL_THEFT5
            st.info("📉 Theft5 pattern (Constant mean)")
            
        elif explain_option == "Moderate Normal":
            explain_values = MODERATE_NORMAL
            st.info("🟡 Moderate Normal pattern (within range)")
            
        else:
            st.write("Enter custom values:")
            explain_values = []
            for i, name in enumerate(original_features):
                val = st.number_input(f"{name}", value=0.0, key=f"explain_{i}", step=1.0)
                explain_values.append(val)
        
        explain_button = st.button("🔍 Explain Prediction", use_container_width=True, key="explain_btn")
    
    with col2:
        st.subheader("📖 How Interpret Results")
        st.markdown("""
        **Feature Importance Shows:**
        - **Higher importance** = Stronger influence on theft detection
        
        **Confidence Interpretation:**
        - **>80%** = Very certain
        - **50-80%** = Moderately certain
        - **<50%** = Uncertain / borderline case
        
        **This model has NO default bias toward Theft3.**
        """)
    
    if explain_button:
        with st.spinner("Analyzing and generating explanation..."):
            full_features = compute_full_features(explain_values, normal_means, normal_stds, original_features)
            input_scaled = scaler.transform([full_features])
            
            prediction = model.predict(input_scaled)[0]
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            probabilities = model.predict_proba(input_scaled)
            confidence = np.max(probabilities) * 100
            
            feature_importance_rf = pd.DataFrame({
                'Feature': original_features,
                'Importance': model.feature_importances_[:len(original_features)]
            }).sort_values('Importance', ascending=False)
            
            st.markdown("---")
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("📊 Prediction Result")
                if predicted_class == "Normal":
                    st.success(f"✅ **{predicted_class}** (Confidence: {confidence:.1f}%)")
                else:
                    st.error(f"⚠️ **{predicted_class}** (Confidence: {confidence:.1f}%)")
            
            with col_res2:
                st.subheader("📈 Model Info")
                st.metric("Model", "Random Forest (Fixed)")
                st.caption("No default bias toward Theft3")
            
            st.markdown("---")
            
            st.subheader("🔍 Which Features Most Influence Theft Detection?")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            imp_df = feature_importance_rf.head(8)
            bars = ax.barh(imp_df['Feature'], imp_df['Importance'] * 100, color='coral')
            ax.set_xlabel('Importance (%)', fontsize=12)
            ax.set_title('Global Feature Importance for Theft Detection', fontsize=14)
            
            for bar, val in zip(bars, imp_df['Importance'] * 100):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}%', va='center', fontsize=10)
            
            st.pyplot(fig)
            
            st.subheader("📊 Class Probability Distribution")
            prob_df = pd.DataFrame({
                'Class': label_encoder.classes_,
                'Probability (%)': probabilities[0] * 100
            }).sort_values('Probability (%)', ascending=False)
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            colors2 = ['red' if c != 'Normal' else 'green' for c in prob_df['Class']]
            ax2.barh(prob_df['Class'], prob_df['Probability (%)'], color=colors2)
            ax2.set_xlabel('Probability (%)')
            ax2.set_title('Model Confidence Across All Classes')
            st.pyplot(fig2)
            
            st.subheader("💡 Interpretation")
            
            if predicted_class != "Normal":
                st.markdown(f"**🔴 Why was this flagged as theft?**")
                st.markdown(f"- Detected **{predicted_class}** pattern with {confidence:.1f}% confidence")
                st.markdown(f"- {class_descriptions.get(predicted_class, 'Suspicious pattern')}")
                
                if confidence < 50:
                    st.info("📌 Low confidence indicates borderline case - manual review recommended")
            else:
                st.markdown("**🟢 Why was this considered normal?**")
                st.markdown(f"- Matches normal behavior with {confidence:.1f}% confidence")

# ============================================
# TAB 6: Recommendations System
# ============================================
with tab6:
    st.header("💡 Smart Recommendations for Utility Companies")
    st.markdown("""
    Based on the detection results, here are actionable recommendations to reduce losses and improve grid security.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 Select Detection Result")
        
        rec_option = st.radio(
            "Choose input method:",
            ["Use last detection result", "Select theft type manually"]
        )
        
        if rec_option == "Use last detection result":
            if st.session_state.last_prediction is not None:
                detected_type = st.session_state.last_prediction
                confidence = st.session_state.last_confidence
                st.success(f"📌 Last detection: **{detected_type}** (Confidence: {confidence:.1f}%)")
            else:
                st.warning("No previous detection. Make a prediction in Tab 1 first.")
                detected_type = None
                confidence = 0
        else:
            detected_type = st.selectbox(
                "Select theft type:",
                ["Normal", "Theft1", "Theft2", "Theft3", "Theft4", "Theft5"]
            )
            confidence = st.slider("Confidence (%)", 0, 100, 70)
    
    with col2:
        st.subheader("📖 About This Feature")
        st.markdown("""
        This recommendations engine provides:
        - **Immediate actions** for detected theft
        - **Priority levels** based on confidence
        - **Cost-saving estimates**
        - **Long-term prevention strategies**
        
        Recommendations are based on industry best practices.
        """)
    
    if detected_type is not None:
        st.markdown("---")
        
        loss_estimates = {
            "Theft1": 12000,
            "Theft2": 30000,
            "Theft3": 6000,
            "Theft4": 4000,
            "Theft5": 3500,
            "Normal": 0
        }
        
        if detected_type == "Normal":
            st.subheader("🟢 No Action Required")
            st.markdown("""
            **Status:** Normal consumption pattern detected
            
            **Recommendations:**
            - ✅ Continue regular monitoring
            - 📊 Review usage periodically (monthly)
            - 📝 No immediate action needed
            """)
            
        elif detected_type == "Theft1":
            st.subheader("🔴 HIGH PRIORITY - Constant Reduction Detected")
            st.markdown("""
            **Theft Type:** Constant reduction (meter tampering)
            
            **Immediate Actions:**
            1. 📞 **Schedule immediate inspection** (within 24 hours)
            2. 🔒 **Remotely disconnect meter** if available
            3. 📸 **Send field team** to verify meter seal integrity
            4. 📋 **Flag customer account** for investigation
            
            **Estimated Loss:** $500-2000 per month
            """)
            
        elif detected_type == "Theft2":
            st.subheader("🔴 HIGH PRIORITY - Zero Consumption Detected")
            st.markdown("""
            **Theft Type:** Meter bypass / Direct tapping
            
            **Immediate Actions:**
            1. 🚨 **URGENT: Dispatch field team immediately**
            2. 🔍 **Check for meter bypass** (most common method)
            3. ⚡ **Inspect service line connections**
            4. 📸 **Document evidence** for legal action
            
            **Estimated Loss:** $1000-5000 per month
            """)
            
        elif detected_type == "Theft3":
            st.subheader("🟠 MEDIUM PRIORITY - Hourly Random Reduction")
            st.markdown("""
            **Theft Type:** Smart meter hacking / software manipulation
            
            **Immediate Actions:**
            1. 🔬 **Analyze meter communication logs**
            2. 💻 **Check for firmware tampering**
            3. 🔄 **Compare with neighboring consumption patterns**
            4. 📞 **Contact customer for explanation** (48 hours)
            
            **Estimated Loss:** $300-1000 per month
            """)
            
        elif detected_type == "Theft4":
            st.subheader("🟡 MEDIUM-LOW PRIORITY - Random Fraction of Mean")
            st.markdown("""
            **Theft Type:** Load limiting / Partial tampering
            
            **Immediate Actions:**
            1. 📊 **Review 30-day consumption history**
            2. 🔍 **Compare with similar consumer profiles**
            3. 📞 **Send notification letter** (7 days)
            4. 🔄 **Monitor for pattern continuation**
            
            **Estimated Loss:** $200-800 per month
            """)
            
        elif detected_type == "Theft5":
            st.subheader("🟡 MEDIUM PRIORITY - Constant Mean Reporting")
            st.markdown("""
            **Theft Type:** Meter freezing / Value manipulation
            
            **Immediate Actions:**
            1. 🔄 **Verify meter firmware version**
            2. 📊 **Check for flat-lining consumption pattern**
            3. 🔋 **Request battery check** (for electronic meters)
            4. 📞 **Schedule meter inspection** (2 weeks)
            
            **Estimated Loss:** $150-600 per month
            """)
        
        st.markdown("---")
        st.subheader("📊 Confidence-Based Actions")
        
        if confidence > 80:
            st.success(f"✅ **HIGH CONFIDENCE ({confidence:.0f}%)** - Take immediate action")
        elif confidence > 50:
            st.warning(f"⚠️ **MODERATE CONFIDENCE ({confidence:.0f}%)** - Investigate within 1 week")
        else:
            st.info(f"ℹ️ **LOW CONFIDENCE ({confidence:.0f}%)** - Manual review recommended")
        
        st.markdown("---")
        st.subheader("💰 Estimated Annual Loss")
        
        annual_loss = loss_estimates.get(detected_type, 0)
        if annual_loss > 0:
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Estimated Annual Loss", f"${annual_loss:,}")
            with col_b:
                st.metric("Potential Savings if Fixed", f"${annual_loss:,}", delta="100%")
        
        with st.expander("🔧 Long-term Prevention Strategies"):
            st.markdown("""
            **Technology Upgrades:**
            - Deploy AI-based real-time detection (this system!)
            - Install advanced smart meters with anti-tampering
            - Implement blockchain-based meter data logging
            
            **Process Improvements:**
            - Regular meter reading audits
            - Customer education programs
            - Whistleblower rewards for theft reporting
            """)
        
        st.markdown("---")
        
        report = f"""
        ELECTRICITY THEFT DETECTION REPORT
        =================================
        Detection Type: {detected_type}
        Confidence: {confidence:.1f}%
        Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        RECOMMENDATIONS:
        {class_descriptions.get(detected_type, 'Review consumption pattern')}
        
        Priority: {'HIGH' if confidence > 70 else 'MEDIUM' if confidence > 50 else 'LOW'}
        
        Action Required: {'Immediate' if confidence > 70 else 'Schedule inspection' if confidence > 50 else 'Monitor only'}
        
        Estimated Annual Loss: ${annual_loss:,}
        
        This report was generated by the Electricity Theft Detection System.
        SRIP 2026 - VIT Chennai
        """
        
        st.download_button(
            "📥 Download Investigation Report",
            report,
            f"theft_report_{detected_type}_{datetime.now().strftime('%Y%m%d')}.txt",
            "text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center">
        <p>⚡ SRIP 2026 - VIT Chennai | Fixed Random Forest Model (No Theft3 Default Bias)</p>
        <p>Based on ETD2022 Benchmark Dataset | Fair & Explainable AI</p>
    </div>
""", unsafe_allow_html=True)
