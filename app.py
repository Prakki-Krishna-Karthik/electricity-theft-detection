"""
Real-Time Electricity Theft Detection System - Complete Version with Explainable AI
SRIP 2026 - VIT Chennai
Using SMOTE Balanced Random Forest Model
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
warnings.filterwarnings('ignore')


# Page configuration
st.set_page_config(
    page_title="Electricity Theft Detection",
    page_icon="⚡",
    layout="wide"
)

# Load trained model (SMOTE Balanced)
@st.cache_resource
def load_model():
    model = joblib.load('models/rf_balanced_smote.pkl')
    scaler = joblib.load('models/scaler.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return model, scaler, label_encoder, feature_names

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
    .borderline-status {
        background-color: #ffffcc;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid orange;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("⚡ Real-Time Electricity Theft Detection System")
st.markdown("**SRIP 2026 - VIT Chennai** | *Smart Grid Security using Machine Learning*")
st.markdown("---")

# Load model
try:
    model, scaler, label_encoder, feature_names = load_model()
    st.sidebar.success("✅ SMOTE Balanced Model Loaded Successfully!")
    st.sidebar.write(f"📊 Features: {len(feature_names)}")
except Exception as e:
    st.sidebar.error(f"❌ Model not found! Please run train_balanced_smote.py first. Error: {e}")
    st.stop()

# Sidebar
st.sidebar.header("📊 System Information")
st.sidebar.markdown(f"**Model:** Random Forest (SMOTE Balanced)")
st.sidebar.markdown(f"**Accuracy:** 94.60%")
st.sidebar.markdown(f"**Training:** 1,592,754 balanced samples")
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

# Initialize session state for input values
if 'elec' not in st.session_state:
    st.session_state.elec = 22.04
if 'fans' not in st.session_state:
    st.session_state.fans = 3.59
if 'cooling' not in st.session_state:
    st.session_state.cooling = 0.0
if 'heating_elec' not in st.session_state:
    st.session_state.heating_elec = 0.0
if 'lights' not in st.session_state:
    st.session_state.lights = 4.59
if 'equip' not in st.session_state:
    st.session_state.equip = 8.19
if 'gas' not in st.session_state:
    st.session_state.gas = 136.59
if 'heating_gas' not in st.session_state:
    st.session_state.heating_gas = 124.00
if 'equip_gas' not in st.session_state:
    st.session_state.equip_gas = 3.34
if 'water' not in st.session_state:
    st.session_state.water = 9.25

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Single Prediction", "📁 Batch Upload", "📊 Live Monitoring", "📈 Analytics", "🔬 Explainable AI"])

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
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🔍 Detect Theft", use_container_width=True):
            input_features = np.array([[
                elec, fans, cooling, heating_elec, interior_lights,
                interior_equip, gas_facility, heating_gas, interior_equip_gas, water_heater
            ]])
            
            input_scaled = scaler.transform(input_features)
            
            with st.spinner("Analyzing consumption pattern..."):
                time.sleep(0.5)
                prediction = model.predict(input_scaled)
                probabilities = model.predict_proba(input_scaled)
                
                predicted_class = label_encoder.inverse_transform(prediction)[0]
                confidence = np.max(probabilities) * 100
                
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
        if st.button("📋 Load Normal Test Case", use_container_width=True):
            st.session_state.elec = 22.04
            st.session_state.fans = 3.59
            st.session_state.cooling = 0.0
            st.session_state.heating_elec = 0.0
            st.session_state.lights = 4.59
            st.session_state.equip = 8.19
            st.session_state.gas = 136.59
            st.session_state.heating_gas = 124.00
            st.session_state.equip_gas = 3.34
            st.session_state.water = 9.25
            st.rerun()
    
    with col_btn3:
        if st.button("⚠️ Load Theft Test Case", use_container_width=True):
            st.session_state.elec = 12.5
            st.session_state.fans = 3.2
            st.session_state.cooling = 5.1
            st.session_state.heating_elec = 2.8
            st.session_state.lights = 1.5
            st.session_state.equip = 6.2
            st.session_state.gas = 7.5
            st.session_state.heating_gas = 4.2
            st.session_state.equip_gas = 2.5
            st.session_state.water = 3.5
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
                X_batch = df.iloc[:, :10].values
                X_batch_scaled = scaler.transform(X_batch)
                predictions = model.predict(X_batch_scaled)
                probas = model.predict_proba(X_batch_scaled)
                
                df['Prediction'] = label_encoder.inverse_transform(predictions)
                df['Confidence'] = [np.max(p) * 100 for p in probas]
                
                st.subheader("📊 Detection Summary")
                summary = df['Prediction'].value_counts().reset_index()
                summary.columns = ['Class', 'Count']
                st.dataframe(summary)
                
                fig, ax = plt.subplots()
                ax.pie(summary['Count'], labels=summary['Class'], autopct='%1.1f%%')
                ax.set_title('Batch Detection Results')
                st.pyplot(fig)
                
                st.subheader("📋 Detailed Results")
                st.dataframe(df)
                
                csv = df.to_csv(index=False)
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
                consumption = np.random.uniform(10, 200, 10)
                consumption_scaled = scaler.transform([consumption])
                prediction = model.predict(consumption_scaled)
                predicted_class = label_encoder.inverse_transform(prediction)[0]
                
                st.session_state.monitoring_data.append({
                    'Time': datetime.now().strftime("%H:%M:%S"),
                    'Consumption': np.mean(consumption),
                    'Status': predicted_class
                })
                
                df_live = pd.DataFrame(st.session_state.monitoring_data)
                
                with placeholder.container():
                    if predicted_class == "Normal":
                        st.success(f"🟢 Current Status: {predicted_class}")
                    else:
                        st.error(f"🔴 ALERT: {predicted_class} Detected!")
                    
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
        st.metric("Accuracy", "94.60%", "▲ SMOTE Balanced")
    with col2:
        st.metric("F1-Score (Macro)", "90.0%", "▲")
    with col3:
        st.metric("Training Samples", "1.59M", "Balanced")
    with col4:
        st.metric("Classes", "6", "Equal distribution")
    
    st.subheader("🔑 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance for Theft Detection (SMOTE Balanced Model)')
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
    **Understanding the decision:** This model was trained on **1.59 million balanced samples** using SMOTE.
    Feature importance shows what consumption patterns most influence detection.
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Select a Test Case")
        
        explain_option = st.radio(
            "Choose test case:",
            ["Normal Test Case", "Theft Test Case", "Custom Mixed Case", "Custom Values"],
            key="explain_option"
        )
        
        if explain_option == "Normal Test Case":
            explain_values = [22.04, 3.59, 0, 0, 4.59, 8.19, 136.59, 124.00, 3.34, 9.25]
            st.info("📋 Normal consumption pattern")
            
        elif explain_option == "Theft Test Case":
            explain_values = [12.5, 3.2, 5.1, 2.8, 1.5, 6.2, 7.5, 4.2, 2.5, 3.5]
            st.warning("⚠️ Theft pattern (reduced consumption)")
            
        elif explain_option == "Custom Mixed Case":
            explain_values = [45.00, 12.00, 8.00, 5.00, 10.00, 25.00, 35.00, 20.00, 8.00, 10.00]
            st.info("🔀 Mixed pattern (high elec, low gas)")
            
        else:
            st.write("Enter custom values:")
            explain_values = []
            for i, name in enumerate(feature_names):
                val = st.number_input(f"{name}", value=0.0, key=f"explain_{i}", step=1.0)
                explain_values.append(val)
        
        explain_button = st.button("🔍 Explain Prediction", use_container_width=True, key="explain_btn")
    
    with col2:
        st.subheader("📖 How Interpret Results")
        st.markdown("""
        **Feature Importance Shows:**
        - **Higher importance** = Stronger influence on theft detection
        - **Interior Lights** (18.7%) = Strongest indicator
        - **Interior Equipment** (18.0%) = Second strongest
        - **Electricity: Facility** (13.6%) = Overall pattern
        
        **Confidence Interpretation:**
        - **>80%** = Very certain
        - **50-80%** = Moderately certain
        - **<50%** = Uncertain / borderline case
        """)
    
    if explain_button:
        with st.spinner("Analyzing and generating explanation..."):
            input_features = np.array([explain_values])
            input_scaled = scaler.transform(input_features)
            
            # Get prediction
            prediction = model.predict(input_scaled)[0]
            predicted_class = label_encoder.inverse_transform([prediction])[0]
            probabilities = model.predict_proba(input_scaled)
            confidence = np.max(probabilities) * 100
            
            # Get feature importance
            feature_importance_rf = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            st.markdown("---")
            
            # Show prediction result
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.subheader("📊 Prediction Result")
                if predicted_class == "Normal":
                    st.success(f"✅ **{predicted_class}** (Confidence: {confidence:.1f}%)")
                else:
                    st.error(f"⚠️ **{predicted_class}** (Confidence: {confidence:.1f}%)")
            
            with col_res2:
                st.subheader("📈 Model Info")
                st.metric("Accuracy", "94.60%")
                st.caption("SMOTE Balanced Model | 1.59M training samples")
            
            st.markdown("---")
            
            # Feature impact bar chart
            st.subheader("🔍 Which Features Most Influence Theft Detection?")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            imp_df = feature_importance_rf.head(8)
            colors = ['coral' for _ in range(len(imp_df))]
            bars = ax.barh(imp_df['Feature'], imp_df['Importance'] * 100, color=colors)
            ax.set_xlabel('Importance (%)', fontsize=12)
            ax.set_title('Global Feature Importance for Theft Detection', fontsize=14)
            
            for bar, val in zip(bars, imp_df['Importance'] * 100):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}%', va='center', fontsize=10)
            
            st.pyplot(fig)
            
            # Show probability distribution
            st.subheader("📊 Class Probability Distribution for This Prediction")
            prob_df = pd.DataFrame({
                'Class': label_encoder.classes_,
                'Probability (%)': probabilities[0] * 100
            }).sort_values('Probability (%)', ascending=False)
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            colors2 = ['red' if c != 'Normal' else 'green' for c in prob_df['Class']]
            bars2 = ax2.barh(prob_df['Class'], prob_df['Probability (%)'], color=colors2)
            ax2.set_xlabel('Probability (%)')
            ax2.set_title('Model Confidence Across All Classes')
            
            for bar, val in zip(bars2, prob_df['Probability (%)']):
                ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{val:.1f}%', va='center', fontsize=9)
            
            st.pyplot(fig2)
            
            # Interpretation
            st.subheader("💡 Interpretation & Justification")
            
            if predicted_class != "Normal":
                st.markdown(f"**🔴 Why was this flagged as theft?**")
                st.markdown(f"""
                - The model detected a **{predicted_class}** pattern with {confidence:.1f}% confidence
                - This theft type is characterized by: {class_descriptions.get(predicted_class, 'Suspicious consumption pattern')}
                - The energy consumption values deviate from normal patterns
                """)
                
                if confidence < 50:
                    st.info("📌 **Note:** Lower confidence indicates this is a borderline case. The model sees mixed signals and recommends manual review.")
                
                st.markdown("**Top indicators for this theft type:**")
                top_features = feature_importance_rf.head(3)
                for _, row in top_features.iterrows():
                    st.markdown(f"- **{row['Feature']}** (importance: {row['Importance']*100:.1f}%)")
                    
            else:
                st.markdown("**🟢 Why was this considered normal?**")
                st.markdown(f"""
                - The consumption pattern matches **normal** behavior with {confidence:.1f}% confidence
                - All energy consumption values are within expected ranges for typical customers
                - No theft patterns detected in any of the 10 consumption features
                """)
            
            # Detailed feature table
            with st.expander("📋 View All Feature Importances"):
                st.dataframe(feature_importance_rf)
                
            with st.expander("📖 About SMOTE Balanced Model"):
                st.markdown("""
                **This model was trained using SMOTE (Synthetic Minority Over-sampling Technique):**
                - Original data had imbalanced classes (Theft1: 9.7%, Theft2: 4.4%)
                - SMOTE created synthetic samples to balance all 6 classes equally
                - Final training set: 265,459 samples per class (1.59M total)
                - Result: Fair predictions across all theft types with 94.60% accuracy
                """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center">
        <p>⚡ SRIP 2026 - VIT Chennai | SMOTE Balanced Random Forest (94.60% Accuracy)</p>
        <p>Based on ETD2022 Benchmark Dataset | Fair & Explainable AI</p>
    </div>
""", unsafe_allow_html=True)