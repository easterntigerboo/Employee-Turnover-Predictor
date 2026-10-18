import streamlit as st
import pandas as pd
import pickle

# 1. Title and Layout of your Web App
st.set_page_config(page_title="HR Attrition Predictor", layout="centered")
st.title("📊 Employee Flight Risk Calculator")
st.write("Input an employee's metrics below to calculate their real-time attrition risk.")

# 2. Reload your pre-trained models from disk
@st.cache_resource # Keeps the model loaded in memory for fast performance
def load_assets():
    with open('final_logreg_turnover_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('final_turnover_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# 3. Create interactive Web Inputs (Sliders & Dropdowns)
st.header("👤 Employee Metrics")

col1, col2 = st.columns(2)

with col1:
    satisfaction = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, step=0.01)
    evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.7, step=0.01)
    tenure = st.slider("Tenure (Years at Company)", 1, 10, 3)

with col2:
    projects = st.number_input("Number of Assigned Projects", min_value=1, max_value=10, value=3)
    hours = st.number_input("Average Monthly Hours", min_value=50, max_value=350, value=160)
    salary_label = st.selectbox("Salary Tier", ["Low", "Medium", "High"])
    accident = st.radio("Has had a Workplace Accident?", ["No", "Yes"])

# Convert labels back to our math mapping
salary_map = {"Low": 0, "Medium": 1, "High": 2}
salary_encoded = salary_map[salary_label]
accident_encoded = 1 if accident == "Yes" else 0

# 4. Trigger Prediction on Button Click
if st.button("🔮 Calculate Retention Risk", type="primary"):
    # Structure data perfectly for our model pipeline
    new_data = {
        'satisfaction_level': satisfaction,
        'number_project': projects,
        'average_montly_hours': hours,
        'time_spend_company': tenure,
        'Work_accident': accident_encoded,
        'salary_encoded': salary_encoded,
        'last_evaluation_squared': evaluation ** 2 # Our critical step 3 fix!
    }
    
    df_new = pd.DataFrame([new_data])
    feature_order = ['satisfaction_level', 'number_project', 'average_montly_hours', 
                     'time_spend_company', 'Work_accident', 'salary_encoded', 'last_evaluation_squared']
    
    # Scale and predict
    X_scaled = scaler.transform(df_new[feature_order])
    risk_percentage = model.predict_proba(X_scaled)[0][1] * 100
    
    # 5. Display the Web UI Results
    st.subheader(f"Risk Rating: {risk_percentage:.1f}%")
    if risk_percentage >= 75:
        st.error("🚨 CRITICAL flight risk status. HR retention intervention is recommended.")
    elif 50 <= risk_percentage < 75:
        st.warning("⚠️ MODERATE risk detected. Consider conducting a stay-interview.")
    else:
        st.success("✅ STABLE track. This employee shows clean retention signals.")
