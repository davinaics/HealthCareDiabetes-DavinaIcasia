import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

st.title('Healthcare Diabetes Prediction App')

# Load model 
try:
    loaded_dt_model = load_model('decision_tree_model')
    st.success("Decision Tree model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Form input user 
st.header("Enter Patient Data:")

pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=1)
glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100)
blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=130, value=70)
skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20)
insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80)
bmi = st.number_input('BMI', min_value=0.0, max_value=60.0, value=25.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=1, max_value=120, value=30)

# Tombol prediksi 
if st.button('Predict'):
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree_function],
        'Age': [age]
    })

    try:
        prediction_result = predict_model(loaded_dt_model, data=input_data)
        pred_label = prediction_result['prediction_label'][0]

        st.subheader("Prediction Result:")
        if pred_label == 1:
            st.error("⚠️ The patient is predicted to have diabetes.")
        else:
            st.success("✅ The patient is predicted NOT to have diabetes.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")