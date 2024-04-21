# careforcast

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load CSV file and process data
def load_data(file_paths):
    data_frames = []
    for file_path in file_paths:
        data_frames.append(pd.read_csv(file_path))
    return data_frames

# Function for trend prediction
def predict_trend(hospital_data, disease):
    for df in hospital_data:
        df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
        df.set_index('Date of Admission', inplace=True)
    
    # Combine all data frames into a single one
    combined_df = pd.concat(hospital_data)
    disease_data = combined_df[combined_df['disease diagnosed'] == disease]
    monthly_disease_counts = disease_data.resample('M').size().reset_index()
    monthly_disease_counts.columns = ['ds', 'y']

    # Train a Prophet model
    model = Prophet()
    model.fit(monthly_disease_counts)

    # Make future predictions for July to December 2024
    future_dates = model.make_future_dataframe(periods=6, freq='M', include_history=False)
    forecast = model.predict(future_dates)

    # Plot the predicted trend
    st.write(f'Predicted Trend of {disease} Patients (July to December 2024)')
    st.write(forecast[['ds', 'yhat']])
    fig = model.plot(forecast)
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("Hospital Data Analysis")
    st.sidebar.title("Options")

    # Sidebar - Upload CSV files
    uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=['csv'], accept_multiple_files=True)

    if uploaded_files:
        hospital_data = load_data(uploaded_files)

        # Sidebar - Select disease for prediction
        st.sidebar.subheader("Trend Prediction")
        selected_disease = st.sidebar.selectbox("Select Disease for Prediction", hospital_data[0]['disease diagnosed'].unique())

        # Perform trend prediction
        predict_trend(hospital_data, selected_disease)

if __name__ == "__main__":
    main()





