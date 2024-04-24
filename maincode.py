import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import streamlit as st

# Function to preprocess data for plot generation
def preprocess_data_plot(hospital_data):
    hospital_data['Date of Admission'] = pd.to_datetime(hospital_data['Date of Admission'])
    hospital_data['Month'] = hospital_data['Date of Admission'].dt.month
    hospital_data = hospital_data[hospital_data['Month'].between(1, 6)]
    return hospital_data

# Function to preprocess data for prediction
def preprocess_data_predict(hospital_data):
    hospital_data['Date of Admission'] = pd.to_datetime(hospital_data['Date of Admission'])
    hospital_data.set_index('Date of Admission', inplace=True)
    return hospital_data

# Function to generate and display the plot
def generate_plot(hospital_data):
    disease_counts = hospital_data.groupby(['Month', 'disease diagnosed']).size().unstack(fill_value=0)
    plt.figure(figsize=(12, 8))
    disease_counts.plot(kind='line', marker='o')
    plt.title('Trend of Patients Based on Diseases (January to June)')
    plt.xlabel('Month')
    plt.ylabel('Number of Patients')
    plt.xticks(range(1, 7), ['January', 'February', 'March', 'April', 'May', 'June'])
    plt.legend(title='Disease', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot()

# Function to train Prophet model and make predictions
def make_predictions(data, disease):
    disease_data = data[data['disease diagnosed'] == disease]
    monthly_disease_counts = disease_data.resample('M').size().reset_index()
    monthly_disease_counts.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(monthly_disease_counts)
    
    future_dates = model.make_future_dataframe(periods=6, freq='M', include_history=False)
    forecast = model.predict(future_dates)
    
    return model, forecast

# Function to analyze dataset and provide suggestions
def analyze_infrastructure(df, heart_failure_monthly_count):
    # Initialize dictionary to store suggestions
    suggestions = {}

    # Analyze the dataset and provide suggestions for each column
    for month, patient_count in heart_failure_monthly_count.items():
        # Filter the dataset for the current month
        month_data = df[df['Month'] == month]

        # Calculate the total infrastructure provided for the month
        total_beds_icu = month_data['Number of Beds in ICU'].sum()
        total_nurses_wardboys = month_data['Number of Nurses and Wardboys'].sum()
        total_test_equipments = month_data['Number of Test Equipments'].sum()
        total_ventilators = month_data['Number of Ventilators'].sum()
        total_medicines_injections = month_data['Number of Medicines and Injections'].sum()

        # Suggestions for each column
        suggestions[month] = {
            'Beds in ICU': f"Increase by {max(0, patient_count - total_beds_icu)} units if needed",
            'Nurses and Wardboys': f"Increase by {max(0, patient_count - total_nurses_wardboys)} units if needed",
            'Test Equipments': f"Increase by {max(0, patient_count - total_test_equipments)} units if needed",
            'Ventilators': f"Increase by {max(0, patient_count - total_ventilators)} units if needed",
            'Medicines and Injections': f"Increase by {max(0, patient_count - total_medicines_injections)} units if needed"
        }
    
    return suggestions

# Streamlit UI
def main():
    st.title("Welcome to CareForcast")

    # Sidebar for file upload and options
    st.sidebar.title("Options")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            hospital_data = pd.read_csv(uploaded_file)
            st.subheader("Sample of uploaded data:")
            st.write(hospital_data.head())

            # Options in sidebar
            option = st.sidebar.radio("Select an option:", ("Data Analysis", "Suggestions"))

            if option == "Data Analysis":
                # Sub options for analysis
                analysis_option = st.sidebar.radio("Select Analysis or Prediction", ("Current Analysis", "Trend"))

                if analysis_option == "Current Analysis":
                    # Preprocess data for plot generation
                    hospital_data_plot = preprocess_data_plot(hospital_data)
                    # Display plot
                    st.subheader("Trend Analysis")
                    generate_plot(hospital_data_plot)
                else:
                    # Sub options for prediction
                    st.subheader("Prediction")
                    disease_options = hospital_data['disease diagnosed'].unique()
                    disease = st.selectbox("Select the disease for prediction:", disease_options)

                    # Preprocess data for prediction
                    hospital_data_predict = preprocess_data_predict(hospital_data)

                    if st.button("Generate Predictions"):
                        st.write(f"Generating predictions for {disease}...")
                        model, forecast = make_predictions(hospital_data_predict, disease)
                        # Plot predicted trend
                        plt.figure(figsize=(10, 6))
                        model.plot(forecast, xlabel='Date', ylabel='Number of Patients', ax=plt.gca())
                        plt.title(f'Predicted Trend of {disease} Patients (July to December 2024)')
                        plt.grid(True)
                        st.pyplot()
                        # Display month-wise predicted forecast values
                        forecast_monthly = forecast.set_index('ds').resample('M').sum()
                        st.write("\nMonth-wise Predicted Forecast Values:\n")
                        st.write(forecast_monthly[['yhat']].rename(columns={'yhat': 'Forecasted Patients'}))
                        # Save forecast data to CSV
                        if st.button("Save Forecast Data"):
                            st.write("Saving forecasted trend to 'forecasted_trend.csv'...")
                            forecast.to_csv('forecasted_trend.csv', index=False)

            elif option == "Suggestions":
                # Load suggestion file
                st.subheader("Infrastructure Suggestions")
                uploaded_suggestion_file = st.file_uploader("Upload Suggestion CSV file", type=["csv"])
                if uploaded_suggestion_file is not None:
                    try:
                        suggestion_data = pd.read_csv(uploaded_suggestion_file)
                        st.write("Sample of uploaded suggestion data:")
                        st.write(suggestion_data.head())

                        # Define the month-wise count of patients (dummy data)
                        heart_failure_monthly_count = {
                            1: 21,
                            2: 18,
                            3: 11,
                            4: 21,
                            5: 20,
                            6: 16
                        }

                        # Provide suggestions
                        suggestions = analyze_infrastructure(suggestion_data, heart_failure_monthly_count)

                        # Display suggestions
                        for month, infra_suggestions in suggestions.items():
                            st.write(f"### Suggestions for Month {month}:")
                            for column, suggestion in infra_suggestions.items():
                                st.write(f"- For {column}: {suggestion}")

                    except Exception as e:
                        st.error(f"An error occurred while processing the suggestion file: {e}")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()



st.set_option('deprecation.showPyplotGlobalUse', False)
