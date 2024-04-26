import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from prophet import Prophet

# Define heart_failure_monthly_count dictionary
heart_failure_monthly_count = {
    1: 21,
    2: 18,
    3: 11,
    4: 21,
    5: 20,
    6: 16
}

# Function to load the current dataset of infrastructure
def load_infrastructure_dataset():
    uploaded_file = st.file_uploader("Upload current infrastructure dataset (CSV)", type=["csv"], key="infrastructure_dataset")
    if uploaded_file is not None:
        df_infrastructure = pd.read_csv(uploaded_file)
        return df_infrastructure
    return None

# Function to load the future trend of heart failure patients
def load_future_trend_dataset():
    uploaded_file = st.file_uploader("Upload future trend dataset (CSV)", type=["csv"], key="future_trend_dataset")
    if uploaded_file is not None:
        df_future_trend = pd.read_csv(uploaded_file)
        # Convert 'Month' column to datetime format
        df_future_trend['Month'] = pd.to_datetime(df_future_trend['Month'])
        return df_future_trend
    return None

# Function to prepare data for Random Forest model
def prepare_data_for_rf(df_future_trend):
    X = df_future_trend['Month'].values.reshape(-1, 1)  # Months
    y = df_future_trend['trend'].values.astype(float) # Number of heart failure patients
    return X, y

# Function to train the Random Forest model
def train_rf_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Function to make predictions for the future trend
def make_predictions(rf_model):
    forecast_values = rf_model.predict(np.array(range(7, 13)).reshape(-1, 1))  # Forecast for 6 months (July to December)
    return forecast_values

# Function to generate suggestions for infrastructure enhancement
def generate_infrastructure_suggestions(df_infrastructure, forecast_values):
    # Calculate average infrastructure for January to June
    avg_infrastructure = df_infrastructure.mean()

    # Generate suggestions based on forecasted patient numbers and average infrastructure
    suggestions = {}

    for month, forecasted_patients in zip(range(7, 13), forecast_values):
        suggested_infrastructure = {}
        for col in df_infrastructure.columns[1:]:  # Exclude 'Month' column
            if avg_infrastructure[col] < forecasted_patients:
                shortfall = forecasted_patients - avg_infrastructure[col]
                suggested_infrastructure[col] = f"Increase by {shortfall:.2f} units if needed"
            else:
                suggested_infrastructure[col] = "Current infrastructure seems adequate"
        suggestions[month] = suggested_infrastructure

    return suggestions

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
def make_prophet_predictions(data, disease):
    disease_data = data[data['disease diagnosed'] == disease]
    monthly_disease_counts = disease_data.resample('M').size().reset_index()
    monthly_disease_counts.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(monthly_disease_counts)
    
    future_dates = model.make_future_dataframe(periods=6, freq='M', include_history=False)
    forecast = model.predict(future_dates)
    
    return model, forecast

# Function to analyze dataset and provide suggestions
def analyze_infrastructure(df):
    # Initialize dictionary to store suggestions
    suggestions = {}

    # Analyze the dataset and provide suggestions for each column
    for month, patient_count in heart_failure_monthly_count.items():
        # Suggestions for each column
        suggestions[month] = {
            'Beds in ICU': f"Increase by {max(0, patient_count - df['Number of Beds in ICU'].sum())} units if needed",
            'Nurses and Wardboys': f"Increase by {max(0, patient_count - df['Number of Nurses and Wardboys'].sum())} units if needed",
            'Test Equipments': f"Increase by {max(0, patient_count - df['Number of Test Equipments'].sum())} units if needed",
            'Ventilators': f"Increase by {max(0, patient_count - df['Number of Ventilators'].sum())} units if needed",
            'Medicines and Injections': f"Increase by {max(0, patient_count - df['Number of Medicines and Injections'].sum())} units if needed"
        }
    
    return suggestions

# Function to analyze dataset and calculate performance scores
def analyze_dataset(df):
    # Initialize lists to store performance scores and ratings
    performance_scores = []
    ratings = []

    # Define a non-linear mapping function from performance score to rating
    def map_to_rating(performance_score):
        if 1 < performance_score < 3.5:
            return "⭐⭐"
        elif 3.5 < performance_score < 4.0:
            return "⭐⭐⭐"
        elif 4 < performance_score < 4.5:
            return "⭐⭐⭐⭐"
        elif 4.5 < performance_score < 5.0:
            return "⭐⭐⭐⭐⭐"
        else:
            return "⭐⭐⭐"
        
    # Analyze the dataset and calculate performance score for each month
    for month, patient_count in heart_failure_monthly_count.items():
        # Filter the dataset for the current month
        month_data = df[df['Month'] == month]

        # Calculate the total infrastructure provided for the month
        total_infrastructure = (
            month_data['Number of Beds in ICU'].sum() +
            month_data['Number of Nurses and Wardboys'].sum() +
            month_data['Number of Test Equipments'].sum() +
            month_data['Number of Ventilators'].sum() +
            month_data['Number of Medicines and Injections'].sum()
        )

        # Calculate the performance score (ratio of infrastructure to patients)
        performance_score = total_infrastructure / patient_count

        # Map performance score to rating using non-linear mapping function
        rating = map_to_rating(performance_score)

        # Add performance score and rating to the lists
        performance_scores.append(performance_score)
        ratings.append(rating)

    return performance_scores, ratings


# Streamlit UI
def main():
    st.title("Welcome to CareForcast")

    # Options on the screen
    st.subheader("Select an Operation:")
    operation = st.selectbox("", ("Data Analysis", "Prediction", "Suggestions and Performance Analysis", "Future Infrastructure Suggestion"))

    if operation == "Data Analysis":
        # File upload for data analysis
        st.subheader("Data Analysis")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="datafile")
        if uploaded_file is not None:
            try:
                hospital_data = pd.read_csv(uploaded_file)
                st.subheader("Sample of uploaded data:")
                st.write(hospital_data.head())

                # Preprocess data for plot generation
                hospital_data_plot = preprocess_data_plot(hospital_data)
                # Display plot
                st.subheader("Trend Analysis")
                generate_plot(hospital_data_plot)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif operation == "Prediction":
        # File upload for prediction
        st.subheader("Prediction")
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="datafile")
        if uploaded_file is not None:
            try:
                hospital_data = pd.read_csv(uploaded_file)
                st.subheader("Sample of uploaded data:")
                st.write(hospital_data.head())

                # Sub options for prediction
                disease_options = hospital_data['disease diagnosed'].unique()
                disease = st.selectbox("Select the disease for prediction:", disease_options)

                # Preprocess data for prediction
                hospital_data_predict = preprocess_data_predict(hospital_data)

                if st.button("Generate Predictions"):
                    st.write(f"Generating predictions for {disease}...")
                    model, forecast = make_prophet_predictions(hospital_data_predict, disease)
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

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif operation == "Suggestions and Performance Analysis":
        # Load suggestion file
        st.subheader("Infrastructure Suggestions")
        uploaded_suggestion_file = st.file_uploader("Upload Suggestion CSV file", type=["csv"], key="suggestion")
        if uploaded_suggestion_file is not None:
            try:
                suggestion_data = pd.read_csv(uploaded_suggestion_file)
                st.write("Sample of uploaded suggestion data:")
                st.write(suggestion_data.head())

                # Provide suggestions
                suggestions = analyze_infrastructure(suggestion_data)

                # Display suggestions
                for month, infra_suggestions in suggestions.items():
                    st.write(f"### Suggestions for Month {month}:")
                    for column, suggestion in infra_suggestions.items():
                        st.write(f"- For {column}: {suggestion}")

                # Performance Analysis sub-option
                if st.button("Performance Analysis"):
                    # Performance Analysis
                    st.subheader("Performance Analysis")
                    st.write("Analyzing performance...")
                    performance_scores, ratings = analyze_dataset(suggestion_data)

                    if performance_scores is not None:
                        st.subheader("Performance Scores")
                        data = {
                            'Month': list(range(1, 7)),
                            'Performance Score': performance_scores,
                        }
                        df_display = pd.DataFrame(data)
                        st.write(df_display)

                        # Plot performance scores
                        st.subheader("Performance Scores Over Months")
                        months = list(range(1, 7))
                        plt.figure(figsize=(6, 4))
                        plt.plot(months, performance_scores, marker='o', linestyle='-')
                        plt.xlabel('Month')
                        plt.ylabel('Performance Score')
                        plt.xticks(months)
                        st.pyplot(plt)

                        # Display ratings
                        st.subheader("Ratings Based on Performance Score")
                        for i, rating in enumerate(ratings):
                            st.write(f"Month {i+1}: {rating}")

                        # Plot ratings
                        st.subheader("Ratings Based on Performance Score")
                        months = list(range(1, 7))
                        plt.figure(figsize=(6, 4))
                        plt.plot(months, ratings, marker='o', linestyle='-')
                        plt.xlabel('Month')
                        plt.ylabel('Rating')
                        plt.xticks(months)
                        st.pyplot(plt)

            except Exception as e:
                st.error(f"An error occurred: {e}")

    elif operation == "Future Infrastructure Suggestion":
        # File upload for current infrastructure dataset
        df_infrastructure = load_infrastructure_dataset()

        # File upload for future trend dataset
        df_future_trend = load_future_trend_dataset()

        if df_infrastructure is not None and df_future_trend is not None:
            try:
                # Prepare data for Random Forest model
                X, y = prepare_data_for_rf(df_future_trend)

                # Split the data into training and testing sets
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train the Random Forest model
                rf_model = train_rf_model(X_train, y_train)

                if st.button("Generate Infrastructure Suggestions"):
                    st.write("Generating infrastructure suggestions...")
                    # Make predictions for the future trend
                    forecast_values = make_predictions(rf_model)

                    # Generate suggestions for infrastructure enhancement
                    suggestions = generate_infrastructure_suggestions(df_infrastructure, forecast_values)

                    # Print suggestions for infrastructure enhancement
                    for month, infra_suggestions in suggestions.items():
                        st.write(f"Suggestions for month {month}:")
                        for col, suggestion in infra_suggestions.items():
                            st.write(f"- For {col}: {suggestion}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    else:
        st.write("Please select an operation.")

if __name__ == "__main__":
    main()


st.set_option('deprecation.showPyplotGlobalUse', False)
