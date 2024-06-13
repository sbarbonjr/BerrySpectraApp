import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

model_firmness = joblib.load('streamlit/models/Firmness (gf)_random_forest_model.pkl')

model_acidity = joblib.load('streamlit/models/Titratable acidity (%)_random_forest_model.pkl')

def predict_brix(dataset):
    model_brix = joblib.load('streamlit/models/SSC (°Brix)_random_forest_model.pkl')
    # Predict using the loaded model
    X = dataset.values
    y_hat = model_brix.predict(X)
    return y_hat



def validate_csv(file):
    """Validate the uploaded CSV file to ensure it has the expected structure."""
    try:
        df = pd.read_csv(file)
        expected_columns = ['Wavelength', 'Absorbance']
        if not all(column in df.columns for column in expected_columns):
            st.error(f"CSV file is missing required columns: {expected_columns}")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None

def validate_excel(file):
    """Validate the uploaded Excel file to ensure it has the expected structure."""
    try:
        df = pd.read_excel(file, header=1, sheet_name=1)
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

def process_data(df):
    """Process the data, perform predictions, and create visualizations."""
    st.subheader("Data Overview")
    st.write(df.head())

    st.subheader("Absorbance Spectrum")
    fig = px.line(df, x='Wavelength', y='Absorbance', title='Absorbance Spectrum')
    st.plotly_chart(fig)

    # Hypothetical feature extraction for predictions
    # Assuming the model requires specific preprocessing
    features = df['Absorbance'].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

        
    # Display predictions
    st.subheader("Predicted Components")
    st.write(f"Predicted BRIX Content: {predict_brix:.2f} %")
    # st.write(f"Predicted Acid Content: {acid_prediction.mean():.2f} %")

def main():
    st.title("Berry Spectra App")
    st.subheader("Coordinator: Dr. Raúl Siche Jara")
    st.subheader("FONDECYT: 75765")


    st.write("Upload a CSV or Excel file containing berry NIR spectra data for processing and prediction.")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = validate_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = validate_excel(uploaded_file)
        
        if df is not None:
            process_data(df)

            st.subheader("Line Plot of Data")
            df_melted = pd.melt(df.reset_index(), id_vars=["index", "Crop year", "Variety", "Maturation stage", "SSC (°Brix)"])
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x='variable', y='value', data=df_melted, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
