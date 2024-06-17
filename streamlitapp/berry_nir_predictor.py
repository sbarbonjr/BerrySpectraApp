import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

from core.modeling import normalize_spectral_data, snv, first_derivative, second_derivative

PATH = os.path.dirname(os.path.realpath(__file__))
#model_firmness = joblib.load('models/Firmness (gf)_random_forest_model.pkl')
#model_acidity = joblib.load('models/Titratable acidity (%)_random_forest_model.pkl')

def predict_brix(X):
    target = 'SSC (°Brix)'
    max_val = joblib.load(PATH+"/models/"+target+'_max_val_norm.pkl')
    min_val = joblib.load(PATH+"/models/"+target+'_min_val_norm.pkl')
    dataset_, _, _ = normalize_spectral_data(X, max_val, min_val)
    
    mean_spectrum_snv = joblib.load(PATH+"/models/"+target+'_mean_spectrum_snv.pkl')
    mean_spectrum_std = joblib.load(PATH+"/models/"+target+'_mean_spectrum_std.pkl')
    dataset_, _, _ = snv(dataset_, mean_spectrum_snv, mean_spectrum_std)

    dataset_ = first_derivative(dataset_)#

    pls_ = joblib.load(PATH+"/models/"+target+'_PLS.pkl')
    lv_features = pls_.transform(dataset_)

    model = joblib.load(PATH+'/models/'+target+'_random_forest_model.pkl')
    # Predict using the loaded model of a RandomForestRegressor
    y_hat = model.predict(lv_features)
    #print('y_hat', y_hat)
    return y_hat.mean()


def predict_firm(X):
    target = 'Firmness (gf)'
    max_val = joblib.load(PATH+"/models/"+target+'_max_val_norm.pkl')
    min_val = joblib.load(PATH+"/models/"+target+'_min_val_norm.pkl')
    dataset_, _, _ = normalize_spectral_data(X, max_val, min_val)

    dataset_ = second_derivative(dataset_)#

    pls_ = joblib.load(PATH+"/models/"+target+'_PLS.pkl')
    lv_features = pls_.transform(dataset_)

    model = joblib.load(PATH+'/models/'+target+'_random_forest_model.pkl')
    # Predict using the loaded model of a RandomForestRegressor
    y_hat = model.predict(lv_features)
    #print('y_hat', y_hat)
    return y_hat.mean()

def predict_ta(X):
    target = 'Titratable acidity (%)'
    max_val = joblib.load(PATH+"/models/"+target+'_max_val_norm.pkl')
    min_val = joblib.load(PATH+"/models/"+target+'_min_val_norm.pkl')
    dataset_, _, _ = normalize_spectral_data(X, max_val, min_val)

    dataset_ = second_derivative(dataset_)#

    model = joblib.load(PATH+'/models/'+target+'_random_forest_model.pkl')
    # Predict using the loaded model of a RandomForestRegressor
    y_hat = model.predict(dataset_)
    #print('y_hat', y_hat)
    return y_hat.mean()




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

    with st.expander("Data Overview"):
        st.dataframe(df)

    st.subheader("Absorbance Spectrum")

    # Load mean spectra data
    mean_spectra = pd.read_csv(PATH + "/models/MeanSpectra.csv")
    
    # Plot both mean spectrum and data spectrum on the same plot
    fig = px.line(title='Absorbance Spectrum')
    fig.add_scatter(x=df['Wavelength'], y=df['Absorbance'], mode='lines', name='Data Spectrum')
    fig.add_scatter(x=mean_spectra['Wavelength'], y=mean_spectra['Absorbance'], mode='lines', name='Mean Spectra')
    
    # Customize layout and display plot
    fig.update_layout(title='Absorbance Spectrum Comparison',
                      xaxis_title='Wavelength',
                      yaxis_title='Absorbance')
    
    st.plotly_chart(fig)

    # Hypothetical feature extraction for predictions
    # Assuming the model requires specific preprocessing
    df_sample = df['Absorbance'].values.reshape(-1, 1)

    # Display predictions
    st.subheader("Predicted Components")
    st.write(f"Predicted SSC (°Brix) content: {predict_brix(df_sample):.2f} % (R² 0.77 / RMSE 1.12)")
    st.write(f"Predicted Firmness(gf) content: {predict_firm(df_sample):.2f} % (R² 0.92 / RMSE 0.49)")
    st.write(f"Predicted Titratable acidity (%) content: {predict_ta(df_sample):.2f} % (R² 0.59 / RMSE 149.70)")

def main():
    st.title("Berry Spectra App")

    markdown_text = '''
    <div style="border: 1px solid #ddd; background-color: #f0f0f0; padding: 10px; border-radius: 5px;">
        <h3>Coordinator: Dr. Raúl Siche Jara</h3>
        <h5>FONDECYT: 75765</h5>
    </div> <hr>
    '''

    # Display the Markdown text with HTML tags and allow unsafe HTML for custom styling
    st.markdown(markdown_text, unsafe_allow_html=True)

    st.subheader("Load Data Spectrum")
    st.write("Upload a CSV or Excel file containing berry NIR spectra data for processing and prediction. :")
    st.write("Example: [link](https://github.com/sbarbonjr/BerrySpectraApp/blob/master/datasets/MeanSpectra_SSC%20(%C2%B0Brix)_TEST_7%2C2.csv)")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df = validate_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = validate_excel(uploaded_file)
        
        if df is not None:
            process_data(df)
            #st.subheader("Line Plot of Data")
            #fig, ax = plt.subplots(figsize=(10, 6))
            #mean_spectra =  pd.read_csv(PATH+"/models/MeanSpectra.csv")
            #sns.lineplot(x='Wavelength', y='Absorbance', data=mean_spectra, ax=ax)
            #sns.lineplot(x='Wavelength', y='Absorbance', data=df, ax=ax)
            #ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            #st.pyplot(fig)

if __name__ == "__main__":
    main()
