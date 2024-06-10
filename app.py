import streamlit as st
import pandas as pd
import dtale
from dtale.app import build_app
from pycaret.regression import setup, compare_models, pull
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Define the Streamlit app
def main():
    st.sidebar.title("Auto ML COEX")
    st.sidebar.info("This application helps you perform Automated Machine Learning.")
    st.sidebar.image("automl-image.jpg", use_column_width=True)

    # Main content
    st.title("Auto ML COEX")

    # Sidebar - File upload and data profiling
    uploaded_file = st.file_uploader("Upload your dataset (.csv)", type=["csv"])
    if uploaded_file is not None:
        @st.cache_data
        def load_data(file):
            return pd.read_csv(file)

        df = load_data(uploaded_file)

        # Display the uploaded dataset
        st.subheader("Uploaded Dataset")
        st.write(df)

        # Data Profiling with D-Tale
        st.subheader("Data Exploration")
        if st.button("Explore Data with D-Tale"):
            d = dtale.show(df)
            st.write(d.main_url())
        
        # Machine Learning
        st.subheader("Machine Learning")
        target_variable = st.selectbox("Select target variable", df.columns)
        
        if df[target_variable].dtype == 'object':
            le = LabelEncoder()
            df[target_variable] = le.fit_transform(df[target_variable])
        
        if st.button("Run AutoML"):
            st.info("Running AutoML...")
            setup(data=df, target=target_variable)
            best_model = compare_models()
            st.success("AutoML completed!")
            
            # Display best model and model performance
            st.subheader("Best Model")
            st.write(best_model)
            
            # Display model performance
            model_results = pull()
            st.subheader("Model Performance")
            st.write(model_results)
            
            # Plot model performance metrics
            plot_model_performance(model_results)

# Function to plot model performance metrics
def plot_model_performance(model_results):
    # Rename columns for better visualization
    model_results.rename(columns={'MAE': 'Mean Absolute Error (MAE)', 'MSE': 'Mean Squared Error (MSE)', 'RMSE': 'Root Mean Squared Error (RMSE)'}, inplace=True)
    
    # Plot model performance metrics
    plt.figure(figsize=(10, 6))
    sns.barplot(data=model_results)
    plt.title('Model Performance Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    st.pyplot()

# Run the app
if __name__ == "__main__":
    main()



#streamlit run "C:\Users\Alaxo joy\OneDrive\Desktop\automl\new2.py"

