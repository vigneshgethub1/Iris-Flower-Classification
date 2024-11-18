import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model and preprocessing objects
@st.cache_resource  # Cache the loading function to speed up the app
def load_model():
    try:
        # Attempt to load the model and preprocessing objects from the pickle file
        with open('iris_model.pkl', 'rb') as file:
            saved_objects = pickle.load(file)
        
        # Ensure all necessary objects are loaded from the pickle file
        model = saved_objects['model']
        scaler = saved_objects.get('scaler')
        pca = saved_objects.get('pca')
        
        # Debug print to check loaded objects
        print("Model, Scaler, and PCA loaded successfully!")
        
        return model, scaler, pca
    except Exception as e:
        # Display any errors that occur during the loading process
        st.error(f"Error loading model: {e}")
        print(f"Error loading model: {e}")
        return None, None, None

model, scaler, pca = load_model()

if model is None:
    st.stop()  # Stop execution if the model couldn't be loaded

# Define the Streamlit application
def main():
    # Set up the application title and description
    st.title("Iris Flower Classification App")
    st.write("""
        This application predicts the species of an Iris flower based on its measurements.
        Adjust the sliders below to input the flower's features.
    """)

    # User input sliders for flower features
    sepal_length = st.slider("Sepal Length (cm)", min_value=4.0, max_value=8.0, value=5.4, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", min_value=2.0, max_value=4.5, value=3.4, step=0.1)
    petal_length = st.slider("Petal Length (cm)", min_value=1.0, max_value=7.0, value=3.8, step=0.1)
    petal_width = st.slider("Petal Width (cm)", min_value=0.1, max_value=2.5, value=1.2, step=0.1)

    # Collect inputs into an array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Debugging: Check the shape of input data
    print(f"Input data shape: {input_data.shape}")

    # Add a dummy feature (if required by the model)
    dummy_feature = np.ones((input_data.shape[0], 1))  # Add a constant feature (e.g., 1)
    input_data = np.hstack((input_data, dummy_feature))  # Combine input data with dummy feature

    # Debugging: Check input data after adding the dummy feature
    print(f"Input data after adding dummy feature: {input_data.shape}")

    # Preprocess the input data (scaling and PCA if applicable)
    if scaler:
        try:
            input_data = scaler.transform(input_data)
            print("Scaling successful!")
        except Exception as e:
            st.error(f"Error during scaling: {e}")
            print(f"Error during scaling: {e}")
            return

    if pca:
        try:
            input_data = pca.transform(input_data)
            print("PCA transformation successful!")
        except Exception as e:
            st.error(f"Error during PCA transformation: {e}")
            print(f"Error during PCA transformation: {e}")
            return

    # Show the input data for confirmation
    st.write(f"Input data: Sepal Length = {sepal_length}, Sepal Width = {sepal_width}, Petal Length = {petal_length}, Petal Width = {petal_width}")

    # Predict the species using the model
    if st.button("Predict"):
        try:
            prediction = model.predict(input_data)
            species = ["Setosa", "Versicolor", "Virginica"]
            st.write(f"The predicted species is: **{species[prediction[0]]}**")
            print(f"Prediction: {species[prediction[0]]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            print(f"Error during prediction: {e}")

    # Display prediction probabilities if checkbox is selected
    if st.checkbox("Show Prediction Probabilities"):
        try:
            probabilities = model.predict_proba(input_data)
            prob_df = pd.DataFrame(probabilities, columns=species)
            st.write("Prediction Probabilities:")
            st.write(prob_df)
            print("Prediction probabilities displayed.")
        except Exception as e:
            st.error(f"Error during probability calculation: {e}")
            print(f"Error during probability calculation: {e}")

# Run the app
if __name__ == '__main__':
    main()
