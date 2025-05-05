# Import libraries
import pandas as pd      
import numpy as np       
import pickle
import streamlit as st 
from catboost import CatBoostRegressor    
import gdown 
import os

# Columns
columns = [
    'number_of_bedrooms', 'number_of_bathrooms', 'living_area', 'lot_area', 'built_year', 'number_of_floors',
    'grade_of_the_house', 'number_of_views', 'postal_code', 'lattitude', 'longitude', 'living_area_renov', 'lot_area_renov',
    'area_of_the_house(excluding_basement)', 'area_of_the_basement'
]

# Load the Random Forest model
url = "https://drive.google.com/uc?id=18xA5kOK4aQQO0UNwWqZ-Qub6Iu8XTtdH"
file_id = "18xA5kOK4aQQO0UNwWqZ-Qub6Iu8XTtdH"
output = 'random_forest_model.pkl'
gdown.download(url, output, quiet=False)
with open(output, 'rb') as file:
    model_rf = pickle.load(file)

# Load the Gradient Boosting model
pickle_gb = open('gradient_boosting_model.pkl', 'rb')
model_gb = pickle.load(pickle_gb)

# Load the CatBoost model
pickle_cb = open('cat_boost_model.pkl', 'rb')
model_cb = pickle.load(pickle_cb)

# Function to perform prediction
def predict(data_frame):
    predict_rf = model_rf.predict(data_frame)
    predict_gb = model_gb.predict(data_frame)
    stack_pred = np.column_stack((predict_rf, predict_gb))
    predict_cb = model_cb.predict(stack_pred)
    return predict_cb


# Function to accept inputs and predict from inputs
def main():
    # Set up the application
    st.title('House Price Predictor')
    html_temp = '''
    <div style="background:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;"> House Price Prediction Prediction App </h2>
    </div>
    '''
    st.markdown(html_temp, unsafe_allow_html=True)

    # Specify the inputs
    st.write("Please enter values for all the parameters!!!")
    num_of_bedrooms = st.text_input("Number of bedrooms: ") 
    num_of_bathrooms = st.text_input("Number of bathrooms: ") 
    living_area = st.text_input("Living area: ")
    lot_area = st.text_input("Lot area: ")
    built_year = st.text_input("Built year: ")
    num_of_floors = st.text_input("Number of floors: ")
    house_grade = st.text_input("Grade of the house: ") 
    num_of_views = st.text_input("Number of views: ")
    postal_code = st.text_input("Postal code: ")
    latitude = st.text_input("Latitude: ")
    longitude = st.text_input("Longitude: ")
    living_area_renov = st.text_input("Living area renovated: ")
    lot_area_renov = st.text_input("Lot area renovated: ")
    house_area_excl_bsmt = st.text_input("Area of the house (excluding basement): ")
    bsmt_area = st.text_input("Basement area: ")


    # Combine into an input data
    data_dict = {
        'number_of_bedrooms': [num_of_bedrooms],
        'number_of_bathrooms': [num_of_bathrooms],
        'living_area': [living_area],
        'lot_area': [lot_area],
        'built_year': [built_year], 
        'number_of_floors': [num_of_floors],
        'grade_of_the_house': [house_grade],
        'number_of_views': [num_of_views],
        'postal_code': [postal_code],
        'lattitude': [latitude],
        'longitude': [longitude],
        'living_area_renov': [living_area_renov],
        'lot_area_renov': [lot_area_renov],
        'area_of_the_house(excluding_basement)': [house_area_excl_bsmt],
        'area_of_the_basement': [bsmt_area],      
    }
    df = pd.DataFrame(data_dict)

    # Predict button
    if st.button('Predict'):
        result = predict(df)
        st.success(f"House price is {result[0]}.")

# Run the application
if __name__ == '__main__': 
    main() 




