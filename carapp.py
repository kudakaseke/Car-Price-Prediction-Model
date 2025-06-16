import numpy as np
import streamlit as st
import joblib

#Loading out model and scaler
model = joblib.load('C:/Users/kvda.kaseke Work/Desktop/car price pred/xgb_model.pkl')

scaler = joblib.load('C:/Users/kvda.kaseke Work/Desktop/car price pred/scaler.pkl')

def car_price_pred(input_data):
    #changing the input into numpy array and reshaping
    input_changed = np.array(input_data).reshape(1,-1)

    #standardize the input data
    std_input = scaler.transform(input_changed)

    prediction = model.predict(std_input)

    return "Estimated car price: " + str(prediction[0])

def main():
    #creating title
    st.title('Ford Car Price Prediction App')

    #getting the input from User
    year = st.text_input('Year')
    transmission = st.text_input('Transmission')
    milage = st.text_input('Mileage')
    fueltype = st.text_input('Fuel Type')
    tax = st.text_input('Tax')
    mpg = st.text_input('mpg')
    enginesize = st.text_input('Engine Size')

    pred_price = ''

    #create a button
    if st.button('Check estimated price'):
        pred_price =car_price_pred([year, transmission, milage, fueltype, tax, mpg, enginesize])

    st.success(pred_price)

if __name__ == '__main__':
    
    main()