import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd



## Load the trained model
model = tf.keras.models.load_model("model.h5")

with open("Label_encoder_gender.pkl","rb") as file:
    label_encoder_gender = pickle.load(file)
    
    with open("Onehot_encoder_geo.pkl","rb") as file:
     Onehot_encoder_geo = pickle.load(file)
     
     
    with open("scaler.pkl","rb") as file:
     scaler = pickle.load(file)
    
    st.title("Customer Churn Prediction")
    
   # user input
   
geography = st.selectbox("Geography",Onehot_encoder_geo.categories_[0])
    
gender = st.selectbox("Gender", label_encoder_gender.classes_)


age = st.slider("Age", 18,92)

balance = st.number_input("Balance")


credit_score = st.number_input("credit_score")

estimated_salary = st.number_input("Estimated salary")

tenure = st.slider("Tenure", 0, 10)


num_of_products = st.slider("Number of products",0,4)

has_cr_card = st.selectbox("Has credit card",[0,1])

is_active_member = st.selectbox("Is Active Member",[1,0])
    
 # prepare the input data 
 
input_data = pd.DataFrame({
    "CreditScore" : [credit_score],
    "Gender" : [label_encoder_gender.transform([gender])[0]],
    "Age" : [age],
    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts" : [num_of_products],
    "HasCrCard" : [has_cr_card],
    "IsActiveMember" : [is_active_member],
    "EstimatedSalary" : [estimated_salary],
    
}) 
 
geo_encoded = Onehot_encoder_geo.transform([[geography]]).toarray()

geo_encoded_df = pd.DataFrame(geo_encoded , columns=Onehot_encoder_geo.get_feature_names_out(["Geography"]))

  
  # combine one_hot encoded colums with  input data 
input_data =  pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# scale the input data
input_Data_scaled = scaler.transform(input_data)

prediction = model.predict((input_Data_scaled))
prediction_probs = prediction[0][0]

st.write(f"Churn Probability: {prediction_probs:.2f}")

if prediction_probs > 0.5:
   st.write("The Customer is likely to churn")
else:
    st.write("The Customer is not likely to churn")

  
  
  
  
  
    