import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
@st.cache_resource
def load_resources():
    model=tf.keras.models.load_model('regression_model.h5')
    with open('onehot_encoder_geo.pkl','rb') as file:
        onehot_encoder_geo=pickle.load(file)
    with open('label_encoder_gender.pkl','rb') as file:
        label_encoder_gender=pickle.load(file)
    with open('salary_scaler.pkl','rb') as file:
        scaler=pickle.load(file)
    return model,onehot_encoder_geo,label_encoder_gender,scaler
def run():
    model,onehot_encoder_geo,label_encoder_gender,scaler=load_resources()

    st.title('Salary Prediction')
    geography=st.selectbox('Geography',onehot_encoder_geo.categories_[0])
    gender=st.selectbox('Gender',label_encoder_gender.classes_)
    age=st.slider('Age',18,92)
    balance=st.number_input('Balance')
    credit_score=st.number_input('Credit Score')
    tenure=st.slider('Tenure',0,10)
    num_of_products=st.slider('Number Of Products',1,4)
    has_cr_card=st.selectbox('Has Credit Card',[0,1])
    is_active_member=st.selectbox('Is Active Member',[0,1])

    input_data=pd.DataFrame({
        'CreditScore':[credit_score],
        'Geography':[geography],
        'Gender':[label_encoder_gender.transform([gender])[0]],
        'Age':[age],
        'Tenure':[tenure],
        'Balance':[balance],
        'NumOfProducts':[num_of_products],
        'HasCrCard':[has_cr_card],
        'IsActiveMember':[is_active_member],
    })
    geo_encoded=onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
    geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    input_data=input_data.drop('Geography',axis=1)
    input_data=pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)
    print(input_data.columns.tolist())
    print(input_data)
    input_data_scaled=scaler.transform(input_data)
    prediction=model.predict(input_data_scaled)
    predicted_salary=prediction[0][0]

    st.write(f'Predicted Estimate Salary:{predicted_salary:.2f}')