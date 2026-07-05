import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle
import churn
import salary
st.set_page_config(page_title="Deep Learning Project",layout="centered")
if "page" not in st.session_state:
    st.session_state.page="Home"
if st.session_state.page=="Home":
    st.title("Deep Learning Projects")
    st.write("Choose a project to continue")
    col1,col2=st.columns(2)
    with col1:
        if st.button("Customer Churn Prediction",use_container_width=True):
            st.session_state.page="Churn"
            st.rerun()
    with col2:
        if st.button("Salary Prediction",use_container_width=True):
            st.session_state.page="Salary"
            st.rerun()
elif st.session_state.page=="Churn":
    if st.button("Back"):
        st.session_state.page="Home"
        st.rerun()
    churn.run()
elif st.session_state.page=="Salary":
    if st.button("Back"):
        st.session_state.page="Home"
        st.rerun()
    salary.run()