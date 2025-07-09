import streamlit as st
import pandas as pd

st.title('UD ML 1st')

st.write('Hellow ML Model')
with st.expander('data'):
 st.write('**Raw Data**')
 df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
 df
 
 st.write('**X**')
 x = df.drop('species',axis=1)
 x
 
 st.write('**Y**')
 y = df.species
 y

with st.expander('Data Visulizer'):
 st.scatter_chart(data=df,x='bill_length_mm',y='body_mass_g',color='species')

# Data preparations
with st.sidebar:
 st.header('Input features')
 island = st.selectbox('Island', ['Biscoe', 'Dream', 'Torgersen'])
 bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
 bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
 flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
 body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
 gender = st.selectbox('Gender', ['male', 'female'])

# Create a DataFrame for the input features
data = {
    'island': island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'gender': gender
}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, X], axis=0)

with st.expander('Input features'):
 st.write('**Input penguin:**')
 input_df
 st.write('**Combined data:**')
 input_penguins

