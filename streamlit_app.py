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
