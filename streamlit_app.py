import streamlit as st
import pandas as pd

st.title('UD ML 1st')

st.write('Hellow ML Model')

df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
df
