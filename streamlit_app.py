import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.title('UD ML 1st')

st.write('Hellow ML Model')
with st.expander('data'):
 st.write('**Raw Data**')
 df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')
 df
 
 st.write('**X**')
 x_raw = df.drop('species',axis=1)
 x_raw
 
 st.write('**Y**')
 y_raw = df.species
 y_raw

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
    'sex': gender
}
input_df = pd.DataFrame(data, index=[0])
input_penguins = pd.concat([input_df, x_raw], axis=0)

with st.expander('Input features'):
 st.write('**Input penguin:**')
 input_df
 st.write('**Combined data:**')
 input_penguins

 
# Data preparation

# Encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)
x = df_penguins[:1]
input_row = df_penguins[:1]

# Encode y
target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}

def target_encode(val):
    return target_mapper[val]
y = y_raw.apply(target_encode)
with st.expander('Data preparation'):
 st.write('**Encoded X (input penguin)**')
 input_row
 st.write('**Encoded y**')
 y

# Model training and inference
# Train the ML model
clf = RandomForestClassifier()
clf.fit(x,y)

# Apply model to make predictions
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

df_prediction_proba = pd.DataFrame(prediction_proba)
df_prediction_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_prediction_proba.rename(columns={
    0: 'Adelie',
    1: 'Chinstrap',
    2: 'Gentoo'
})

# Display predicted species
st.subheader('Predicted Species')
df_prediction_proba

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguins_species[prediction[0]]))






