import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title('🧊 Penguin Species Classifier - UD ML 1st')

st.write('Hello! This is a simple ML model to predict Penguin species.')

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/penguins_cleaned.csv')

# Display raw data
with st.expander('📊 Data'):
    st.write('**Raw Data**')
    st.dataframe(df)

    st.write('**X (Features)**')
    x_raw = df.drop('species', axis=1)
    st.dataframe(x_raw)

    st.write('**Y (Labels)**')
    y_raw = df['species']
    st.dataframe(y_raw)

# Data visualization
with st.expander('📈 Data Visualizer'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')

# Sidebar for user input
with st.sidebar:
    st.header('🔧 Input Features')
    island = st.selectbox('Island', ['Biscoe', 'Dream', 'Torgersen'])
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
    gender = st.selectbox('Gender', ['male', 'female'])

# Create a DataFrame for the user input
input_df = pd.DataFrame({
    'island': [island],
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [gender]
})

with st.expander('📥 Input Features'):
    st.write('**User Input:**')
    st.dataframe(input_df)

# Feature encoding
encode = ['island', 'sex']
x_encoded = pd.get_dummies(x_raw, prefix=encode)
input_encoded = pd.get_dummies(input_df, prefix=encode)

# Align input features with training features
input_encoded = input_encoded.reindex(columns=x_encoded.columns, fill_value=0)

# Encode target labels
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y = y_raw.map(target_mapper)

with st.expander('🧪 Encoded Data'):
    st.write('**Encoded X (Training):**')
    st.dataframe(x_encoded.head())
    st.write('**Encoded y:**')
    st.dataframe(y)

# Train model
clf = RandomForestClassifier()
clf.fit(x_encoded, y)

# Prediction
prediction = clf.predict(input_encoded)
prediction_proba = clf.predict_proba(input_encoded)

# Prepare prediction output
penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
predicted_species = penguin_species[prediction[0]]

# Display prediction
st.subheader('🔍 Predicted Species')
st.success(f'The predicted species is: **{predicted_species}**')


# Display predicted species st.subheader('Predicted Species')



# Display prediction probabilities
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

st.write('**Prediction Probabilities:**')
st.dataframe(df_prediction_proba)
