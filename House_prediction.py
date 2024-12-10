import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# Increase the number of cells allowed for styling
pd.set_option("styler.render.max_elements", 500000)

# Load dataset
dataset = pd.read_csv(r"C:\Users\divya\OneDrive\Documents\App\House price detection app\House_data.csv")

# Add background image with CSS
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #1e3c72, #2a5298); /* Background theme */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(30, 30, 30, 0.95); /* Dark gray with slight transparency */
        color: white; /* Text color */
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3); /* Optional shadow */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app title and dataset preview
st.title("🏡 House Price Prediction App 💸")
st.write("Explore house price predictions based on user input and machine learning! 🤖")

st.subheader("Dataset Preview 📊")
st.write(dataset.head())

# Define features and target
X = dataset.iloc[:, [2, 3]].values  # Replace column indices as per dataset
y = dataset.iloc[:, -1].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train K-NN model
knn = KNeighborsClassifier(n_neighbors=4, weights="distance", p=1)
knn.fit(X_train, y_train)

# Train Random Forest model
rf = RandomForestRegressor(random_state=0)
rf.fit(X_train, y_train)

# User input fields
st.sidebar.header("Enter Input Features 🔢")
age = st.sidebar.number_input('Enter age:', min_value=18, max_value=100, step=1)
salary = st.sidebar.number_input('Enter estimated salary 💼:', min_value=150000, max_value=200000, step=1000)

# Prediction
if st.sidebar.button('Predict 🔮'):
    input_data = np.array([[age, salary]])
    input_data_scaled = sc.transform(input_data)
    knn_prediction = knn.predict(input_data_scaled)
    rf_prediction = rf.predict(input_data_scaled)

    st.subheader("Prediction Results 🔍")
    st.write(f"**K-NN Predicted Price:** ${knn_prediction[0]:,.2f}")
    st.write(f"**Random Forest Predicted Price:** ${rf_prediction[0]:,.2f}")

# Metrics
st.subheader("Model Performance Metrics 📈")
knn_accuracy = knn.score(X_test, y_test)
rf_accuracy = rf.score(X_test, y_test)
knn_mae = mean_absolute_error(y_test, knn.predict(X_test))
rf_mae = mean_absolute_error(y_test, rf.predict(X_test))

st.write(f"**K-NN R² Score:** {knn_accuracy:.2f} 📉")
st.write(f"**Random Forest R² Score:** {rf_accuracy:.2f} 📊")
st.write(f"**K-NN Mean Absolute Error (MAE):** ${knn_mae:,.2f} 💵")
st.write(f"**Random Forest MAE:** ${rf_mae:,.2f} 💰")

# Data insights
st.subheader("Dataset Insights 🧠")
styled_df = dataset.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='lightcoral')
st.dataframe(styled_df)

# Export Results
if st.button('Download Dataset 📥'):
    output = dataset.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download CSV 📋", data=output, file_name="house_price_dataset.csv", mime="text/csv")

st.sidebar.info("Explore metrics, visualizations, and prediction tools! 🎨📊")
