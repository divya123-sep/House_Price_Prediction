# 🏘️House Price Prediction🪴💵

This repository contains a machine learning model that predicts house prices based on input features like age and salary. The model uses algorithms like K-Nearest Neighbors (K-NN) and Random Forest to predict the house price. The project also includes a Streamlit app to interact with the model and make predictions.

## Features
- **K-Nearest Neighbors (K-NN)** for classification-based predictions.
- **Random Forest Regressor** for regression-based predictions.
- Streamlit app for easy user input and predictions.
- Visualizations and model performance metrics to evaluate the model.
  
## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/house-price-prediction.git
    cd house-price-prediction
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

Once the Streamlit app is running, you can input your data in the sidebar (age, estimated salary), and the model will predict the house price. You can also see various metrics and download the dataset as a CSV.

## Dataset

The dataset used in this project contains features related to house prices, such as:
- Age of the house
- Estimated salary of the potential buyer

## Models Used

- **K-Nearest Neighbors (K-NN)**: A classification model to predict the house price based on input features.
- **Random Forest Regressor**: A regression model used to predict continuous house prices.

## Model Performance

- **K-NN R² Score**: Measures how well the K-NN model fits the data.
- **Random Forest R² Score**: Evaluates the Random Forest model's accuracy.
- **Mean Absolute Error (MAE)**: Indicates the error in the predictions made by both models.

## Screenshot
- ![Screenshot 2024-12-10 150055](https://github.com/user-attachments/assets/fa23a55f-b6a8-4ff6-a0f3-04120f5e666c)
- ![Screenshot 2024-12-10 150216](https://github.com/user-attachments/assets/3ea4e3ba-f8e2-426c-8056-ff5985134ec2)

