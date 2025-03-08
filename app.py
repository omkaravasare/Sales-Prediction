import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model, scaler, and selector
@st.cache_resource

def load_model():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    selector = joblib.load('selector.pkl')
    return model, scaler, selector

model, scaler, selector = load_model()

# Load dataset
file_path = "walmart.csv"
df = pd.read_csv(file_path)

# Preprocessing function
def preprocess_data(df):
    drop_cols = ["User_ID", "Product_ID"]
    for col in drop_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["Gender", "City_Category", "Stay_In_Current_City_Years"], drop_first=True)

    # Age mapping
    age_mapping = {'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6}
    df["Age"] = df["Age"].map(age_mapping)

    return df

# Sidebar inputs
st.sidebar.header("Enter Data")
input_data = []
for col in df.drop(columns=["Purchase"]).columns:
    value = st.sidebar.number_input(f"{col}", value=float(df[col].mean()))
    input_data.append(value)

# Prediction
input_data = np.array(input_data).reshape(1, -1)
input_data = scaler.transform(input_data)
input_data = selector.transform(input_data)
prediction = model.predict(input_data)
st.write(f"### Predicted Purchase: ${prediction[0]:.2f}")

# Forecasting
future_data = pd.DataFrame(np.tile(input_data, (365, 1)))
future_prediction = model.predict(future_data)

# Plotting
st.write("## Future Sales Prediction for One Year")
plt.figure(figsize=(10, 5))
plt.plot(range(365), future_prediction)
plt.xlabel("Days Ahead")
plt.ylabel("Predicted Purchase")
plt.title("Sales Prediction for Next Year")
st.pyplot(plt)
