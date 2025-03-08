import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Web App Title
# --------------------------
st.title("📊 ERP Sales Prediction System")
st.write("🚀 Predict Sales for 2025 Using Machine Learning")
st.write("Upload your sales data (2023-2024) to forecast sales for 2025.")

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'])

if uploaded_file is not None:
    # Load Dataset
    df = pd.read_csv(uploaded_file)

    # --------------------------
    # Preprocessing
    # --------------------------
    df.drop(['User_ID', 'Product_ID'], axis=1, inplace=True)

    # Encode Categorical Variables
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df['City_Category'] = label_encoder.fit_transform(df['City_Category'])
    df['Stay_In_Current_City_Years'] = label_encoder.fit_transform(df['Stay_In_Current_City_Years'])

    # Split Data for 2023-2024
    df = df[df['Year'].isin([2023, 2024])]

    X = df.drop(['Purchase', 'Year'], axis=1)
    y = df['Purchase']

    # Convert categorical data
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col])

    # Fill missing values
    X.fillna(0, inplace=True)

    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------
    # Train XGBoost Model
    # --------------------------
    st.write("⚙️ Training The Model... Please Wait 5-10 seconds...")
    model = XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6)
    model.fit(X_scaled, y)

    # Predict Sales for 2025
    future_data = X.tail(1)  # Take last available data
    future_sales = pd.DataFrame({
        "Year": [2025],
        "Predicted Sales": model.predict(future_data)[0]
    })

    # Calculate Growth %
    last_year_sales = y.sum()
    predicted_sales = future_sales['Predicted Sales'].sum()
    growth_percent = ((predicted_sales - last_year_sales) / last_year_sales) * 100

    # --------------------------
    # Dashboard - Show Results
    # --------------------------
    st.subheader("✅ Model Performance")
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"💯 **R² Score (Accuracy):** {r2 * 100:.2f}%")
    st.write(f"💸 **Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"📊 **Root Mean Squared Error (RMSE):** {rmse:.2f}")

    # --------------------------
    # Future Sales Prediction
    # --------------------------
    st.subheader("📈 Sales Prediction for 2025")
    st.write(f"💸 **Predicted Sales (2025):** ₹{predicted_sales:,.2f}")
    st.write(f"📊 **Growth % Compared to 2024:** {growth_percent:.2f}%")

    # --------------------------
    # Sales Graph
    # --------------------------
    st.subheader("📊 Sales Graph")
    fig, ax = plt.subplots()
    sns.lineplot(x=[2023, 2024, 2025], y=[last_year_sales, y.sum(), predicted_sales], marker='o')
    plt.xlabel("Year")
    plt.ylabel("Sales")
    plt.title("Sales Trend 2023-2025")
    st.pyplot(fig)

    st.success("✅ Prediction Completed In 5-10 Seconds")
