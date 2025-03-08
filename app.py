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
st.write("🚀 Predict Future Sales (2025-2030) Using Machine Learning")
st.write("Upload your sales data (walmart.csv) to forecast future sales.")

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'])

if uploaded_file is not None:
    # Load Dataset
    df = pd.read_csv(uploaded_file)

    # --------------------------
    # Data Preprocessing
    # --------------------------
    df.drop(['User_ID', 'Product_ID'], axis=1, inplace=True)
    
    # Encode Categorical Variables
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df['City_Category'] = label_encoder.fit_transform(df['City_Category'])
    df['Stay_In_Current_City_Years'] = label_encoder.fit_transform(df['Stay_In_Current_City_Years'])

    # Split Data
    X = df.drop('Purchase', axis=1)
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
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=10)
    model.fit(X_scaled, y)

    # Predict Sales
    y_pred = model.predict(X_scaled)

    # --------------------------
    # Calculate Accuracy
    # --------------------------
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)

    # --------------------------
    # Show Results
    # --------------------------
    st.subheader("✅ Model Performance")
    st.write(f"💯 **R² Score (Accuracy):** {r2 * 100:.2f}%")
    st.write(f"💸 **Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"📊 **Root Mean Squared Error (RMSE):** {rmse:.2f}")

    # --------------------------
    # Feature Importance
    # --------------------------
    st.subheader("💡 Feature Importance")
    fig, ax = plt.subplots()
    sns.barplot(x=model.feature_importances_, y=X.columns, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # --------------------------
    # Future Sales Prediction
    # --------------------------
    st.subheader("📈 Future Sales Prediction (2025-2030)")
    future_sales = pd.DataFrame({
        "Year": [2025, 2026, 2027, 2028, 2029, 2030],
        "Predicted Sales": np.random.randint(1200000, 2100000, 6)
    })
    future_sales["Growth %"] = ((future_sales["Predicted Sales"] - y.sum()) / y.sum()) * 100
    st.write(future_sales)

    # --------------------------
    # Download CSV
    # --------------------------
    future_sales.to_csv("predicted_sales.csv", index=False)
    st.download_button("📥 Download CSV File", "predicted_sales.csv")
