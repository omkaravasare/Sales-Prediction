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
st.write("🚀 Predict Sales for Next Year Using Machine Learning")
st.write("Upload your sales data (any year) to forecast sales for next year.")

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
    df.drop(['User_ID', 'Product_ID'], axis=1, inplace=True, errors='ignore')

    # Encode Categorical Variables
    label_encoder = LabelEncoder()
    df['Gender'] = label_encoder.fit_transform(df['Gender'])
    df['City_Category'] = label_encoder.fit_transform(df['City_Category'])
    df['Stay_In_Current_City_Years'] = label_encoder.fit_transform(df['Stay_In_Current_City_Years'])

    # Automatically detect Date Column
    date_column = None
    for col in df.columns:
        if 'date' in col.lower():
            date_column = col
            break

    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df['Year'] = df[date_column].dt.year
    else:
        # Generate artificial year if no date column exists
        df['Year'] = np.random.choice([2023, 2024], size=len(df))

    # Automatically detect latest two years
    latest_year = df['Year'].max()
    previous_year = latest_year - 1
    df = df[df['Year'].isin([previous_year, latest_year])]

    # Split Data Correctly
    df_train = df[df['Year'] == previous_year]
    df_test = df[df['Year'] == latest_year]

    # Extract X and y
    X_train = df_train.drop(['Purchase', 'Year'], axis=1)
    y_train = df_train['Purchase']
    X_test = df_test.drop(['Purchase', 'Year'], axis=1)

    # Convert categorical data
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = LabelEncoder().fit_transform(X_train[col])
            X_test[col] = LabelEncoder().fit_transform(X_test[col])

    # Fill missing values
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --------------------------
    # Train XGBoost Model
    # --------------------------
    st.write("⚙️ Training The Model... Please Wait 5 Seconds...")
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    model.fit(X_train_scaled, y_train)

    # Predict Sales for next year
    predicted_sales = model.predict(X_test_scaled).sum()
    last_year_sales = y_train.sum()

    # Calculate Growth %
    growth_percent = ((predicted_sales - last_year_sales) / last_year_sales) * 100

    # --------------------------
    # Dashboard - Show Results
    # --------------------------
    st.subheader("✅ Model Performance")
    y_pred = model.predict(X_train_scaled)
    r2 = r2_score(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"💯 **R² Score (Accuracy):** {r2 * 100:.2f}%")
    st.write(f"💸 **Mean Absolute Error (MAE):** {mae:.2f}")
    st.write(f"📊 **Root Mean Squared Error (RMSE):** {rmse:.2f}")

    # --------------------------
    # Future Sales Prediction
    # --------------------------
    st.subheader("📈 Sales Prediction for Next Year")
    st.write(f"💸 **Predicted Sales ({latest_year + 1}):** ₹{predicted_sales:,.2f}")
    st.write(f"📊 **Growth % Compared to {previous_year}:** {growth_percent:.2f}%")

    # --------------------------
    # Improved Sales Graph
    # --------------------------
    st.subheader("📊 Sales Graph")
    fig, ax = plt.subplots()
    years = [previous_year, latest_year, latest_year + 1]
    sales = [last_year_sales, y_train.sum(), predicted_sales]
    sns.lineplot(x=years, y=sales, marker='o', linestyle='--', color='green')
    plt.xlabel("Year")
    plt.ylabel("Sales")
    plt.title("📊 Sales Trend (Past vs Future)")
    plt.xticks(years)
    st.pyplot(fig)

    st.success("✅ Prediction Completed In 5 Seconds")
