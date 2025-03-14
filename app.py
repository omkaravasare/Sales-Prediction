import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set page configuration
st.set_page_config(page_title="ERP Sales Prediction System", layout="wide")

# --------------------------
# Web App Title
# --------------------------
st.title("📊 ERP Sales Prediction System")
st.write("🚀 Predict Sales for Next Year Using Machine Learning")
st.write("Upload your sales data (any year) to forecast sales for next year.")

# Function to validate the uploaded file
def validate_data(df):
    required_columns = ['Purchase']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Required columns missing: {', '.join(missing_columns)}")
        return False
        
    if df.empty:
        st.error("❌ Uploaded file contains no data")
        return False
        
    if df['Purchase'].isnull().all():
        st.error("❌ Purchase column contains no valid data")
        return False
        
    return True

# --------------------------
# File Upload
# --------------------------
uploaded_file = st.file_uploader("📁 Upload CSV File", type=['csv'])

if uploaded_file is not None:
    try:
        # Load Dataset
        df = pd.read_csv(uploaded_file)
        
        # Validate the data
        if not validate_data(df):
            st.stop()
            
        # Make a copy of the original data
        original_df = df.copy()
        
        # Display raw data sample
        with st.expander("🔍 View Raw Data Sample"):
            st.dataframe(df.head())
            st.write(f"Total records: {len(df)}")
            st.write(f"Columns: {', '.join(df.columns)}")
        
        # --------------------------
        # Preprocessing
        # --------------------------
        with st.spinner("🔄 Preprocessing data..."):
            # Remove ID columns if they exist
            id_columns = [col for col in df.columns if 'id' in col.lower() or '_id' in col.lower()]
            if id_columns:
                df.drop(id_columns, axis=1, inplace=True)
                st.info(f"ℹ️ Removed ID columns: {', '.join(id_columns)}")
            
            # Handle categorical variables safely
            categorical_cols = []
            for col in df.columns:
                if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                    categorical_cols.append(col)
            
            # Create encoders dictionary to store all encoders
            encoders = {}
            
            for col in categorical_cols:
                if col in df.columns:
                    encoders[col] = LabelEncoder()
                    df[col] = encoders[col].fit_transform(df[col].astype(str))
            
            # Detect date columns
            date_columns = []
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower() or 'year' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        date_columns.append(col)
                    except:
                        pass
            
            if date_columns:
                # Use the first date column found
                date_column = date_columns[0]
                df['Year'] = df[date_column].dt.year
                df['Month'] = df[date_column].dt.month
                df['Day'] = df[date_column].dt.day
                
                # Extract more features from date
                df['DayOfWeek'] = df[date_column].dt.dayofweek
                df['Quarter'] = df[date_column].dt.quarter
                
                st.success(f"✅ Extracted date features from '{date_column}'")
            else:
                # If no date column exists, look for a year column
                year_col = None
                for col in df.columns:
                    if col.lower() == 'year':
                        year_col = col
                        break
                
                if year_col:
                    df['Year'] = df[year_col]
                else:
                    st.warning("⚠️ No date column found. Please add a column with year information.")
                    df['Year'] = st.number_input("Enter base year for the data:", min_value=2000, max_value=2030, value=2023)
            
            # Check if we have enough years for training and testing
            years_present = df['Year'].unique()
            years_present = sorted(years_present)
            
            if len(years_present) < 2:
                st.error("❌ Need at least 2 different years in the data for training and prediction")
                st.stop()
            
            # Use the latest two years
            latest_year = max(years_present)
            previous_year = latest_year - 1
            
            if previous_year not in years_present:
                previous_year = sorted(years_present)[-2]
            
            # Filter data for these years
            year_data = df[df['Year'].isin([previous_year, latest_year])]
            
            if len(year_data) < 100:
                st.warning(f"⚠️ Limited data available ({len(year_data)} records). Predictions may be less accurate.")
            
            # Split data correctly
            df_train = df[df['Year'] == previous_year].copy()
            df_test = df[df['Year'] == latest_year].copy()
            
            if df_train.empty or df_test.empty:
                st.error("❌ Insufficient data for either training or testing year")
                st.stop()
            
            # Extract features and target
            target_col = 'Purchase'
            drop_cols = [target_col, 'Year']
            
            if date_column in df_train.columns:
                drop_cols.append(date_column)
            
            X_train = df_train.drop(drop_cols, axis=1, errors='ignore')
            y_train = df_train[target_col]
            
            X_test = df_test.drop(drop_cols, axis=1, errors='ignore')
            y_test = df_test[target_col] if target_col in df_test.columns else None
            
            # Handle missing values with mean imputation
            for col in X_train.columns:
                if X_train[col].dtype in [np.float64, np.int64]:
                    # For numeric columns, use mean
                    mean_val = X_train[col].mean()
                    X_train[col].fillna(mean_val, inplace=True)
                    X_test[col].fillna(mean_val, inplace=True)
                else:
                    # For other columns, use mode
                    mode_val = X_train[col].mode()[0]
                    X_train[col].fillna(mode_val, inplace=True)
                    X_test[col].fillna(mode_val, inplace=True)
            
            # Feature importance based preprocessing
            feature_cols = X_train.columns.tolist()
            
            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save column names for later
            feature_names = X_train.columns.tolist()
        
        # --------------------------
        # Model Training with Option Selection
        # --------------------------
        st.subheader("⚙️ Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type",
                ["XGBoost Regressor", "Quick Training (Less Accurate)", "Hyperparameter Tuning (More Accurate)"]
            )
        
        with col2:
            if model_type == "Hyperparameter Tuning (More Accurate)":
                tune_time = st.slider("Tuning Time (seconds)", min_value=10, max_value=120, value=30, step=10)
            else:
                tune_time = 0
        
        with st.spinner("⚙️ Training The Model... Please Wait..."):
            if model_type == "Quick Training (Less Accurate)":
                # Quick model with default parameters
                model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
                model.fit(X_train_scaled, y_train)
                
            elif model_type == "Hyperparameter Tuning (More Accurate)":
                # Hyperparameter tuning with time budget
                import time
                
                # Define the parameter grid
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7, 9],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
                
                # Use RandomizedSearchCV to respect time budget
                from sklearn.model_selection import RandomizedSearchCV
                
                base_model = XGBRegressor(random_state=42)
                
                # Adjust n_iter based on time budget
                n_iter = max(5, tune_time // 5)
                
                search = RandomizedSearchCV(
                    base_model, 
                    param_grid, 
                    n_iter=n_iter,
                    scoring='neg_mean_squared_error',
                    cv=3,
                    random_state=42,
                    n_jobs=-1
                )
                
                start_time = time.time()
                search.fit(X_train_scaled, y_train)
                elapsed_time = time.time() - start_time
                
                model = search.best_estimator_
                
                # Display tuning results
                st.success(f"✅ Hyperparameter tuning completed in {elapsed_time:.2f} seconds")
                st.write("Best Parameters:", search.best_params_)
                
            else:  # Default XGBoost
                model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=42)
                model.fit(X_train_scaled, y_train)
            
            # Save the model
            model_filename = "sales_prediction_model.pkl"
            joblib.dump(model, model_filename)
            
            # Predict on training data for evaluation
            y_train_pred = model.predict(X_train_scaled)
            train_r2 = r2_score(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            
            # Predict on test data if available
            if y_test is not None:
                y_test_pred = model.predict(X_test_scaled)
                test_r2 = r2_score(y_test, y_test_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # Predict future sales
            future_sales_pred = model.predict(X_test_scaled).sum()
            last_year_sales = y_train.sum()
            latest_year_sales = y_test.sum() if y_test is not None else None
            
            # Calculate Growth %
            growth_percent = ((future_sales_pred - last_year_sales) / last_year_sales) * 100
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
        
        # --------------------------
        # Dashboard - Show Results
        # --------------------------
        st.subheader("✅ Model Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Score (Training)", f"{train_r2 * 100:.2f}%")
        
        with col2:
            st.metric("MAE (Training)", f"{train_mae:.2f}")
        
        with col3:
            st.metric("RMSE (Training)", f"{train_rmse:.2f}")
        
        if y_test is not None:
            st.subheader("🧪 Test Data Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Score (Test)", f"{test_r2 * 100:.2f}%")
            
            with col2:
                st.metric("MAE (Test)", f"{test_mae:.2f}")
            
            with col3:
                st.metric("RMSE (Test)", f"{test_rmse:.2f}")
        
        # --------------------------
        # Future Sales Prediction
        # --------------------------
        st.subheader("📈 Sales Prediction for Next Year")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                f"Predicted Sales ({latest_year + 1})", 
                f"₹{future_sales_pred:,.2f}",
                f"{growth_percent:.2f}%"
            )
        
        with col2:
            st.metric(
                f"Actual Sales ({previous_year})",
                f"₹{last_year_sales:,.2f}"
            )
            if latest_year_sales is not None:
                st.metric(
                    f"Actual Sales ({latest_year})",
                    f"₹{latest_year_sales:,.2f}"
                )
        
        # --------------------------
        # Improved Sales Graph
        # --------------------------
        st.subheader("📊 Sales Trend Visualization")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            years = [previous_year, latest_year, latest_year + 1]
            sales = [last_year_sales]
            
            if latest_year_sales is not None:
                sales.append(latest_year_sales)
            else:
                sales.append(None)
                
            sales.append(future_sales_pred)
            
            # Create a DataFrame for better plotting
            plot_df = pd.DataFrame({
                'Year': years,
                'Sales': sales,
                'Type': ['Actual', 'Actual' if latest_year_sales is not None else 'Unknown', 'Predicted']
            })
            
            # Plot actual data points
            sns.lineplot(
                data=plot_df[plot_df['Type'] == 'Actual'], 
                x='Year', 
                y='Sales', 
                marker='o', 
                markersize=10,
                color='green', 
                label='Actual'
            )
            
            # Plot predicted data point
            sns.lineplot(
                data=plot_df[plot_df['Type'] == 'Predicted'],
            x='Year', 
                y='Sales', 
                marker='*', 
                markersize=15,
                color='blue', 
                label='Predicted'
            )
            
            # Add trend line
            plot_df_clean = plot_df.dropna(subset=['Sales'])
            sns.regplot(
                data=plot_df_clean,
                x='Year', 
                y='Sales',
                scatter=False,
                color='gray',
                line_kws={'linestyle':'--', 'alpha':0.7}
            )
            
            # Format the plot
            plt.title('📊 Sales Trend (Past vs Future)', fontsize=16)
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Sales (₹)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(years)
            
            # Format y-axis labels with commas for thousands
            import matplotlib.ticker as ticker
            formatter = ticker.StrMethodFormatter('{x:,.0f}')
            plt.gca().yaxis.set_major_formatter(formatter)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Sales Table
            st.subheader("Sales Summary")
            
            table_data = {
                'Year': years,
                'Sales': [f"₹{s:,.2f}" if not pd.isna(s) else "N/A" for s in sales],
                'Type': ['Actual', 'Actual' if latest_year_sales is not None else 'Unknown', 'Predicted']
            }
            sales_table = pd.DataFrame(table_data)
            st.table(sales_table)
            
            # Growth metrics
            st.subheader("Growth Metrics")
            st.metric(
                "Year-over-Year Growth",
                f"{growth_percent:.2f}%",
                delta_color="normal"
            )
            
            # Confidence level based on model performance
            confidence_level = "High" if train_r2 > 0.7 else "Medium" if train_r2 > 0.5 else "Low"
            st.info(f"Prediction Confidence: {confidence_level}")
        
        # --------------------------
        # Feature Importance
        # --------------------------
        st.subheader("🔍 Feature Importance Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, max(5, len(feature_importance) * 0.3)))
            
            # Show top 10 features only if there are many
            if len(feature_importance) > 10:
                plot_importance = feature_importance.head(10).copy()
                plot_title = "Top 10 Feature Importance"
            else:
                plot_importance = feature_importance.copy()
                plot_title = "Feature Importance"
            
            # Sort for horizontal bar chart
            plot_importance = plot_importance.sort_values('Importance')
            
            sns.barplot(
                data=plot_importance,
                y='Feature',
                x='Importance',
                palette='viridis'
            )
            
            plt.title(plot_title, fontsize=14)
            plt.xlabel('Importance Score', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.tight_layout()
            
            st.pyplot(fig)
        
        with col2:
            # Show feature importance table
            st.write("Feature Ranking:")
            importance_table = feature_importance.copy()
            importance_table['Importance'] = importance_table['Importance'].apply(lambda x: f"{x:.4f}")
            st.table(importance_table)
        
        # --------------------------
        # Download Section
        # --------------------------
        st.subheader("📥 Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Option to download the model
            with open(model_filename, 'rb') as f:
                model_download = f.read()
            
            st.download_button(
                label="Download Trained Model",
                data=model_download,
                file_name=model_filename,
                mime="application/octet-stream"
            )
        
        with col2:
            # Generate prediction report
            report_data = {
                "Model Type": model_type,
                "Training Data Year": previous_year,
                "Test Data Year": latest_year,
                "Prediction Year": latest_year + 1,
                "Training Accuracy (R²)": f"{train_r2 * 100:.2f}%",
                "Training RMSE": f"{train_rmse:.2f}",
                "Previous Year Sales": f"₹{last_year_sales:,.2f}",
                "Predicted Next Year Sales": f"₹{future_sales_pred:,.2f}",
                "Predicted Growth": f"{growth_percent:.2f}%",
                "Top Features": ", ".join(feature_importance['Feature'].head(5).tolist())
            }
            
            if y_test is not None:
                report_data["Test Accuracy (R²)"] = f"{test_r2 * 100:.2f}%"
                report_data["Test RMSE"] = f"{test_rmse:.2f}"
            
            # Convert to string format
            report_text = "# ERP Sales Prediction Report\n\n"
            for key, value in report_data.items():
                report_text += f"**{key}:** {value}\n\n"
            
            # Download button for report
            st.download_button(
                label="Download Prediction Report",
                data=report_text,
                file_name="sales_prediction_report.md",
                mime="text/markdown"
            )
        
        # --------------------------
        # What-If Analysis
        # --------------------------
        st.subheader("🧪 What-If Analysis")
        st.write("Adjust parameters to see how they affect the prediction")
        
        # Get the top 3 most important features
        top_features = feature_importance['Feature'].head(3).tolist()
        
        # Create sliders for these features
        col1, col2, col3 = st.columns(3)
        
        what_if_values = {}
        
        # Get the original feature statistics
        feature_stats = {}
        for feature in top_features:
            feature_stats[feature] = {
                'min': X_train[feature].min(),
                'max': X_train[feature].max(),
                'mean': X_train[feature].mean(),
                'default': X_test[feature].mean()  # Use test data mean as default
            }
        
        # Create sliders
        with col1:
            if len(top_features) > 0:
                feature = top_features[0]
                stat = feature_stats[feature]
                what_if_values[feature] = st.slider(
                    f"{feature}", 
                    min_value=float(stat['min']),
                    max_value=float(stat['max']),
                    value=float(stat['default']),
                    format="%.2f"
                )
        
        with col2:
            if len(top_features) > 1:
                feature = top_features[1]
                stat = feature_stats[feature]
                what_if_values[feature] = st.slider(
                    f"{feature}", 
                    min_value=float(stat['min']),
                    max_value=float(stat['max']),
                    value=float(stat['default']),
                    format="%.2f"
                )
        
        with col3:
            if len(top_features) > 2:
                feature = top_features[2]
                stat = feature_stats[feature]
                what_if_values[feature] = st.slider(
                    f"{feature}", 
                    min_value=float(stat['min']),
                    max_value=float(stat['max']),
                    value=float(stat['default']),
                    format="%.2f"
                )
        
        # Create a what-if scenario prediction
        if st.button("Run What-If Analysis"):
            # Create a copy of the test data
            X_what_if = X_test.copy()
            
            # Modify the selected features
            for feature, value in what_if_values.items():
                X_what_if[feature] = value
            
            # Scale the what-if data
            X_what_if_scaled = scaler.transform(X_what_if)
            
            # Make prediction
            what_if_prediction = model.predict(X_what_if_scaled).sum()
            
            # Calculate difference
            difference = what_if_prediction - future_sales_pred
            percentage = (difference / future_sales_pred) * 100
            
            # Display results
            st.subheader("What-If Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Original Prediction",
                    f"₹{future_sales_pred:,.2f}"
                )
            
            with col2:
                st.metric(
                    "What-If Prediction",
                    f"₹{what_if_prediction:,.2f}",
                    f"{percentage:.2f}%"
                )
            
            with col3:
                st.metric(
                    "Difference",
                    f"₹{difference:,.2f}"
                )
            
            # Provide business insights
            st.subheader("Business Insights")
            
            if percentage > 0:
                st.success(f"The what-if scenario shows a potential {percentage:.2f}% increase in sales.")
            else:
                st.warning(f"The what-if scenario shows a potential {abs(percentage):.2f}% decrease in sales.")
            
            # Provide specific insights for each changed feature
            for feature, value in what_if_values.items():
                default_val = feature_stats[feature]['default']
                
                if value > default_val:
                    st.info(f"Increasing '{feature}' from {default_val:.2f} to {value:.2f} contributed to this change.")
                elif value < default_val:
                    st.info(f"Decreasing '{feature}' from {default_val:.2f} to {value:.2f} contributed to this change.")
    
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.write("Please check your data format and try again.")
else:
    # Show some sample data info when no file is uploaded
    st.info("ℹ️ Please upload a CSV file with sales data to start the prediction process.")
    st.write("The file should contain at least the following columns:")
    st.write("- Purchase: the sales amount")
    st.write("- Date or Year: time information")
    st.write("- Other features like: Gender, Age, City, Product Category, etc.")
    
    # Show a sample format
    st.subheader("📋 Sample Format")
    sample_data = pd.DataFrame({
        'User_ID': [101, 102, 103],
        'Product_ID': ['P01', 'P02', 'P03'],
        'Gender': ['M', 'F', 'M'],
        'Age': [25, 30, 45],
        'City_Category': ['A', 'B', 'C'],
        'Stay_In_Current_City_Years': ['1', '2', '3+'],
        'Purchase Date': ['2023-01-15', '2023-02-20', '2023-03-10'],
        'Purchase': [15000, 12000, 18500]
    })
    
    st.dataframe(sample_data)
    
    st.write("You can download this sample data as a starting point:")
    
    # Create a downloadable sample csv
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV",
        data=csv,
        file_name="sample_sales_data.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("ERP Sales Prediction System | Made with Streamlit & XGBoost")
