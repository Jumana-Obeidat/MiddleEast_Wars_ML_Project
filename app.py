# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Middle East Wars Prediction", layout="wide")
st.title("Middle East Wars Fatalities Prediction (2015-2024)")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Basic Info")
    st.write(df.info())
    st.write("Missing values:", df.isnull().sum())

    # Example preprocessing (adapt from your notebook)
    df = df.sample(frac=0.3, random_state=42)  # optional for speed
    fill_null_col = ['actor2', 'inter2', 'admin1', 'admin2', 'admin3']
    for col in fill_null_col:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Feature extraction example
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df['year'] = df['event_date'].dt.year
        df['month'] = df['event_date'].dt.month
        df['day'] = df['event_date'].dt.day
        df.drop('event_date', axis=1, inplace=True)

    # Prepare X and y
    if 'fatalities' in df.columns:
        X = df.drop('fatalities', axis=1)
        y = np.log1p(df['fatalities'])  # log-transform as in your notebook

        # For simplicity, use numeric columns only for Streamlit demo
        X_num = X.select_dtypes(include=['int64', 'float64']).fillna(0)

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)

        # Train models
        st.subheader("Training Models (may take some seconds)...")
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=15, min_samples_leaf=7, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1),
            "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.1),
            "Poisson Regression": PoissonRegressor(alpha=1.0, max_iter=1000)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            results[name] = {"R2": r2, "MAE": mae, "MSE": mse}

        # Show results
        st.subheader("Model Evaluation Metrics")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        # Bar chart
        st.subheader("R2 Comparison")
        st.bar_chart(results_df['R2'])

        # Feature importance (for Random Forest as example)
        if "Random Forest" in models:
            rf_model = models["Random Forest"]
            importances = pd.Series(rf_model.feature_importances_, index=X_num.columns)
            st.subheader("Top 15 Feature Importances (Random Forest)")
            st.bar_chart(importances.sort_values(ascending=False).head(15))

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        df_encoded = pd.concat([X_num, y.rename('fatalities')], axis=1)
        corr_matrix = df_encoded.corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Column 'fatalities' not found in dataset")
else:
    st.info("Please upload a CSV dataset to start")
