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

# Tabs
tabs = st.tabs(["Dataset", "Preprocessing", "Model Training", "Model Comparison", "Insights"])

# ----------------- Dataset -----------------
with tabs[0]:
    uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.subheader("Basic Info")
        st.write(df.info())
        st.write("Missing values:", df.isnull().sum())
    else:
        st.info("Please upload a CSV dataset to start")

# ----------------- Preprocessing -----------------
with tabs[1]:
    if uploaded_file:
        df_sample = df.sample(frac=0.3, random_state=42)
        fill_null_col = ['actor2', 'inter2', 'admin1', 'admin2', 'admin3']
        for col in fill_null_col:
            if col in df_sample.columns:
                df_sample[col] = df_sample[col].fillna('Unknown')

        if 'event_date' in df_sample.columns:
            df_sample['event_date'] = pd.to_datetime(df_sample['event_date'])
            df_sample['year'] = df_sample['event_date'].dt.year
            df_sample['month'] = df_sample['event_date'].dt.month
            df_sample['day'] = df_sample['event_date'].dt.day
            df_sample.drop('event_date', axis=1, inplace=True)

        st.subheader("Preprocessed Data Preview")
        st.dataframe(df_sample.head())
    else:
        st.warning("Upload dataset first")

# ----------------- Model Training -----------------
with tabs[2]:
    if uploaded_file and 'fatalities' in df_sample.columns:
        X = df_sample.drop('fatalities', axis=1)
        y = np.log1p(df_sample['fatalities'])

        X_num = X.select_dtypes(include=['int64', 'float64']).fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)

        st.subheader("Training Models (may take a few seconds)...")
        trained_models = {}
        results = {}
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=15, min_samples_leaf=7, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1),
            "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.1),
            "Poisson Regression": PoissonRegressor(alpha=1.0, max_iter=1000)
        }

        for name, model in models.items():
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            results[name] = {"R2": r2, "MAE": mae, "MSE": mse}
            trained_models[name] = model

        st.subheader("Training Completed")
    else:
        st.warning("Upload dataset first or check 'fatalities' column")

# ----------------- Model Comparison -----------------
with tabs[3]:
    if results:
        results_df = pd.DataFrame(results).T

        st.write("### Bar Chart of R²")
        st.bar_chart(results_df['R2'])

        st.write("### Line Chart of R² (different colors per model)")
        fig, ax = plt.subplots(figsize=(10,5))
        colors = ['blue','green','red','purple','orange','brown','cyan']
        for i, col in enumerate(results_df.index):
            ax.plot(col, results_df.loc[col,'R2'], marker='o', linestyle='-', color=colors[i], label=col)
        ax.set_ylabel("R²")
        ax.set_xlabel("Model")
        ax.set_title("R² Comparison Across Models")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        best_model = results_df['R2'].idxmax()
        worst_model = results_df['R2'].idxmin()
        st.write(f"**Best Model:** {best_model} (R²={results_df['R2'].max():.2f})")
        st.write(f"**Worst Model:** {worst_model} (R²={results_df['R2'].min():.2f})")

        st.write("### Top 10 Feature Importances (Random Forest)")
        rf_model = trained_models.get("Random Forest")
        if rf_model:
            importances = pd.Series(rf_model.feature_importances_, index=X_num.columns).sort_values(ascending=False)
            st.bar_chart(importances.head(10))
    else:
        st.warning("Train models first")

# ----------------- Insights -----------------
with tabs[4]:
    st.subheader("Insights from the Project")
    st.write("""
    - **Data Type:** Middle East conflicts (2015-2024), fatalities, actors, region, event type.
    - **Problem Solved:** Predicting the number of fatalities in conflicts using ML models.
    - **Best Model:** Based on R² score.
    - **Worst Model:** Based on R² score.
    - **Why:** Helps decision makers understand risk, severity, and key influencing factors.
    - **Additional Notes:** Interactive visualization of features, comparison across models, easy dataset upload.
    """)
