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

    # Sample for speed
    df = df.sample(frac=0.3, random_state=42)
    
    fill_null_col = ['actor2', 'inter2', 'admin1', 'admin2', 'admin3']
    for col in fill_null_col:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df['year'] = df['event_date'].dt.year
        df['month'] = df['event_date'].dt.month
        df['day'] = df['event_date'].dt.day
        df.drop('event_date', axis=1, inplace=True)

    if 'fatalities' in df.columns:
        X = df.drop('fatalities', axis=1)
        y = np.log1p(df['fatalities'])

        # Numeric columns only
        X_num = X.select_dtypes(include=['int64', 'float64']).fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)

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
            results[name] = {
                "R2": r2_score(y, y_pred),
                "MAE": mean_absolute_error(y, y_pred),
                "MSE": mean_squared_error(y, y_pred)
            }

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset", "Model Training", "Feature Analysis", "Prediction", "Insights"])

        # --- Tab 1: Dataset ---
        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())
            st.subheader("Dataset Info")
            st.write(df.info())
            st.write("Missing values:", df.isnull().sum())

        # --- Tab 2: Model Training ---
        with tab2:
            st.subheader("Model Evaluation Metrics")
            results_df = pd.DataFrame(results).T
            st.dataframe(results_df)

            st.subheader("R2 Comparison")
            st.bar_chart(results_df['R2'])

            best_model_name = results_df['R2'].idxmax()
            worst_model_name = results_df['R2'].idxmin()
            st.write(f"Best model: **{best_model_name}** with R2={results_df['R2'].max():.3f}")
            st.write(f"Worst model: **{worst_model_name}** with R2={results_df['R2'].min():.3f}")

        # --- Tab 3: Feature Analysis ---
        with tab3:
            if "Random Forest" in models:
                rf_model = models["Random Forest"]
                importances = pd.Series(rf_model.feature_importances_, index=X_num.columns)
                st.subheader("Top 15 Feature Importances (Random Forest)")
                st.bar_chart(importances.sort_values(ascending=False).head(15))

            st.subheader("Correlation Heatmap")
            df_encoded = pd.concat([X_num, y.rename('fatalities')], axis=1)
            corr_matrix = df_encoded.corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
            st.pyplot(fig)

        # --- Tab 4: Prediction ---
        with tab4:
            st.subheader("Predict Fatalities")
            input_data = {}
            for col in X_num.columns:
                input_data[col] = st.number_input(f"Input {col}", value=float(X_num[col].mean()))
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            pred_model_name = st.selectbox("Choose model for prediction", list(models.keys()))
            pred_model = models[pred_model_name]
            pred_value = pred_model.predict(input_scaled)
            st.write(f"Predicted fatalities (log-transformed): {pred_value[0]:.3f}")
            st.write(f"Predicted fatalities (original scale): {np.expm1(pred_value[0]):.0f}")

        # --- Tab 5: Insights ---
        with tab5:
            st.subheader("Problem Description")
            st.write("""
            The dataset contains conflict events in the Middle East from 2015-2024, including details like actors, event types, and locations.
            The project predicts the number of fatalities for each event using multiple regression models.
            """)
            st.subheader("Insights")
            st.write(f"- Best model: {best_model_name} achieves the highest R2, indicating it fits the data better.")
            st.write(f"- Worst model: {worst_model_name}, which may underfit or not capture data complexity.")
            st.write("- Log-transforming fatalities helps stabilize variance.")
            st.write("- Feature importance shows which variables most influence fatalities.")
    else:
        st.warning("Column 'fatalities' not found in dataset")
else:
    st.info("Please upload a CSV dataset to start")
