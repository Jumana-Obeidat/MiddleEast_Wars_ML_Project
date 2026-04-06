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

# ================== PAGE SETUP ==================
st.set_page_config(page_title="Middle East Wars ML Project", layout="wide")
st.title("Middle East Wars Fatalities Prediction (2015-2024)")

# ================== SIDEBAR ==================
st.sidebar.header("About the Project")

st.sidebar.write("""
**Problem:**
Predicting the number of fatalities in Middle East conflict events.

**Why this project?**
To analyze conflict patterns and understand factors affecting casualties.

**Dataset:**
Real-world conflict data (2015–2024) including:
- Actors
- Locations
- Event types
- Fatalities

**Challenges:**
- Large dataset
- Missing values
- Many categorical features
- Outliers
""")

# ================== FILE UPLOAD ==================
uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ================== TABS ==================
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset & Info",
        "Model Training",
        "Feature Analysis",
        "Insights"
    ])

    # ================== TAB 1 ==================
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Basic Info")
        st.write(df.describe())
        st.write("Missing values:")
        st.write(df.isnull().sum())

    # ================== PREPROCESSING ==================
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

    # ================== MODEL ==================
    if 'fatalities' in df.columns:

        X = df.drop('fatalities', axis=1)
        y = np.log1p(df['fatalities'])

        X_num = X.select_dtypes(include=['int64', 'float64']).fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=15, min_samples_leaf=7),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(n_estimators=200, max_depth=6),
            "LightGBM": LGBMRegressor(n_estimators=200),
            "Poisson": PoissonRegressor(max_iter=1000)
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

        results_df = pd.DataFrame(results).T

        # ================== TAB 2 ==================
        with tab2:
            st.subheader("Model Comparison")
            st.dataframe(results_df)

            st.bar_chart(results_df["R2"])

            # BEST & WORST
            best_model = results_df["R2"].idxmax()
            worst_model = results_df["R2"].idxmin()

            st.success(f"Best Model: {best_model}")
            st.error(f"Worst Model: {worst_model}")

            st.write(f"""
**Why is {best_model} the best?**
Because it captures complex patterns and relationships in the data better.

**Why is {worst_model} the worst?**
It struggles with non-linear relationships or the data distribution.
""")

        # ================== TAB 3 ==================
        with tab3:

            st.subheader("Feature Importance (Random Forest)")

            rf_model = models["Random Forest"]
            importances = pd.Series(rf_model.feature_importances_, index=X_num.columns)

            st.bar_chart(importances.sort_values(ascending=False).head(15))

            st.subheader("Correlation Heatmap")

            df_corr = pd.concat([X_num, y.rename('fatalities')], axis=1)
            corr = df_corr.corr()

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # ================== TAB 4 ==================
        with tab4:

            st.subheader("Trend Analysis (Fatalities over Years)")

            if 'year' in df.columns:
                yearly = df.groupby('year')['fatalities'].sum()

                st.line_chart(yearly)

            st.subheader("Top 10 Countries by Fatalities")

            if 'country' in df.columns:
                top_countries = df.groupby('country')['fatalities'].sum().sort_values(ascending=False).head(10)
                st.bar_chart(top_countries)

    else:
        st.warning("Column 'fatalities' not found!")

else:
    st.info("Please upload a CSV dataset to start")
