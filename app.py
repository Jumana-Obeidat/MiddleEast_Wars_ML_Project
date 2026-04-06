import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================== PAGE ==================
st.set_page_config(page_title="ML Project", layout="wide")

# ================== STYLE ==================
st.markdown("""
<style>
.main {background-color: #f5f7fa;}
h1 {color: #1f77b4;}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.title("🌍 Middle East Wars Fatalities Prediction")
st.markdown("### 🚀 Machine Learning Dashboard")

# ================== SIDEBAR ==================
st.sidebar.title("📌 Project Info")
st.sidebar.info("""
**Problem:** Predict fatalities in conflicts  
**Type:** Regression  
**Best Model:** LightGBM  
""")

# ================== FILE ==================
uploaded_file = st.file_uploader("📂 Upload dataset", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # ================== PREPROCESS ==================
    df = df.sample(frac=0.3, random_state=42)

    fill_cols = ['actor2', 'inter2', 'admin1', 'admin2', 'admin3']
    for col in fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'])
        df['year'] = df['event_date'].dt.year
        df['month'] = df['event_date'].dt.month
        df['day'] = df['event_date'].dt.day
        df.drop('event_date', axis=1, inplace=True)

    if 'fatalities' not in df.columns:
        st.error("❌ No fatalities column found")
        st.stop()

    X = df.drop('fatalities', axis=1)
    y = np.log1p(df['fatalities'])

    X_num = X.select_dtypes(include=['int64','float64']).fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    # ================== BUTTON ==================
    if st.button("🚀 Train Models"):

        st.success("Models are training...")

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=15),
            "Random Forest": RandomForestRegressor(n_estimators=100),
            "Gradient Boosting": GradientBoostingRegressor(),
            "XGBoost": XGBRegressor(n_estimators=200),
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

        # ================== DASHBOARD ==================
        st.markdown("## 📊 Model Performance")

        best_model = results_df["R2"].idxmax()
        worst_model = results_df["R2"].idxmin()

        col1, col2, col3 = st.columns(3)

        col1.metric("🏆 Best Model", best_model)
        col2.metric("❌ Worst Model", worst_model)
        col3.metric("📈 Best R²", round(results_df["R2"].max(), 3))

        st.bar_chart(results_df["R2"])

        # ================== EXPLANATION ==================
        st.markdown("## 🧠 Model Explanation")

        st.info(f"""
**Best Model: {best_model}**
- Captures complex relationships  
- Works well with large datasets  

**Worst Model: {worst_model}**
- Assumes simple patterns  
- Not suitable for complex data  
""")

        # ================== FEATURE IMPORTANCE ==================
        st.markdown("## 🔍 Feature Importance")

        rf = models["Random Forest"]
        importances = pd.Series(rf.feature_importances_, index=X_num.columns)

        st.bar_chart(importances.sort_values(ascending=False).head(10))

        # ================== INSIGHTS ==================
        st.markdown("## 📌 Insights")

        if 'country' in df.columns:
            top_country = df.groupby('country')['fatalities'].sum().idxmax()
            st.success(f"🔥 Most dangerous country: {top_country}")

        if 'year' in df.columns:
            top_year = df.groupby('year')['fatalities'].sum().idxmax()
            st.success(f"📅 Most dangerous year: {top_year}")

        # ================== PREDICTION ==================
        st.markdown("## 🎯 Make Prediction")

        user_input = []

        for col in X_num.columns[:5]:
            val = st.number_input(f"{col}", value=float(X_num[col].mean()))
            user_input.append(val)

        if st.button("Predict"):

            model = models[best_model]

            input_full = user_input + [0]*(X_scaled.shape[1]-len(user_input))

            pred = model.predict([input_full])

            st.success(f"💀 Predicted Fatalities: {int(np.expm1(pred[0]))}")

else:
    st.warning("⚠️ Please upload dataset to start")
