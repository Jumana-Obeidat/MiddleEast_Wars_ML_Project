MiddleEast_Wars_ML_Project

Machine Learning project predicting fatalities in Middle East conflicts (2015-2024) using multiple regression models.

Dataset

Download the dataset from Kaggle:
https://www.kaggle.com/willianoliveiragibin/middle-east-country-wars

Note: The dataset is too large to include here. After downloading, place the CSV file in the same directory as the notebook or script.

Project Overview
Data preprocessing: handling missing values, encoding categorical variables, outlier handling, feature engineering
Models trained: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM (tuned), Poisson Regression
Model evaluation using R², MAE, MSE
Feature importance and correlation visualization
How to Run
Install required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm kagglehub
Download the dataset from Kaggle and place the CSV file in the same directory as the notebook/script.
Open and run the notebook/script in your Python environment (e.g., Jupyter, Colab).
Outputs will include preprocessed data, model metrics, feature importance plots, and correlation heatmaps.
Project Structure
dataset/ # Place the downloaded CSV here
notebooks/ # Jupyter notebook(s)
scripts/ # Optional: Python scripts
README.md # This file
requirements.txt # Optional: list of dependencies
.gitignore # Recommended to ignore large files and cache
Notes
The dataset is large (>25MB), so it is not included in the repo.
LightGBM model has hyperparameter tuning using RandomizedSearchCV.
All preprocessing steps are included in the notebook/script: missing value handling, encoding, clipping outliers, feature engineering.
License & Usage

Private Project – All Rights Reserved

This code and associated materials are for personal use only.
Unauthorized use, reproduction, distribution, or publication of any part of this project is strictly prohibited.
Sharing this project or any of its contents without explicit permission is not allowed.
