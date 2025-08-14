
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False

# Set page config and styling
st.set_page_config(page_title="Melbourne House Price Predictor", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f7f9;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Melbourne House Price Prediction App")
st.write("An interactive machine learning web app to predict house prices with multiple models, EDA, and hyperparameter tuning.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("melb_data.csv")
    df = df.dropna(subset=["Price"])
    df = df.fillna(df.median(numeric_only=True))  # fill numeric NaNs with median
    df = df.fillna("Unknown")  # fill categorical NaNs with "Unknown"
    return df

df = load_data()

# EDA Section
st.header("Exploratory Data Analysis")
fig1 = px.histogram(df, x="Price", nbins=50, title="House Price Distribution", color_discrete_sequence=["#636EFA"])
st.plotly_chart(fig1, use_container_width=True)

if "Rooms" in df.columns:
    fig2 = px.box(df, x="Rooms", y="Price", title="Price vs Number of Rooms", color="Rooms")
    st.plotly_chart(fig2, use_container_width=True)

# Feature selection
features = ["Rooms", "Distance", "Bedroom2", "Bathroom", "Landsize", "BuildingArea", "YearBuilt", "Latitude", "Longitude"]
X = df[features]
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model selection
st.sidebar.header("Model & Parameters")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["Linear Regression", "Random Forest", "XGBoost"] if xgb_available else ["Linear Regression", "Random Forest"]
)

# Preprocessor
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, features)])

# Define models
if model_choice == "Linear Regression":
    model = LinearRegression()
    param_grid = {"model__fit_intercept": [True, False]}

elif model_choice == "Random Forest":
    model = RandomForestRegressor(random_state=42)
    param_grid = {"model__n_estimators": [50, 100], "model__max_depth": [None, 10, 20]}

elif model_choice == "XGBoost" and xgb_available:
    model = XGBRegressor(random_state=42)
    param_grid = {"model__n_estimators": [50, 100], "model__max_depth": [3, 5]}

# Pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Hyperparameter tuning
st.sidebar.subheader("Hyperparameter Optimization")
if st.sidebar.button("Run GridSearchCV"):
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="r2", n_jobs=-1)
    grid.fit(X_train, y_train)
    st.write("**Best Parameters:**", grid.best_params_)
    pipeline = grid.best_estimator_

# Train model
pipeline.fit(X_train, y_train)

# Predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

st.subheader("Model Performance")
col1, col2 = st.columns(2)
col1.metric("Train R²", f"{train_r2:.3f}")
col2.metric("Test R²", f"{test_r2:.3f}")

# Performance comparison chart
results_df = pd.DataFrame({
    "Dataset": ["Train", "Test"],
    "R2 Score": [train_r2, test_r2]
})
fig3 = px.bar(results_df, x="Dataset", y="R2 Score", color="Dataset", title="Train vs Test R² Score",
              text="R2 Score", color_discrete_sequence=["#00CC96", "#EF553B"])
st.plotly_chart(fig3, use_container_width=True)

# Prediction form
st.header("Predict House Price")
with st.form("prediction_form"):
    input_data = {}
    for feature in features:
        input_data[feature] = st.number_input(f"Enter {feature}", value=float(X[feature].median()))
    submitted = st.form_submit_button("Predict Price")
    if submitted:
        input_df = pd.DataFrame([input_data])
        pred_price = pipeline.predict(input_df)[0]
        st.success(f"Predicted Price: ${pred_price:,.2f}")
