# Melbourne-House-Price-Prediction-Project

## 1. Business Objective
Predict the selling prices of houses in Melbourne to assist buyers, sellers, and real estate agencies in making informed decisions.

## 2. Business Problem
Given the property features, predict the market value of a house as accurately as possible.

## 3. Constraints
- **Maximum**: Achieve the highest possible R² score on test data without overfitting.
- **Minimum**: Maintain interpretability and reasonable training time.

---

## Dataset
Dataset Description

The dataset contains historical records of residential property sales in Melbourne.

Key Features include:

Rooms: Number of rooms in the property

Distance: Distance from the Central Business District in kilometers

Bathroom: Number of bathrooms

Car: Number of car spaces

Landsize: Land size in square meters

BuildingArea: Area of the building in square meters

YearBuilt: Year the property was built

Propertycount: Number of properties in the suburb

Latitude and Longitude: Geographical coordinates

Suburb, Type, Method, SellerG, CouncilArea, Regionname: Categorical information about location, property type, sales method, seller group, and region

Month and Year: Derived from sale date to capture seasonality effects

Price: Target variable representing the sale price of the property

Target variable: **Price**

---

## Models and Performance

### 1. Linear Regression
- **Train R²**: 0.504
- **Test R²**: 0.487

### 2. Random Forest Regressor
- **Train R²**: 0.969
- **Test R²**: 0.804

### 3. XGBoost Regressor
- **Train R²**: 0.907
- **Test R²**: 0.818

---

## Features

EDA and visualizations using Plotly and Seaborn

Model training and evaluation with train/test accuracy and error metrics

Hyperparameter optimization using GridSearchCV

Interactive prediction with user input via Streamlit sidebar

Multiple model options including Linear Regression, Random Forest, and XGBoost

## Tech Stack

Python – Data processing and modeling

Pandas, NumPy – Data manipulation

Scikit-learn – Model building and preprocessing

XGBoost – Advanced boosting algorithm

Plotly, Seaborn, Matplotlib – Data visualization

Streamlit – Web app deployment

## Key Insights
-XGBoost had the best generalization with Test R² = 0.818.

-Random Forest performed almost as well but was slightly overfitted.

-Linear Regression is not suitable for this dataset due to its complexity and non-linearity.
- Random Forest and XGBoost models significantly outperform Linear Regression.
- XGBoost achieves the best balance between train and test accuracy.

---

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run melbourne_price_app.py
   ```
3. Use the "EDA & Insights" section for data exploration.
4. Use "Train & Evaluate" to train models and compare performance.
5. Use "Predict" to estimate prices for new house data.

---
