# Flight Arrival Delay Prediction

## Overview

This project focuses on predicting **flight arrival delays** using machine learning models trained on historical flight data. The dataset contains operational flight details such as departure time, airline, airport information, delay causes, and geographical coordinates.

Two models are implemented:

* **Linear Regression**
* **Random Forest Regressor**

The goal is to evaluate how accurately arrival delay can be predicted using available flight features.

---

# Project Structure

```
.
├── convert_categorical.py       # Encodes categorical variables (airline, airports, etc.)
├── dataste_merge.py             # Merges datasets used in preprocessing
├── drop_feature.py              # Removes unnecessary or redundant features
├── fill_missing_values.py       # Handles missing values in the dataset
├── remove_correlated.py         # Removes highly correlated features

├── lr_model.py                  # Linear Regression training script
├── rf_model.py                  # Random Forest training script
├── predict_delay.py             # Script to predict arrival delay using trained models


├── lr_metrics.txt               # Evaluation metrics for Linear Regression
├── rf_metrics.txt               # Evaluation metrics for Random Forest

├── residual_plot_lr.png         # Residual plot for Linear Regression
├── residual_plot_rf.png         # Residual plot for Random Forest

└── README.md                    # Project documentation
```

---

# Dataset Features

The final dataset includes the following features:

* YEAR
* MONTH
* DAY
* DAY_OF_WEEK
* AIRLINE
* ORIGIN_AIRPORT
* DESTINATION_AIRPORT
* SCHEDULED_DEPARTURE
* DEPARTURE_TIME
* DEPARTURE_DELAY
* SCHEDULED_TIME
* DISTANCE
* SCHEDULED_ARRIVAL
* ARRIVAL_TIME
* DIVERTED
* CANCELLED
* AIR_SYSTEM_DELAY
* SECURITY_DELAY
* AIRLINE_DELAY
* LATE_AIRCRAFT_DELAY
* WEATHER_DELAY
* LATITUDE
* LONGITUDE

Target Variable:

```
ARRIVAL_DELAY
```

---

# Data Processing Pipeline

The dataset is processed through several stages:

1. **Dataset Merge**

   * Combines multiple datasets into one unified dataset.

2. **Missing Value Handling**

   * Null values are filled or removed using appropriate techniques.

3. **Categorical Encoding**

   * Categorical features such as airline and airport codes are converted into numerical values.

4. **Feature Cleaning**

   * Redundant and highly correlated features are removed.

5. **Final Dataset Creation**

   * A cleaned dataset ready for machine learning training.

---

# Models Implemented

## Linear Regression

A baseline regression model used to understand linear relationships between flight features and arrival delays.

Metrics evaluated:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² Score

Outputs:

* `lr_model.pkl`
* `lr_metrics.txt`
* `residual_plot_lr.png`

---

## Random Forest Regressor

An ensemble machine learning model that improves prediction accuracy by combining multiple decision trees.

Outputs:

* `rf_model.pkl`
* `rf_metrics.txt`
* `residual_plot_rf.png`

---

# Model Evaluation

Evaluation metrics used:

**MAE (Mean Absolute Error)**
Average prediction error in minutes.

**MSE (Mean Squared Error)**
Measures squared difference between predicted and actual values.

**RMSE (Root Mean Squared Error)**
Provides error magnitude in the same units as delay.

**R² Score**
Measures how well the model explains variance in arrival delay.

---

# Residual Analysis

Residual plots are generated to analyze model performance.

Residual = Actual Delay − Predicted Delay

A well-performing model should produce residuals that are randomly distributed around zero.

---

# Running the Project

## 1. Train Linear Regression Model

```
python lr_model.py
```

Outputs:

* Model file
* Metrics file
* Residual plot

---

## 2. Train Random Forest Model

```
python rf_model.py
```

Outputs:

* Model file
* Metrics file
* Residual plot

---

## 3. Predict Arrival Delay

```
python predict_delay.py
```

This script loads a trained model and predicts the arrival delay for given flight parameters.

---

# Requirements

Install required dependencies:

```
pip install pandas scikit-learn matplotlib joblib
```

---

# Summary

This project demonstrates a complete **machine learning pipeline** including:

* Data preprocessing
* Feature engineering
* Model training
* Model evaluation
* Prediction using saved models

The results show that machine learning models can effectively estimate flight arrival delays based on operational and environmental factors.

---
