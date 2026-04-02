import pandas as pd
from sklearn.model_selection import train_test_split   
from sklearn.metrics import r2_score
import joblib
df = pd.read_csv('final_data_set.csv')
X = df.drop(columns=["ARRIVAL_DELAY"])
y = df["ARRIVAL_DELAY"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = joblib.load('lr_model.pkl')
rf_model = joblib.load('rf_model.pkl')  
xgb_model = joblib.load('xgb_model.pkl')
lr_pred = lr_model.predict(X_test)
rf_pred = rf_model.predict(X_test)     
xgb_pred = xgb_model.predict(X_test)
lr_r2 = r2_score(y_test, lr_pred)
rf_r2 = r2_score(y_test, rf_pred)
xgb_r2 = r2_score(y_test, xgb_pred)
n = len(X_test)
p = X_test.shape[1]
lr_adj_r2 = 1 - (1 - lr_r2) * (n - 1) / (n - p - 1)
rf_adj_r2 = 1 - (1 - rf_r2) * (n - 1) / (n - p - 1)
xgb_adj_r2 = 1 - (1 - xgb_r2) * (n - 1) / (n - p - 1)
print(f"Linear Regression Adjusted R²: {lr_adj_r2:.4f}")
print(f"Random Forest Adjusted R²: {rf_adj_r2:.4f}")
print(f"XGBoost Adjusted R²: {xgb_adj_r2:.4f}")

with open('adjusted_r2_comparison.txt', 'w') as f:
    f.write(f"Linear Regression Adjusted R²: {lr_adj_r2:.4f}\n")
    f.write(f"Random Forest Adjusted R²: {rf_adj_r2:.4f}\n")
    f.write(f"XGBoost Adjusted R²: {xgb_adj_r2:.4f}\n")
