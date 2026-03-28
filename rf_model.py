import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("final_data_set.csv")

print(df)
X = df.drop(columns=["ARRIVAL_DELAY"])
y = df["ARRIVAL_DELAY"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = RandomForestRegressor(n_estimators=100, random_state=42)
model1.fit(X_train, y_train)

joblib.dump(model1, 'rf_model.pkl')

pred = model1.predict(X_test)

mae = mean_absolute_error(y_test, pred)
mse = mean_squared_error(y_test, pred)
rmse = root_mean_squared_error(y_test, pred)
r2 = r2_score(y_test, pred)

with open('rf_metrics.txt', 'w') as f:
    f.write(f"MAE: {mae}\n")
    f.write(f"MSE: {mse}\n")
    f.write(f"RMSE: {rmse}\n")
    f.write(f"R2: {r2}\n")
    
print("MAE: ", mae)
print("MSE: ", mse)
print("RMSE: ", rmse)
print("R2: ", r2)

residuals = y_test - pred
plt.scatter(pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')     
plt.xlabel('Predicted Values')
plt.ylabel('Residuals') 
plt.title('Residual Plot')
plt.savefig('residual_plot_rf.png')
plt.show()