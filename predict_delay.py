import pandas as pd
import joblib

# Load trained models
lr_model = joblib.load('lr_model.pkl')
rf_model = joblib.load('rf_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')

input_file = "flight_delay_sample_inputs.csv"
df = pd.read_csv(input_file)

lr_predictions = lr_model.predict(df)
rf_predictions = rf_model.predict(df)
xgb_predictions = xgb_model.predict(df)

df["LR_PREDICTED_DELAY"] = lr_predictions.round()
df["RF_PREDICTED_DELAY"] = rf_predictions.round()
df["XGB_PREDICTED_DELAY"] = xgb_predictions.round()

output_file = "flight_delay_predictions.csv"
df.to_csv(output_file, index=False)

print("Predictions saved to:", output_file)

for i in range(len(df)):
    print(f"\nFlight {i+1}")
    print("Linear Regression Delay:", round(lr_predictions[i]), "minutes")
    print("Random Forest Delay:", round(rf_predictions[i]), "minutes")
    print("XGBoost Delay:", round(xgb_predictions[i]), "minutes")