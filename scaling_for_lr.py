import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("final_data_set2.csv")

X = df.drop(columns=["ARRIVAL_DELAY"])
y = df["ARRIVAL_DELAY"]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Add target column back
X_scaled_df["ARRIVAL_DELAY"] = y.values

# Save single scaled dataset
X_scaled_df.to_csv("final_scaled_data_set2.csv", index=False)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("Scaled dataset saved as final_scaled_data_set2.csv")