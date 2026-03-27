from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("final_data_set.csv")
label_encoder = LabelEncoder()  

cat_cols = ["AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]
for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col].astype(str))
df.to_csv("final_data_set.csv", index=False)  
