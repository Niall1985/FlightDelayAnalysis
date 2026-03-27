import pandas as pd

dataset = pd.read_csv("final_data_set.csv")
cols = ["DEPARTURE_TIME", "DEPARTURE_DELAY", "SCHEDULED_TIME", "ARRIVAL_TIME", "ARRIVAL_DELAY", "LATITUDE", "LONGITUDE"]
for col in cols:
    dataset[col] = dataset[col].ffill().bfill()
print(dataset)
dataset.to_csv("final_data_set.csv", index=False)