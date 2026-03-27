import pandas as pd

dataset1 = pd.read_csv("filled_dataset.csv")
dataset2 = pd.read_csv("airports.csv")
dataset1 = dataset1.merge(dataset2, left_on="ORIGIN_AIRPORT", right_on="IATA_CODE", how="left")
print(dataset1)

dataset1.to_csv("merged_dataset.csv", index=False)

