import pandas as pd

df = pd.read_csv("final_data_set.csv")
# df = df.drop(columns=["CITY", "STATE", "COUNTRY", "WHEELS_ON", "WHEELS_OFF", "TAXI_OUT", "AIRPORT", "FLIGHT_NUMBER", "TAIL_NUMBER"])
# df.to_csv("final_data_set.csv", index=False)

print(df.columns)