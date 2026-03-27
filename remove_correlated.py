import numpy as np
import pandas as pd

df = pd.read_csv("final_dataset.csv")
corr = df.corr(numeric_only=True)
print(corr["ARRIVAL_DELAY"].sort_values(ascending=False))