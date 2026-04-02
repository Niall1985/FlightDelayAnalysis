import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

df = pd.read_csv("final_data_set.csv")


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR = Q3 - Q1

iqr_outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))

iqr_counts = iqr_outliers.sum()

print("IQR Outliers per column:")
print(iqr_counts)


numeric_df = df.select_dtypes(include=[np.number])

z_scores = np.abs(zscore(numeric_df))

z_outliers = (z_scores > 3)

z_counts = pd.Series(z_outliers.sum(axis=0), index=numeric_df.columns)

print("\nZ-Score Outliers per column:")
print(z_counts)

with open("outlier_results.txt", "w") as f:
    f.write("IQR Outliers per column:\n")
    f.write(iqr_counts.to_string())
    f.write("\n\nZ-Score Outliers per column:\n")
    f.write(z_counts.to_string())

fig, ax = plt.subplots(2, 1, figsize=(12, 10))

ax[0].bar(iqr_counts.index, iqr_counts.values)
ax[0].set_title("IQR Outliers per Column")
ax[0].set_ylabel("Number of Outliers")
ax[0].tick_params(axis='x', rotation=90)

ax[1].bar(z_counts.index, z_counts.values)
ax[1].set_title("Z-Score Outliers per Column")
ax[1].set_ylabel("Number of Outliers")
ax[1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig("outlier_comparison.png")
plt.show()