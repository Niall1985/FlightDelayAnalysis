import matplotlib.pyplot as plt

metrics = ["MAE", "MSE", "RMSE", "R2", "Adj R2"]

linear_regression = [
9.91112951694123,
237.51025764339767,
15.411367805726968,
0.8716984436511174,
0.87
]

random_forest = [
5.945462794745251,
109.12890529575853,
10.44647812881253,
0.9410492475945287,
0.94
]

xgboost = [
5.557383152925011,
87.79879467470424,
9.370101102693836,
0.9525716304779053,
0.95
]

plt.figure()

plt.plot(metrics, linear_regression, marker='o', label="Multiple Regression")
plt.plot(metrics, random_forest, marker='o', label="Random Forest")
plt.plot(metrics, xgboost, marker='o', label="XGBoost")

plt.xlabel("Metrics")
plt.ylabel("Score / Error Value")
plt.title("Model Performance Comparison")

plt.legend()
plt.grid(True)

# Save the image
plt.savefig("model_metrics_comparison.png", dpi=300)

plt.show()