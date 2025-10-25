"""Generating data.csv for perceptron.py"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_POINTS = 100

x1 = np.random.uniform(-1, 1, N_POINTS)
x2 = np.random.uniform(-1, 1, N_POINTS)

y = np.where(x2 > 0.5 * x1 + 0.2, 1, -1)

df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})

df.to_csv("data.csv", sep=";", index=False)

print("CSV generato con 100 punti: data.csv")
print(df.head())
